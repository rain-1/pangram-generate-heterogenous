"""Update an authorized Hub dataset from a checked release, guarding its parent."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


def publish(folder, repo_id, expected_parent, receipt):
    release=json.loads((folder/'manifest/release.json').read_text())
    splits=json.loads((folder/'manifest/splits.json').read_text())
    assert release['repo_id']==repo_id
    for line in (folder/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1)
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
    # Validate the exact view readers use and the removal across every config.
    for config,expected in splits.items():
        data=load_dataset(str(folder),name=config)
        assert {s:len(rows) for s,rows in data.items()}==expected
        assert all('source_text_sha256' not in rows.column_names for rows in data.values())
    api=HfApi();before=api.dataset_info(repo_id)
    assert before.sha==expected_parent,'Hub head changed; inspect the intervening update before publishing.'
    assert not before.private,'The authorized dataset is expected to remain public.'
    info=api.upload_folder(repo_id=repo_id,repo_type='dataset',folder_path=str(folder),parent_commit=expected_parent,
        commit_message=f"Release v{release['release_version']}: {release['mixed_documents']} mixed documents with audited model expansions",
        commit_description='Adds audited generation cohorts and matched controls; preserves earlier document IDs, text and splits. All current Parquet configurations omit source_text_sha256.',
        ignore_patterns=['**/__pycache__/**','**/*.pyc'])
    print('uploaded_commit',info.oid,flush=True)
    api.create_tag(repo_id,tag=f"v{release['release_version']}",revision=info.oid,repo_type='dataset',exist_ok=True)
    remote=api.dataset_info(repo_id,revision=info.oid,files_metadata=True)
    assert not remote.private
    files={f.rfilename:f for f in remote.siblings};verified=0
    for path in folder.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        name=path.relative_to(folder).as_posix();entry=files[name];content=path.read_bytes()
        assert entry.size==len(content),name
        if entry.lfs:assert entry.lfs.sha256==hashlib.sha256(content).hexdigest(),name
        else:assert entry.blob_id==hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest(),name
        verified+=1
    downloaded=Path(hf_hub_download(repo_id,'manifest/release.json',repo_type='dataset',revision=info.oid))
    assert downloaded.read_bytes()==(folder/'manifest/release.json').read_bytes()
    loaded={}
    for config,expected in splits.items():
        data=load_dataset(repo_id,config,revision=info.oid)
        loaded[config]={s:len(rows) for s,rows in data.items()}
        assert loaded[config]==expected
        assert all('source_text_sha256' not in rows.column_names for rows in data.values())
        for rows in data.values():
            for row in rows:
                assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
                cursor=0
                for span in row['spans']:
                    assert span['start']==cursor
                    if span['label']=='human':
                        assert row['text'][span['start']:span['end']]==row['source_text'][span['source_start']:span['source_end']]
                    cursor=span['end']
                assert cursor==len(row['text'])
        print('remote_config_verified',config,loaded[config],flush=True)
    report={'repo_id':repo_id,'url':f'https://huggingface.co/datasets/{repo_id}',
        'version':release['release_version'],'public':True,'previous_commit':expected_parent,'commit':info.oid,
        'verified_at':datetime.now(timezone.utc).isoformat(),'remote_files_hash_verified':verified,
        'remote_config_counts':loaded,'removed_columns':['source_text_sha256']}
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--repo-id',default='open-text-detector/heterogeneous-ai-spans')
    p.add_argument('--expected-parent',required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();publish(a.folder.resolve(),a.repo_id,a.expected_parent,a.receipt)
