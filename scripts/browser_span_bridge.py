"""Local review form for copying browser-visible generations into the pilot.

All claude.ai interaction stays in the browser. This loopback server only
displays our prompts and saves text submitted through its ordinary HTML form.
"""
from __future__ import annotations
import argparse
import base64
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from run_span_pilot import assemble, export_browser, import_browser
from heterogeneous.pipeline import load_plan


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scope', choices=['smoke','all'], default='smoke'); p.add_argument('--port', type=int, default=8766)
    p.add_argument('--capture-only',action='store_true')
    args=p.parse_args(); run=args.run.resolve()
    renderer=ThreadPoolExecutor(max_workers=1)
    def render_saved(future):
        try:future.result()
        except Exception as error:
            print(json.dumps({'assembly_error':str(error)}),flush=True)
    def pending():
        if args.capture_only:return json.loads((run/f'browser-{args.scope}-queue.json').read_text())['batches']
        return export_browser(run,args.scope)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path in ['/review','/verification']:
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers()
                self.wfile.write((run/('verification-15.html' if path=='/verification' else 'review.html')).read_bytes());return
            if path=='/proof':
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers()
                self.wfile.write(b'<h1>Save verification screenshot</h1><form method="post" action="/proof"><label>Screenshot image base64<textarea name="image" required></textarea></label><button>Save screenshot</button></form>');return
            batches=pending()
            requested=parse_qs(urlsplit(self.path).query).get('request',[None])[0]
            b=next((b for b in batches if b['id']==requested),batches[0] if batches else None)
            page='''<!doctype html><meta charset="utf-8"><title>Opus 3 browser pilot capture</title>
<style>body{max-width:1050px;margin:30px auto;font:16px/1.6 system-ui;color:#20332b;background:#f6f7f2}textarea,input{display:block;width:100%;box-sizing:border-box;margin:8px 0 20px;padding:10px}textarea{height:220px}button{padding:12px 22px;background:#17623a;color:white;border:0;border-radius:8px}small{color:#627168}</style>
<h1>Opus 3 browser pilot</h1><p>Prompts and responses are transferred through the visible browser interface. No claude.ai private endpoints are used.</p>'''
            queue=json.loads((run/f'browser-{args.scope}-queue.json').read_text())
            page+=f'<p>{len(batches)} ready browser blocks in {escape(args.scope)} scope; {len(queue.get("missing_briefs",[]))} blocks await briefs or retry resolution. <a href="/review">Review labelled documents</a></p>'
            plan=load_plan(run)
            page+='<p>Complete documents cached: '
            for name in ['haiku','sonnet','opus','opus3']:
                count=sum(all((run/'cache'/'generation'/name/f"{block['id']}.json").exists() for block in variant['blocks']) for variant in plan['variants'] if variant['generator_names']==[name])
                page+=f'<strong>{escape(name)} {count}/100</strong> &nbsp; '
            page+='</p>'
            saved=parse_qs(urlsplit(self.path).query).get('saved',[None])[0]
            if saved:page+=f'<p>Saved and validated request <code>{escape(saved)}</code>.</p>'
            if batches:
                page += '<nav><p>Ready requests: ' + ' '.join(
                    f'<a href="/?request={item["id"]}">Open request {item["id"]}</a>'
                    for item in batches[:8]) + '</p></nav>'
            if b:
                page+=f'''<p>Source collection: {escape(b['tasks'][0]['dataset'])}</p>
<form method="post" action="/capture"><label>Request ID<input name="request_id" readonly value="{b['id']}"></label>
<label>Generation prompt<textarea name="prompt" readonly>{escape(b['prompt'])}</textarea></label>
<label>Claude chat URL<input name="chat_url" required></label>
<label>Generated response<textarea name="response" required></textarea></label>
<label>Screenshot image base64 (optional)<textarea name="screenshot"></textarea></label>
<button type="submit">Validate and save response</button></form>'''
            else:page+='<p>No pending requests with ready briefs. Refresh after the brief worker completes.</p>'
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(page.encode())
        def do_POST(self):
            if self.path=='/proof':
                n=int(self.headers.get('Content-Length',0))
                if n>12_000_000:self.send_error(413);return
                data=base64.b64decode(parse_qs(self.rfile.read(n).decode())['image'][0],validate=True)
                extension='png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'jpg' if data.startswith(b'\xff\xd8\xff') else None
                if not extension:self.send_error(422);return
                (run/f'verification-overview.{extension}').write_bytes(data)
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers()
                self.wfile.write(b'<h1>Verification screenshot saved</h1>');return
            if self.path!='/capture':self.send_error(404);return
            n=int(self.headers.get('Content-Length',0))
            if n>12_000_000:self.send_error(413);return
            f=parse_qs(self.rfile.read(n).decode(),keep_blank_values=True)
            try:
                batches=pending()
                batch=next(b for b in batches if b['id']==f['request_id'][0])
                if f['prompt'][0].replace('\r\n','\n')!=batch['prompt']:raise ValueError('Submitted prompt differs from frozen generation prompt')
                response=run/'browser-captures'/f"{batch['id']}.txt";response.parent.mkdir(parents=True,exist_ok=True)
                response.write_text(f['response'][0].replace('\r\n','\n'),encoding='utf-8')
                if args.capture_only:
                    (run/'browser-captures'/f"{batch['id']}.json").write_text(json.dumps({'request':batch,'raw_response':response.read_text(),'chat_url':f['chat_url'][0],
                        'status':'quarantined_before_prompt_improvements','selected_ui_model':'Opus 3'},ensure_ascii=False,indent=2))
                else:import_browser(run,run/f'browser-{args.scope}-queue.json',batch['id'],response,f['chat_url'][0])
                if f.get('screenshot',[''])[0]:
                    data=base64.b64decode(f['screenshot'][0],validate=True)
                    extension='png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'jpg' if data.startswith(b'\xff\xd8\xff') else None
                    if not extension:raise ValueError('Screenshot is not a supported image')
                    (run/'browser-captures'/f"{batch['id']}.{extension}").write_bytes(data)
                if not args.capture_only:
                    # Acknowledge a validated cache promptly; serialize export updates.
                    renderer.submit(assemble,run).add_done_callback(render_saved)
            except Exception as e:
                self.send_response(422);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers()
                self.wfile.write(f'<h1>Capture rejected</h1><pre>{escape(str(e))}</pre><a href="/">Return to queue</a>'.encode());return
            if args.capture_only:
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers()
                self.wfile.write(b'<h1>Browser evidence saved</h1><p>This preliminary result is quarantined for quality review.</p>');return
            self.send_response(303);self.send_header('Location',f"/?saved={batch['id']}");self.end_headers()
        def log_message(self,*args):pass
    print(f'Local capture form: http://127.0.0.1:{args.port}/',flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
