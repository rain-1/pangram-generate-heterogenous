from __future__ import annotations

from .core import validate_partition


def boundaries(spans: list[dict]) -> list[int]:
    return [right["start"] for left, right in zip(spans, spans[1:]) if left["label"] != right["label"]]


def matched_boundaries(gold: list[int], predicted: list[int], tolerance: int) -> int:
    i = j = matched = 0
    while i < len(gold) and j < len(predicted):
        if abs(gold[i] - predicted[j]) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif gold[i] < predicted[j]:
            i += 1
        else:
            j += 1
    return matched


def scores(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(gold: list[dict], predictions: list[dict], tolerance: int = 20) -> dict:
    if not gold:
        raise ValueError("gold dataset is empty")
    if len({r["id"] for r in gold}) != len(gold):
        raise ValueError("duplicate gold ids")
    predicted = {r["id"]: r for r in predictions}
    if len(predicted) != len(predictions) or set(predicted) != {r["id"] for r in gold}:
        raise ValueError("predictions must contain exactly one result for every gold id")
    tp = fp = fn = tn = matched = gold_boundaries = predicted_boundaries = 0
    errors = []
    human_docs = human_doc_false_positives = 0
    for record in gold:
        prediction = predicted[record["id"]]
        if prediction.get("text_sha256") != record["text_sha256"]:
            raise ValueError(f"{record['id']}: predicted text hash differs from gold")
        actual, proposed = record["spans"], prediction["spans"]
        validate_partition(record["text"], actual)
        validate_partition(record["text"], proposed)
        i = j = 0
        while i < len(actual) and j < len(proposed):
            a, p = actual[i], proposed[j]
            overlap = min(a["end"], p["end"]) - max(a["start"], p["start"])
            if a["label"] == "ai" and p["label"] == "ai":
                tp += overlap
            elif a["label"] == "human" and p["label"] == "ai":
                fp += overlap
            elif a["label"] == "ai":
                fn += overlap
            else:
                tn += overlap
            if a["end"] <= p["end"]:
                i += 1
            if p["end"] <= a["end"]:
                j += 1
        actual_ai = sum(s["end"] - s["start"] for s in actual if s["label"] == "ai")
        proposed_ai = sum(s["end"] - s["start"] for s in proposed if s["label"] == "ai")
        errors.append(abs(actual_ai - proposed_ai) / len(record["text"]))
        if actual_ai == 0:
            human_docs += 1
            human_doc_false_positives += proposed_ai > 0
        gb, pb = boundaries(actual), boundaries(proposed)
        matched += matched_boundaries(gb, pb, tolerance)
        gold_boundaries += len(gb)
        predicted_boundaries += len(pb)
    return {
        "documents": len(gold), "offset_unit": "unicode_codepoint",
        "character_metrics": {"accuracy": (tp + tn) / (tp + fp + fn + tn), **scores(tp, fp, fn),
                              "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "boundary_metrics": {"tolerance_chars": tolerance, **scores(matched, predicted_boundaries - matched, gold_boundaries - matched),
                             "matched": matched, "gold": gold_boundaries, "predicted": predicted_boundaries},
        "document_ai_fraction_mae": sum(errors) / len(errors),
        "human_control_document_fpr": human_doc_false_positives / human_docs if human_docs else None,
        "human_control_documents": human_docs,
    }
