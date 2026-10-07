
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import assistant, config, fallback  # noqa: E402
from app.actions import normalize_app, normalize_folder  # noqa: E402

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def target_ok(label: str, expected, predicted) -> bool | None:
    """Only apps/folders have a canonical target that can be compared exactly."""
    if label == "OPEN_APP":
        return normalize_app(predicted or "") == expected
    if label == "OPEN_FOLDER":
        return normalize_folder(predicted or "") == expected
    return None


def run(method: str, df: pd.DataFrame) -> tuple[dict, list]:
    rows, json_ok = [], 0
    for _, s in df.iterrows():
        text = s["text"]
        if method == "rules":
            intent = fallback.rule_intent(text)
            target = fallback.extract_target(intent, text)
            json_ok += 1
        elif method == "classifier":
            r = fallback.understand(text)
            intent, target = r.intent, r.target
            json_ok += 1
        else:
            version = method.split("-")[1]
            try:
                raw = assistant.chat(assistant.build_prompt(text, version))
                parsed = assistant.parse_json(raw)
                intent, target = parsed["intent"], parsed["target"]
                json_ok += 1
            except Exception as exc:
                intent, target = "INVALID_OUTPUT", None
                print(f"   [{method}] invalid output for {text!r}: {exc}")
        rows.append({"method": method, "text": text, "label": s["label"], "predicted": intent,
                     "expected_target": s["target"], "predicted_target": target,
                     "intent_correct": intent == s["label"],
                     "target_correct": target_ok(s["label"], s["target"], target)})
    res = pd.DataFrame(rows)
    tgt = res["target_correct"].dropna()
    summary = {
        "method": method,
        "intent_accuracy": round(res["intent_correct"].mean() * 100, 1),
        "target_accuracy_apps_folders": round(tgt.mean() * 100, 1) if len(tgt) else None,
        "valid_json_rate": round(json_ok / len(df) * 100, 1),
        "samples": len(df),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "all"])
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    file = "dataset_clean.csv" if args.split == "all" else f"{args.split}.csv"
    df = pd.read_csv(DATA / file).fillna("")
    if args.limit:
        df = df.head(args.limit)

    methods = ["rules", "classifier"]
    if not args.skip_llm and config.LLM_PROVIDER != "none":
        methods += ["llm-v1", "llm-v2", "llm-v3"]
    elif not args.skip_llm:
        print("LLM_PROVIDER=none -> skipping LLM prompts (set it in backend/.env)")

    summaries, details = [], []
    for m in methods:
        start = time.time()
        print(f"Evaluating {m} on {len(df)} samples...")
        summary, rows = run(m, df)
        summary["seconds"] = round(time.time() - start, 1)
        summaries.append(summary)
        details += rows

    table = pd.DataFrame(summaries)
    print("\n", table.to_string(index=False))
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "prompt_evaluation.csv", index=False)
    pd.DataFrame(details).to_csv(RESULTS / "prompt_evaluation_details.csv", index=False)
    with open(RESULTS / "prompt_evaluation.md", "w", encoding="utf-8") as f:
        f.write(f"# Intent evaluation ({args.split} split, provider={config.LLM_PROVIDER}, "
                f"model={config.LLM_MODEL})\n\n")
        f.write(table.to_markdown(index=False) if hasattr(table, "to_markdown") else table.to_string())
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
