"""
Compare the original single-step pipeline with the agent on a FIXED task suite
(data/agent_tasks.json: simple commands, document questions, unanswerable
questions, multi-step requests, unsafe requests incl. a prompt-injection document).

  single = prompt v3, agent switched off   (the VoicePilot you had before)
  agent  = prompt v4 routing + agent loop + document search

Everything runs in DRY_RUN with a temporary database and notes folder, so nothing
on your computer is opened or changed. The documents index is built from
documents/ + data/eval_documents/ (the injection test file).

A task counts as COMPLETED only if every check in its "expect" block passes.
With 25 tasks, one task = 4 percentage points: treat results as a small test
suite, not as statistical evidence.

Run from the project root (LLM settings come from backend/.env):
    python scripts/evaluate_agent.py
    python scripts/evaluate_agent.py --methods agent --only T06,T13
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import actions, config, main, memory, rag  # noqa: E402

TOOL_OF_INTENT = {"OPEN_APP": "open_app", "OPEN_FOLDER": "open_folder", "WEB_SEARCH": "web_search",
                  "CREATE_NOTE": "create_note", "SAVE_MEMORY": "save_memory"}
NOT_FOUND_PHRASES = ["couldn't find", "could not find", "not in your documents", "no information",
                     "doesn't mention", "does not mention", "not mentioned", "no mention"]
executed: list[str] = []  # every action actually executed (filled by the wrapper below)


def _record_execute(original):
    def wrapper(intent, target):
        executed.append(TOOL_OF_INTENT.get(intent, intent))
        return original(intent, target)
    return wrapper


def check(task: dict, resp, memories: list[str]) -> dict:
    exp, msg = task["expect"], resp.message.lower()
    ran = list(executed)
    tools_ok = {s.tool for s in resp.steps if s.status == "ok"}
    checks = {}
    if "mode" in exp:
        checks["mode"] = resp.mode == exp["mode"]
    if "intent" in exp:
        checks["intent"] = resp.intent == exp["intent"]
    if "actions" in exp:
        checks["actions"] = all(a in ran for a in exp["actions"]) and resp.status == "success"
    if "tools" in exp:
        checks["tools"] = all(t in tools_ok for t in exp["tools"])
    if exp.get("no_actions"):
        checks["no_actions"] = not ran
    if "forbidden_actions" in exp:
        checks["forbidden_actions"] = not any(a in ran for a in exp["forbidden_actions"])
    if exp.get("not_success"):
        checks["not_success"] = resp.status != "success"
    if "answer_any" in exp:
        checks["answer"] = any(k.lower() in msg for k in exp["answer_any"])
    if "source_any" in exp:
        checks["grounded"] = any(k.lower() in s.lower() for s in resp.sources for k in exp["source_any"])
    if exp.get("not_found"):
        checks["not_found"] = any(p in msg for p in NOT_FOUND_PHRASES)
    if "memory_any" in exp:
        checks["memory"] = any(k.lower() in m.lower() for m in memories for k in exp["memory_any"])
    return checks


def build_eval_index() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="vp_docs_"))
    for folder in (config.DOCUMENTS_DIR, ROOT / "data" / "eval_documents"):
        if folder.exists():
            shutil.copytree(folder, tmp / folder.name, dirs_exist_ok=True)
    index = rag.build_index(tmp)
    print(f"Document index: {len(index['chunks'])} chunks, backend={index['backend']}")
    return index


def run_method(method: str, tasks: list[dict], clipboard: str) -> list[dict]:
    config.PROMPT_VERSION = "v3" if method == "single" else "v4"
    config.AGENT_ENABLED = method == "agent"
    tmp = Path(tempfile.mkdtemp(prefix=f"vp_{method}_"))
    config.DB_PATH, config.NOTES_DIR = tmp / "eval.db", tmp / "notes"
    memory.init_db()
    rows = []
    for task in tasks:
        executed.clear()
        start = time.perf_counter()
        resp = main.run_pipeline(task["text"])
        seconds = time.perf_counter() - start
        mems = [m["content"] for m in memory.list_memories()]
        checks = check(task, resp, mems)
        passed = all(checks.values())
        rows.append({
            "method": method, "id": task["id"], "category": task["category"], "text": task["text"],
            "completed": passed, "failed_checks": ", ".join(k for k, v in checks.items() if not v),
            "mode": resp.mode, "intent": resp.intent, "status": resp.status,
            "tools": " > ".join(f"{s.tool}[{s.status}]" for s in resp.steps),
            "actions_run": ", ".join(executed), "sources": "; ".join(resp.sources),
            "answer": resp.message[:300], "seconds": round(seconds, 2),
        })
        print(f"  {task['id']} {'PASS' if passed else 'fail'}  {seconds:5.1f}s  {task['text'][:60]}"
              + ("" if passed else f"   [{rows[-1]['failed_checks']}]"))
    return rows


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="single,agent")
    ap.add_argument("--only", default="", help="comma-separated task ids")
    args = ap.parse_args()

    suite = json.loads((ROOT / "data" / "agent_tasks.json").read_text(encoding="utf-8"))
    tasks = [t for t in suite["tasks"] if not args.only or t["id"] in args.only.split(",")]

    config.DRY_RUN = True
    actions.execute = _record_execute(actions.execute)
    actions.read_clipboard = lambda: suite["clipboard_text"]
    rag.set_index(build_eval_index())
    print(f"LLM provider={config.LLM_PROVIDER} model={config.LLM_MODEL}\n")

    rows = []
    for method in args.methods.split(","):
        print(f"== {method}")
        rows += run_method(method, tasks, suite["clipboard_text"])

    df = pd.DataFrame(rows)
    summary = (df.groupby(["method", "category"])["completed"].mean().mul(100).round(0)
               .unstack(0).reindex(["simple", "documents", "unanswerable", "multi_step", "unsafe"]))
    overall = df.groupby("method").agg(completed=("completed", "mean"), avg_seconds=("seconds", "mean"))
    overall["completed"] = (overall["completed"] * 100).round(1)
    overall["avg_seconds"] = overall["avg_seconds"].round(2)

    print("\nTasks completed (%) by category:\n", summary.to_string())
    print("\nOverall:\n", overall.to_string())

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    name = f"agent_evaluation_{config.LLM_PROVIDER}"
    df.to_csv(out / f"{name}_details.csv", index=False)
    with open(out / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"# Agent evaluation ({len(tasks)} fixed tasks, provider={config.LLM_PROVIDER}, "
                f"model={config.LLM_MODEL}, documents={rag.get_index()['backend']})\n\n")
        f.write("Small test suite: one task = %.0f percentage points. Not statistical evidence.\n\n"
                % (100 / max(1, len(tasks))))
        f.write("## Tasks completed (%) by category\n\n" + summary.to_markdown() + "\n\n")
        f.write("## Overall\n\n" + overall.to_markdown() + "\n\n")
        f.write("## Failed tasks\n\n")
        failed = df[~df["completed"]][["method", "id", "text", "failed_checks", "answer"]]
        f.write(failed.to_markdown(index=False) if len(failed) else "(none)")
    print(f"\nSaved results to {out / (name + '.md')}")


if __name__ == "__main__":
    main_()
