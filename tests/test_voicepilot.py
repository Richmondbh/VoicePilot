"""
Unit tests for VoicePilot. Run from the project root:  pytest -q
Uses FastAPI's TestClient: https://fastapi.tiangolo.com/tutorial/testing/
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# Test settings: no LLM, never launch anything, temporary database/notes.
os.environ["LLM_PROVIDER"] = "none"
os.environ["DRY_RUN"] = "1"

import pytest  # noqa: E402

from app import actions, assistant, config, fallback  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    monkeypatch.setattr(config, "DRY_RUN", True)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- safety ---
def test_unknown_intent_is_rejected():
    with pytest.raises(actions.ActionRejected):
        actions.validate("RUN_SHELL", "rm -rf /")


def test_app_not_on_allowlist_is_rejected():
    with pytest.raises(actions.ActionRejected):
        actions.validate("OPEN_APP", "photoshop")


def test_app_aliases_and_speech_errors():
    assert actions.normalize_app("Visual Studio Code for me") == "vscode"
    assert actions.normalize_app("crome") == "chrome"
    assert actions.normalize_folder("my downloads") == "downloads"


# ------------------------------------------------------------- LLM output --
def test_parse_json_handles_markdown_fences():
    raw = 'Sure!\n```json\n{"intent": "open app", "target": "chrome", "reply": "ok"}\n```'
    assert assistant.parse_json(raw) == {"intent": "OPEN_APP", "target": "chrome", "reply": "ok"}


def test_parse_json_unknown_intent_becomes_unknown():
    assert assistant.parse_json('{"intent": "HACK", "target": null}')["intent"] == "UNKNOWN"


def test_prompt_v3_contains_memory_and_history():
    prompt = assistant.build_prompt(
        "open it again", "v3",
        history=[{"user_message": "open chrome", "intent": "OPEN_APP", "target": "chrome"}],
        memories=[{"content": "my exam is on Friday"}])
    assert "my exam is on Friday" in prompt and "open chrome" in prompt


# ------------------------------------------------------------- fallback ----
@pytest.mark.parametrize("text,intent,target", [
    ("Could you open Visual Studio Code for me?", "OPEN_APP", "vscode"),
    ("Show my Documents", "OPEN_FOLDER", "documents"),
    ("Search Google for FastAPI tutorials", "WEB_SEARCH", "FastAPI tutorials"),
    ("Create a note saying that I need to finish my report", "CREATE_NOTE", "I need to finish my report"),
    ("Remember that my presentation is on Friday", "SAVE_MEMORY", "my presentation is on Friday"),
    ("Summarize what is in my clipboard", "SUMMARIZE_CLIPBOARD", None),
    ("Delete all my files", "UNKNOWN", None),
])
def test_rules(text, intent, target):
    got = fallback.rule_intent(text)
    assert got == intent
    assert fallback.extract_target(got, text) == target


# ------------------------------------------------------------------- API ---
def test_full_pipeline_memory_roundtrip(client):
    r = client.post("/api/command", json={"text": "Remember that my presentation is on Friday"}).json()
    assert r["intent"] == "SAVE_MEMORY" and r["status"] == "success"
    r = client.post("/api/command", json={"text": "What did I ask you to remember about Friday?"}).json()
    assert r["intent"] == "RECALL_MEMORY" and "Friday" in r["message"]
    assert len(client.get("/api/memories").json()) == 1
    assert len(client.get("/api/history").json()) == 2


def test_open_app_dry_run(client):
    r = client.post("/api/command", json={"text": "open chrome please"}).json()
    assert r["status"] == "success" and r["target"] == "chrome"


def test_dangerous_command_rejected(client):
    r = client.post("/api/command", json={"text": "delete my downloads folder"}).json()
    assert r["status"] == "rejected"


def test_note_is_written(client):
    client.post("/api/command", json={"text": "make a note: buy milk"})
    notes = list(config.NOTES_DIR.glob("*.txt"))
    assert len(notes) == 1 and "buy milk" in notes[0].read_text()
