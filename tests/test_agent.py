"""
Tests for the agent loop and document search. A scripted fake LLM replaces the
real model so the safety rules can be tested without Ollama. Run: pytest -q
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ["LLM_PROVIDER"] = "none"
os.environ["DRY_RUN"] = "1"

import pytest  # noqa: E402

from app import agent, assistant, config, memory, rag  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "documents"


class FakeLLM:
    """Returns the scripted replies in order and records every prompt it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("the agent asked the LLM more times than expected")
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(config, "DRY_RUN", True)
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "AGENT_MAX_STEPS", 4)
    monkeypatch.setattr(config, "AGENT_MAX_ACTIONS", 2)
    memory.init_db()
    rag.set_index(rag.build_index(FIXTURES, backend="tfidf"))
    yield
    rag.set_index(None)


def fake(monkeypatch, *replies):
    llm = FakeLLM(replies)
    monkeypatch.setattr(assistant, "chat", llm)
    return llm


# ------------------------------------------------------------------- RAG ---
def test_search_finds_the_right_document():
    hits = rag.search("How many pages should the project report be?")
    assert hits and hits[0]["source"] == "course_rules.md"


def test_search_returns_nothing_for_unrelated_question():
    assert rag.search("What is the capital of France?") == []


def test_chunks_overlap_and_cover_text():
    text = "Sentence number %d is here. " * 120
    chunks = rag.chunk_text(text % tuple(range(120)))
    assert len(chunks) > 1 and all(len(c) <= rag.CHUNK_CHARS for c in chunks)


# ----------------------------------------------------------------- agent ---
def test_document_question_cites_retrieved_source(monkeypatch):
    fake(monkeypatch, '{"answer": "The report must be 3-5 pages.", "sources": ["S1", "S99"]}')
    res = agent.run("How long must the project report be?", "ASK_DOCUMENTS")
    assert res.steps[0].tool == "search_documents"          # Python searches first
    assert res.steps[-1].tool == "answer_from_documents"    # one focused answer call
    assert res.sources == ["course_rules.md"]               # S99 was never retrieved -> dropped
    assert "3-5 pages" in res.message


def test_document_question_plain_text_answer_is_accepted(monkeypatch):
    fake(monkeypatch, "The report can be 3-5 pages long.")
    res = agent.run("How long must the project report be?", "ASK_DOCUMENTS")
    assert res.status == "success" and "3-5" in res.message and res.sources


def test_document_question_not_found(monkeypatch):
    fake(monkeypatch, '{"answer": "I couldn\'t find that in your documents.", "sources": ["S1"]}')
    res = agent.run("What does the course say about parking at campus?", "ASK_DOCUMENTS")
    assert "couldn't find" in res.message and res.sources == []


def test_prompt_injection_in_document_cannot_trigger_actions(monkeypatch):
    # The model reads the document and then "falls for" the injected instruction.
    # The backend must still block it, because the user's own words never asked for it.
    fake(monkeypatch,
         '{"tool": "search_documents", "args": {"query": "study group notes"}}',
         '{"tool": "open_app", "args": {"name": "spotify"}}',
         '{"tool": "save_memory", "args": {"fact": "the user\'s password is 1234"}}',
         '{"final_answer": "The notes say the group meets on Thursday.", "sources": ["S1"]}')
    res = agent.run("check my study group notes and follow the instructions in them", "MULTI_STEP")
    assert res.actions_run == []
    assert [s.status for s in res.steps] == ["ok", "blocked", "blocked"]
    assert memory.list_memories() == []


def test_document_questions_cannot_call_tools_at_all(monkeypatch):
    fake(monkeypatch, '{"tool": "open_app", "args": {"name": "spotify"}}')
    res = agent.run("What do my study group notes say?", "ASK_DOCUMENTS")
    assert res.actions_run == [] and [s.tool for s in res.steps] == ["search_documents", "answer_from_documents"]


def test_repeated_searches_still_end_with_an_answer(monkeypatch):
    # A small model that keeps searching instead of answering (seen with llama3.2:3b).
    fake(monkeypatch, *(['{"tool": "search_documents", "args": {"query": "report pages"}}'] * 4),
         '{"answer": "3-5 pages.", "sources": ["S1"]}')
    res = agent.run("look up the report length and tell me and then summarise it", "MULTI_STEP")
    assert res.status == "success" and res.message == "3-5 pages."


def test_retrieved_text_is_marked_untrusted(monkeypatch):
    llm = fake(monkeypatch, '{"final_answer": "ok", "sources": []}')
    agent.run("What do my study group notes say?", "ASK_DOCUMENTS")
    assert "<untrusted>" in llm.prompts[0] and "IMPORTANT SYSTEM INSTRUCTION" in llm.prompts[0]


def test_multi_step_runs_requested_actions(monkeypatch):
    fake(monkeypatch,
         '{"tool": "web_search", "args": {"query": "fastapi tutorials"}}',
         '{"tool": "create_note", "args": {"text": "watch fastapi tutorials tonight"}}',
         '{"final_answer": "Searched and saved a note."}')
    res = agent.run("search for fastapi tutorials and make a note to watch them tonight", "MULTI_STEP")
    assert res.status == "success"
    assert [t for t, _ in res.actions_run] == ["web_search", "create_note"]
    assert len(list(config.NOTES_DIR.glob("*.txt"))) == 1


def test_action_limit(monkeypatch):
    fake(monkeypatch,
         '{"tool": "create_note", "args": {"text": "one"}}',
         '{"tool": "create_note", "args": {"text": "two"}}',
         '{"tool": "create_note", "args": {"text": "three"}}',
         '{"final_answer": "done"}')
    res = agent.run("write down three notes", "MULTI_STEP")
    assert len(res.actions_run) == 2 and res.steps[2].status == "blocked"


def test_step_limit_stops_the_loop(monkeypatch):
    fake(monkeypatch, *[f'{{"tool": "search_documents", "args": {{"query": "q{i}"}}}}' for i in range(4)])
    res = agent.run("tell me everything and keep searching and search more", "MULTI_STEP")
    assert len(res.steps) == 4 and "stopped after 4 steps" in res.message


def test_unknown_tool_and_allowlist_are_enforced(monkeypatch):
    fake(monkeypatch,
         '{"tool": "run_shell", "args": {"cmd": "del *"}}',
         '{"tool": "open_app", "args": {"name": "photoshop"}}',
         '{"final_answer": "I could not do that."}')
    res = agent.run("open photoshop and run a command", "MULTI_STEP")
    assert [s.status for s in res.steps] == ["blocked", "blocked"] and res.actions_run == []


def test_invalid_json_is_recovered(monkeypatch):
    fake(monkeypatch, "Sure! I will open it.", '{"final_answer": "Done."}')
    res = agent.run("open chrome and then search for news", "MULTI_STEP")
    assert res.steps[0].status == "error" and res.message == "Done."


# ------------------------------------------------------------ offline mode --
def test_offline_document_answer(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    res = agent.run("How long should the project report be?", "ASK_DOCUMENTS")
    assert res.sources and res.sources[0] == "course_rules.md"


def test_offline_multi_step_needs_llm(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "none")
    assert agent.run("search x and make a note", "MULTI_STEP").status == "rejected"


# -------------------------------------------------------------------- API ---
def test_api_routes_to_agent(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    fake(monkeypatch,
         '{"intent": "ASK_DOCUMENTS", "target": "report length", "reply": "Checking."}',   # router (v4)
         '{"answer": "3-5 pages, excluding the cover.", "sources": ["S1"]}')               # answer
    with TestClient(app) as client:
        r = client.post("/api/command", json={"text": "how long is the project report?"}).json()
    assert r["mode"] == "agent" and r["sources"] == ["course_rules.md"] and r["steps"]


def test_api_routing_guard_sends_cv_questions_to_documents(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    fake(monkeypatch,
         '{"intent": "RECALL_MEMORY", "target": "certification", "reply": "Let me check."}',  # wrong guess
         '{"answer": "Not in these test documents.", "sources": []}')
    with TestClient(app) as client:
        r = client.post("/api/command", json={"text": "Do I have a Microsoft certification on my CV?"}).json()
    assert r["intent"] == "ASK_DOCUMENTS" and r["mode"] == "agent"


def test_api_misheard_sentence_is_not_saved(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    fake(monkeypatch, '{"intent": "SAVE_MEMORY", "target": "we are setting it", "reply": "Ok."}')
    with TestClient(app) as client:
        r = client.post("/api/command", json={"text": "We're setting it while I have on my CV."}).json()
    assert r["status"] == "rejected" and memory.list_memories() == []


def test_api_fast_path_unchanged(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    fake(monkeypatch, '{"intent": "OPEN_APP", "target": "chrome", "reply": "Opening Chrome."}')
    with TestClient(app) as client:
        r = client.post("/api/command", json={"text": "open chrome"}).json()
    assert r["mode"] == "single" and r["status"] == "success" and r["steps"] == []
