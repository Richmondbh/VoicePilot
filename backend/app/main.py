"""
VoicePilot FastAPI application.

Pipeline:  audio -> speech.py (Whisper) -> assistant.py (LLM intent)
           -> actions.validate (allowlist) -> actions.execute -> memory.py (log)

FastAPI references:
- First steps:      https://fastapi.tiangolo.com/tutorial/first-steps/
- File uploads:     https://fastapi.tiangolo.com/tutorial/request-files/
- CORS middleware:  https://fastapi.tiangolo.com/tutorial/cors/
- Lifespan events:  https://fastapi.tiangolo.com/advanced/events/
Run with:  uvicorn app.main:app --reload   (from the backend folder)
"""
import csv
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import actions, assistant, config, fallback, memory, speech
from .models import CommandRequest, CommandResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    memory.init_db()
    print(f"[VoicePilot] LLM provider={config.LLM_PROVIDER} model={config.LLM_MODEL} "
          f"prompt={config.PROMPT_VERSION} dry_run={config.DRY_RUN}")
    yield


app = FastAPI(title="VoicePilot API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================ pipeline ==
def run_pipeline(text: str) -> CommandResponse:
    start = time.perf_counter()
    text = text.strip()
    history = memory.recent_interactions(config.HISTORY_TURNS)
    memories = memory.list_memories()[:20]

    # 1) AI interpretation (the AI only *suggests*)
    result = assistant.understand(text, history=history, memories=memories)
    intent, target = result.intent, result.target
    # Safety net: "delete my downloads" must never become "open downloads".
    if fallback.is_dangerous(text) and intent in {"OPEN_APP", "OPEN_FOLDER"}:
        intent, target = "UNKNOWN", None

    # 2) Validation + 3) execution (Python decides)
    try:
        if not text:
            raise actions.ActionRejected("I didn't hear anything. Please try again.")
        if intent == "UNKNOWN":
            raise actions.ActionRejected(
                result.reply if result.source == "llm" and result.reply
                else "Sorry, I can't help with that. Try opening an app, searching, or saving a note."
            )
        intent, target = actions.validate(intent, target)

        if intent == "RECALL_MEMORY":
            message = assistant.answer_from_memory(
                text, memory.list_memories(), memory.search_memories(target or text))
        elif intent == "SUMMARIZE_CLIPBOARD":
            clip = actions.read_clipboard()
            if not clip.strip():
                raise actions.ActionRejected("Your clipboard is empty - copy some text first.")
            message = assistant.summarize(clip)
        else:
            message = actions.execute(intent, target)
            if config.DRY_RUN and intent in {"OPEN_APP", "OPEN_FOLDER", "WEB_SEARCH"}:
                message += " (dry-run: nothing was launched)"
        status = "success"
    except actions.ActionRejected as exc:
        status, message = "rejected", str(exc)
    except Exception as exc:
        status, message = "error", f"Something went wrong: {exc}"

    memory.add_interaction(text, intent, target, status, message)
    return CommandResponse(
        transcript=text, intent=intent, target=target, status=status, message=message,
        interpreted_by=result.source, prompt_version=config.PROMPT_VERSION,
        duration_ms=int((time.perf_counter() - start) * 1000),
    )


def _suffix(upload: UploadFile) -> str:
    name = upload.filename or "audio.webm"
    return "." + name.rsplit(".", 1)[-1] if "." in name else ".webm"


# =============================================================== endpoints ==
@app.get("/api/health")
def health():
    return {"status": "ok", "llm_provider": config.LLM_PROVIDER, "llm_model": config.LLM_MODEL,
            "prompt_version": config.PROMPT_VERSION, "whisper_model": config.WHISPER_MODEL,
            "dry_run": config.DRY_RUN, "platform": actions.SYSTEM}


@app.post("/api/command", response_model=CommandResponse)
def command(req: CommandRequest):
    """Typed command (also handy for testing in Swagger at /docs)."""
    return run_pipeline(req.text)


@app.post("/api/voice", response_model=CommandResponse)
async def voice(audio: UploadFile = File(...)):
    """Recorded audio from the browser -> transcription -> full pipeline."""
    data = await audio.read()
    if len(data) < 1000:
        raise HTTPException(400, "Recording is empty or too short.")
    try:
        text = speech.transcribe_bytes(data, _suffix(audio))
    except Exception as exc:
        raise HTTPException(500, f"Speech recognition failed: {exc}")
    return run_pipeline(text)


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    return {"text": speech.transcribe_bytes(await audio.read(), _suffix(audio))}


@app.get("/api/history")
def history(limit: int = 50):
    return memory.recent_interactions(limit)


@app.delete("/api/history")
def clear_history():
    memory.clear_interactions()
    return {"ok": True}


@app.get("/api/memories")
def memories():
    return memory.list_memories()


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int):
    memory.delete_memory(memory_id)
    return {"ok": True}


# -------------------------------------------------- dataset collection ------
@app.post("/api/dataset/sample")
async def dataset_sample(audio: UploadFile = File(...), label: str = Form(...),
                         target: str = Form(""), prompt_text: str = Form("")):
    """Save a raw voice recording + its label for the dataset (data/raw/)."""
    audio_dir = config.RAW_DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sample_id = uuid.uuid4().hex[:10]
    filename = f"{sample_id}{_suffix(audio)}"
    (audio_dir / filename).write_bytes(await audio.read())

    csv_path = config.RAW_DATA_DIR / "recorded_commands.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "timestamp", "audio_file", "label", "target", "prompt_text"])
        writer.writerow([sample_id, datetime.now().isoformat(timespec="seconds"),
                         f"audio/{filename}", label, target, prompt_text])
    return {"ok": True, "id": sample_id, "file": filename}


@app.get("/api/dataset/count")
def dataset_count():
    csv_path = config.RAW_DATA_DIR / "recorded_commands.csv"
    if not csv_path.exists():
        return {"count": 0}
    with csv_path.open(encoding="utf-8") as f:
        return {"count": max(0, sum(1 for _ in f) - 1)}
