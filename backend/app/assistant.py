"""
LLM interaction: prompt building (three refined versions), intent extraction
and response generation.

The `openai` Python SDK is used for every provider because Ollama, Groq and
OpenAI all expose the same Chat Completions API.
- OpenAI SDK:            https://github.com/openai/openai-python
- Ollama compatibility:  https://ollama.com/blog/openai-compatibility
- Groq compatibility:    https://console.groq.com/docs/openai
- Prompt-engineering guidance used for v2/v3 (clear instructions, few-shot
  examples, structured output): https://platform.openai.com/docs/guides/prompt-engineering
"""
import json
import os
import re

from . import config
from .fallback import understand as offline_understand
from .models import IntentResult

INTENTS = ["OPEN_APP", "OPEN_FOLDER", "WEB_SEARCH", "CREATE_NOTE",
           "SAVE_MEMORY", "RECALL_MEMORY", "SUMMARIZE_CLIPBOARD", "UNKNOWN",
           "ASK_DOCUMENTS", "MULTI_STEP"]  # the last two are routed to the agent (v4)

# =============================================================== PROMPTS ===
# v1: the naive first attempt.
PROMPT_V1 = """Determine what action the user wants.
Possible actions: OPEN_APP, OPEN_FOLDER, WEB_SEARCH, CREATE_NOTE, SAVE_MEMORY, RECALL_MEMORY, SUMMARIZE_CLIPBOARD, UNKNOWN.
Answer with JSON containing "intent" and "target".

User: {text}"""

# v2: adds a role, definitions for every intent, target rules and allowed values.
PROMPT_V2 = """You are the intent engine of VoicePilot, a desktop voice assistant.
Your job is to convert ONE spoken command into a structured action.

Available intents:
- OPEN_APP: start an application. target = the application name.
- OPEN_FOLDER: open a folder. target = folder name.
- WEB_SEARCH: search the internet. target = the search query only.
- CREATE_NOTE: write a text note to a file. target = the note text.
- SAVE_MEMORY: the user wants you to remember a fact. target = the fact, rewritten from the user's perspective ("my presentation is on Friday").
- RECALL_MEMORY: the user asks about something they told you before. target = the topic.
- SUMMARIZE_CLIPBOARD: summarise the text currently on the clipboard. target = null.
- UNKNOWN: anything else, or requests that are not one of the above.

Allowed apps: {apps}
Allowed folders: {folders}

Rules:
- Return valid JSON only, no explanation, no markdown.
- Use exactly one of the intent names above.
- The text comes from speech recognition and may contain small errors (e.g. "crome" = chrome).

JSON format: {{"intent": "...", "target": "..."}}

Command: "{text}"
"""

# v3: v2 + few-shot examples + conversation history + saved memories + a short reply.
PROMPT_V3 = """You are the intent engine of VoicePilot, a desktop voice assistant.
Convert the user's latest spoken command into ONE structured action and a short friendly reply.

## Intents
- OPEN_APP: start an application. target = canonical app name from the allowed list.
- OPEN_FOLDER: open a folder. target = folder name from the allowed list.
- WEB_SEARCH: search the internet. target = the search query only (remove words like "search for", "google").
- CREATE_NOTE: write a note to a file. target = only the content of the note.
- SAVE_MEMORY: remember a fact for later. target = the fact in first person ("my presentation is on Friday").
- RECALL_MEMORY: a question about something the user told you earlier. target = the topic being asked about.
- SUMMARIZE_CLIPBOARD: summarise the copied/clipboard text. target = null.
- UNKNOWN: small talk, unsupported or unsafe requests (deleting files, shutting down, sending emails...). target = null.

Allowed apps: {apps}
Allowed folders: {folders}

## Rules
1. Output ONLY one JSON object: {{"intent": "...", "target": "...", "reply": "..."}}
2. "reply" is one short sentence spoken back to the user.
3. Speech recognition can make mistakes ("crome" -> chrome, "vs coat" -> vscode). Fix them.
4. Questions ("what did I...", "when is my...") are RECALL_MEMORY, statements with "remember" are SAVE_MEMORY.
5. Use the conversation history to resolve references like "open it again" or "search that".
6. If the request is not supported, use UNKNOWN. Never invent new intents.

## Examples
Command: "could you start visual studio code for me"
{{"intent": "OPEN_APP", "target": "vscode", "reply": "Opening Visual Studio Code."}}
Command: "show me my downloads"
{{"intent": "OPEN_FOLDER", "target": "downloads", "reply": "Opening your Downloads folder."}}
Command: "google how to use fastapi with react"
{{"intent": "WEB_SEARCH", "target": "how to use fastapi with react", "reply": "Searching for that now."}}
Command: "make a note that I need to finish my report tomorrow"
{{"intent": "CREATE_NOTE", "target": "I need to finish my report tomorrow", "reply": "Note created."}}
Command: "remember that my project presentation is on friday"
{{"intent": "SAVE_MEMORY", "target": "my project presentation is on Friday", "reply": "Got it, I'll remember that."}}
Command: "what did I tell you about friday"
{{"intent": "RECALL_MEMORY", "target": "friday", "reply": "Let me check."}}
Command: "summarize what I just copied"
{{"intent": "SUMMARIZE_CLIPBOARD", "target": null, "reply": "Summarising your clipboard."}}
Command: "delete everything in my documents"
{{"intent": "UNKNOWN", "target": null, "reply": "Sorry, I'm not allowed to do that."}}

## Things the user asked you to remember
{memories}

## Recent conversation (oldest first)
{history}

## Latest command
Command: "{text}"
"""

# v4: v3 + two routing intents for the agent. Simple commands keep the fast path;
#     questions about the user's documents and requests needing several actions
#     are handed to agent.py.
PROMPT_V4 = PROMPT_V3.replace(
    "- UNKNOWN: small talk,",
    "- ASK_DOCUMENTS: a question that must be answered from the user's documents (their CV, course\n"
    "  documents, project plan, notes). target = the question.\n"
    "- MULTI_STEP: one request that needs two or more of the actions above, or a document lookup\n"
    "  followed by an action (\"find X in my CV and save it as a note\"). target = the full request.\n"
    "- UNKNOWN: small talk,",
).replace(
    "4. Questions (\"what did I...\", \"when is my...\") are RECALL_MEMORY, statements with \"remember\" are SAVE_MEMORY.",
    "4. Questions (\"what did I...\", \"when is my...\") are RECALL_MEMORY, statements with \"remember\" are SAVE_MEMORY.\n"
    "   RECALL_MEMORY is ONLY for things the user earlier asked you to remember.\n"
    "   Questions about the user's CV, profile, certificates, skills, experience, course rules or project plan\n"
    "   are ASK_DOCUMENTS. SAVE_MEMORY only when the user says remember / keep in mind / don't forget.\n"
    "   If the request contains two actions joined by \"and\" / \"then\", use MULTI_STEP.",
).replace(
    'Command: "delete everything in my documents"',
    'Command: "what certifications do I have according to my CV"\n'
    '{{"intent": "ASK_DOCUMENTS", "target": "what certifications do I have according to my CV", "reply": "Let me check your documents."}}\n'
    'Command: "search for fastapi tutorials and make a note to watch them tonight"\n'
    '{{"intent": "MULTI_STEP", "target": "search for fastapi tutorials and make a note to watch them tonight", "reply": "On it."}}\n'
    'Command: "delete everything in my documents"',
)

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3, "v4": PROMPT_V4}

RECALL_PROMPT = """You are VoicePilot. Answer the user's question using ONLY the saved memories below.
If the answer is not in the memories, say you don't have that saved. Answer in one or two short sentences,
addressing the user as "you" (e.g. "You told me that ...").

Saved memories:
{memories}

Question: {question}"""

SUMMARY_PROMPT = """Summarise the following text in at most 3 short bullet points.
Keep the most important facts, no introduction.

Text:
\"\"\"{text}\"\"\""""


# ============================================================ LLM CLIENT ===
_client = None


def get_client():
    """Create one OpenAI-compatible client for the configured provider."""
    global _client
    if _client is None and config.LLM_PROVIDER != "none":
        from openai import OpenAI
        p = config.PROVIDERS[config.LLM_PROVIDER]
        key = os.getenv(p["api_key_env"]) if p["api_key_env"] else "not-needed"
        _client = OpenAI(api_key=key or "missing-key", base_url=p["base_url"], timeout=60)
    return _client


def chat(prompt: str, temperature: float = 0.0, max_tokens: int = 200) -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("No LLM provider configured")
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def parse_json(raw: str) -> dict:
    """Extract the first {...} block - small models sometimes add text or ```json fences."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in model output: {raw!r}")
    data = json.loads(match.group(0))
    intent = str(data.get("intent", "UNKNOWN")).strip().upper().replace(" ", "_").replace("-", "_")
    target = data.get("target")
    if isinstance(target, str) and target.strip().lower() in {"", "null", "none"}:
        target = None
    return {"intent": intent if intent in INTENTS else "UNKNOWN",
            "target": target if target is None else str(target),
            "reply": data.get("reply")}


def build_prompt(text: str, version: str, history: list[dict] | None = None,
                 memories: list[dict] | None = None) -> str:
    from .actions import APP_COMMANDS, FOLDERS
    history_txt = "\n".join(
        f'User: "{h["user_message"]}" -> {h["intent"]} ({h.get("target") or "-"})'
        for h in (history or [])
    ) or "(none)"
    memory_txt = "\n".join(f"- {m['content']}" for m in (memories or [])) or "(none)"
    return PROMPTS[version].format(
        text=text.replace('"', "'"), apps=", ".join(APP_COMMANDS), folders=", ".join(FOLDERS),
        history=history_txt, memories=memory_txt,
    )


# ============================================================ PUBLIC API ===
def understand(text: str, history=None, memories=None, version: str | None = None) -> IntentResult:
    """Natural language -> IntentResult. Falls back to the offline model on any LLM problem."""
    version = version or config.PROMPT_VERSION
    if config.LLM_PROVIDER == "none":
        return offline_understand(text)
    try:
        raw = chat(build_prompt(text, version, history, memories))
        data = parse_json(raw)
        return IntentResult(**data, source="llm")
    except Exception as exc:  # network down, Ollama not running, bad JSON ...
        print(f"[assistant] LLM failed ({exc}); using offline fallback")
        result = offline_understand(text)
        result.source += " (LLM unavailable)"
        return result


def answer_from_memory(question: str, memories: list[dict], keyword_hits: list[dict]) -> str:
    if not memories:
        return "You haven't asked me to remember anything yet."
    if config.LLM_PROVIDER != "none":
        try:
            mem = "\n".join(f"- ({m['timestamp'][:10]}) {m['content']}" for m in memories[:30])
            return chat(RECALL_PROMPT.format(memories=mem, question=question), temperature=0.2).strip()
        except Exception as exc:
            print(f"[assistant] recall via LLM failed: {exc}")
    if keyword_hits:
        return "You told me: " + "; ".join(m["content"] for m in keyword_hits[:3]) + "."
    return "I couldn't find anything about that in my memory."


def summarize(text: str) -> str:
    if config.LLM_PROVIDER != "none":
        try:
            return chat(SUMMARY_PROMPT.format(text=text[:6000]), temperature=0.3, max_tokens=250).strip()
        except Exception as exc:
            print(f"[assistant] summary via LLM failed: {exc}")
    # Offline fallback: first two sentences (extractive "summary").
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:2]) + ("" if len(sentences) <= 2 else " …")
