# VoicePilot – AI-powered voice assistant for desktop task automation

A voice assistant that turns spoken requests into controlled desktop actions and answers questions from your own documents. Built as project work for **DA598A Introduction to Generative AI** by Richmond Boakye.

VoicePilot transcribes speech with faster-whisper and uses an LLM to interpret the request. Simple commands run straight away after the proposed action is checked against an allowlist. Requests that need several steps or your documents go to a bounded agent. The AI suggests, and Python decides what is allowed to run.

## Screenshots

### Assistant

![VoicePilot assistant screen](docs/speakingview.png)

### Dataset recorder

![VoicePilot dataset recorder](docs/dataview.png)

## Features

- Record or type a command in the React interface.
- Open approved applications, folders, and web searches.
- Create notes, save and recall memories, and summarise clipboard text.
- **Agent mode:** handles requests that need several steps ("search for X and make a note") with a bounded agent that calls tools one at a time. The interface shows each step and whether it was allowed or blocked.
- **Document search (RAG):** read-only search over the files in `documents/` (CV, course documents, project plan). Each answer shows its source file and page.
- View conversation history and optionally hear spoken replies.
- Record labelled voice commands for the project dataset.
- If the LLM is unavailable, a local classifier and keyword rules take over.

The LLM can suggest an action, but it cannot run commands directly. The Python backend validates every action against an allowlist and rejects unsupported requests.

## Technology

- **Frontend:** React, TypeScript, Vite
- **Backend:** Python, FastAPI
- **Speech recognition:** faster-whisper (local)
- **LLM:** Ollama (`llama3.2:3b`, local and free). Groq, OpenAI or an offline mode can be used instead.
- **Document search:** Ollama embeddings (`nomic-embed-text`) with a TF-IDF fallback; pypdf and python-docx read the files
- **Offline classifier:** scikit-learn TF-IDF and Logistic Regression
- **Storage:** SQLite
- **Tests:** pytest (40 tests, using a scripted fake LLM)

## How it works

```
 Microphone (React + TypeScript)
        │  audio (.webm)
        ▼
 FastAPI ──► speech.py     faster-whisper (local)              "could you open vs code"
         ──► assistant.py  LLM + prompt v4 (memory/history)     {"intent":"OPEN_APP","target":"vscode"}
                │
                ├─ simple command ──► actions.py  allowlist check ──► run ──► memory.py (log)
                │
                └─ ASK_DOCUMENTS / MULTI_STEP ──► agent.py (bounded tool loop, rag.py for documents)
        ▼
 UI shows: You said → VoicePilot understood → (agent steps + sources) → Result
```

### Single-step commands

| Intent | Example | What happens |
|---|---|---|
| `OPEN_APP` | "Could you open Visual Studio Code?" | launches an allow-listed app (chrome, edge, firefox, vscode, calculator, notepad, explorer, word, excel, spotify) |
| `OPEN_FOLDER` | "Show my Downloads" | opens downloads / documents / desktop / pictures / music / videos / home / notes |
| `WEB_SEARCH` | "Search Google for FastAPI tutorials" | opens a Google search in the default browser |
| `CREATE_NOTE` | "Create a note saying I must finish my report" | writes a `.txt` file to `~/VoicePilot Notes` |
| `SAVE_MEMORY` | "Remember that my presentation is on Friday" | stores the fact in SQLite |
| `RECALL_MEMORY` | "What did I ask you to remember about Friday?" | LLM answers using only the saved memories |
| `SUMMARIZE_CLIPBOARD` | "Summarize what's in my clipboard" | LLM summarises the copied text in 3 bullet points |
| `UNKNOWN` | "Delete all my files" | politely rejected |

### Agent requests (prompt v4)

| Intent | Example | Handled by |
|---|---|---|
| `ASK_DOCUMENTS` | "What certification do I have according to my CV?" | Python searches the documents first, then one LLM call answers only from the retrieved excerpts and cites them |
| `MULTI_STEP` | "Search for FastAPI tutorials and make a note to watch them tonight" | a loop where the LLM calls one tool at a time and sees each result (max 4 steps, max 2 actions) |

Agent tools: `search_documents`, `recall_memory` and `read_clipboard` are read-only. `open_app`, `open_folder`, `web_search`, `create_note` and `save_memory` are actions.

**Safety rules enforced by Python, not by the prompt:**
1. Only the listed tools exist. Every action goes through the same `actions.validate()` allowlist as the single-step path.
2. An action may only run if **the user's own words** asked for that kind of action. For example, `create_note` needs "note" or "write down" in the request. Text from a document or the clipboard can never trigger an action.
3. At most 2 actions and 4 steps per request, and no identical repeated calls.
4. Document and clipboard text is marked as `<untrusted>` in the prompt. This is defence in depth: rules 1–3 hold even if the model ignores the marking.
5. A document question with no relevant excerpts is always answered "I couldn't find that in your documents."
6. Saving a memory or note needs words like "remember" or "make a note", so misheard speech is not saved by mistake.
7. Before accepting the final answer, a completion check sends the model back once if a requested action has not been done.

**Document search (`backend/app/rag.py`):** files in `documents/` (`.pdf`, `.docx`, `.txt`, `.md`) are split into overlapping chunks of about 800 characters. PDFs are split per page, so answers can cite a page. Each chunk is embedded with Ollama's `nomic-embed-text`. If Ollama isn't available, a TF-IDF keyword search is used instead. The index is saved in `backend/rag_index.json` and rebuilt automatically when files change, or when you select **Re-index** in the UI.

---

## 1. Installation (Windows, macOS or Linux)

**Requirements:** Python 3.10+, Node.js 18+ and [Ollama](https://ollama.com/download). You need about 3 GB of disk space for the Whisper, LLM and embedding models.

### One-time model download
```bash
ollama pull llama3.2:3b          # language model
ollama pull nomic-embed-text     # embeddings for document search
ollama list                      # both should be listed
```

### Quick way (Windows)
1. Double-click **`setup_windows.bat`**. It creates a virtual environment, installs everything, builds the dataset and trains the classifier.
2. Open `backend\.env` and check `LLM_PROVIDER=ollama` and `PROMPT_VERSION=v4`.
3. Double-click **`run_windows.bat`**. The app opens at http://localhost:5173.

### Manual way (all operating systems)
```bash
# 1) backend
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env   # Windows: copy backend\.env.example backend\.env

# 2) dataset + offline classifier
python scripts/preprocess.py
python scripts/train_classifier.py

# 3) start the API (terminal 1)
cd backend
uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs = Swagger UI

# 4) start the UI (terminal 2)
cd frontend
npm install
npm run dev                              # open http://localhost:5173
```
The first voice command downloads the Whisper `base` model (about 150 MB), so it takes a little longer.

> Always activate the virtual environment (`(.venv)` shown in the terminal) before running `pip` or the scripts. Otherwise packages are installed into a different Python.

### Settings (`backend/.env`)

| Setting | Values |
|---|---|
| `LLM_PROVIDER` | `ollama` (default, free and local) · `groq` (free tier, needs `GROQ_API_KEY`) · `openai` (cheap `gpt-4o-mini`, needs `OPENAI_API_KEY`) · `none` (offline classifier and rules) |
| `PROMPT_VERSION` | `v4` (default, enables agent routing) · `v1`–`v3` (single-step only) |
| `AGENT_ENABLED`, `AGENT_MAX_STEPS`, `AGENT_MAX_ACTIONS` | `1`, `4`, `2` |
| `RAG_BACKEND` | `auto` (Ollama embeddings, otherwise TF-IDF) · `ollama` · `tfidf` |
| `WHISPER_MODEL` | `tiny` · `base` (default) · `small` (more accurate, slower) |
| `DRY_RUN` | `1` = validate actions but don't open anything (good for testing) |

If the LLM can't be reached, VoicePilot falls back to the offline classifier, and the UI shows `classifier (LLM unavailable)`.

---

## 2. Using the app
* **Assistant tab:** click the microphone, speak, then click again to send. You can also type a command. The result card shows *You said → VoicePilot understood → Result*. For agent requests it also shows each step and the sources used. The right-hand column shows your documents, saved memories and conversation history. Tick **Speak replies** to hear the answer through the browser's text-to-speech.
* **Documents panel:** lists the indexed files. Select **Re-index** after adding files. The line under the list should read "Semantic search · nomic-embed-text".
* **Dataset tab:** pick an intent label and record yourself saying a command. The recording is saved to `data/raw/audio/` and `data/raw/recorded_commands.csv`, and `scripts/preprocess.py` transcribes and cleans it.

### Your own documents
Put files you want to ask about in `documents/`, for example `documents/course/` and `documents/personal/`. **`documents/personal/` is in `.gitignore`**, so private files such as a CV stay on your computer and are not pushed to GitHub. The search index (`backend/rag_index.json`) is also ignored, because it contains text from your documents.

---

## 3. Data, preprocessing and evaluation

```
data/raw/commands_raw.csv        seed set: typed / paraphrased / transcribed commands (messy on purpose)
data/raw/recorded_commands.csv   own voice recordings (created by the Dataset tab)
        │  scripts/preprocess.py
        ▼
 transcribe recordings with Whisper → drop empty/noise rows ("[inaudible]", "...", 1-word) →
 normalise text (case, whitespace, filler words) → map label spellings ("open app", "Open_App") to 8 labels →
 validate targets ("crome" → chrome; missing targets derived from the transcription) →
 remove duplicates → stratified 70/15/15 split
        ▼
data/processed/{dataset_clean,train,val,test}.csv
data/processed/preprocessing_report.json   counts for every step
data/processed/recordings_report.csv       what Whisper heard for each recording, kept or why dropped
        │  scripts/train_classifier.py   TF-IDF (word + char n-grams) → Logistic Regression
        │  scripts/evaluate_prompts.py   rules vs classifier vs prompt v1–v4
        │  scripts/evaluate_agent.py     single-step vs agent on 25 fixed tasks
        ▼
results/classifier_report.txt, results/prompt_evaluation.md, results/agent_evaluation_<provider>.md
```

**Current dataset:** 259 raw rows (114 typed, 64 Whisper transcripts, 52 paraphrases, 29 recordings) → **211 clean examples**. Removed: 8 empty or noisy rows, 1 unlabelled row, 21 unrecognised targets and 18 duplicates. Split: 147 / 32 / 32. 10 of the 29 recordings were kept. Whisper `base` misheard many short clips, and `recordings_report.csv` shows which.

### Intent recognition (test split, 32 examples, llama3.2:3b)

| Method | Intent accuracy | Target accuracy* |
|---|---|---|
| Keyword rules | 93.8 % | 90.9 % |
| TF-IDF + Logistic Regression | 90.6 % | 90.9 % |
| LLM prompt v1 | 81.2 % | 100 % |
| LLM prompt v2 | 93.8 % | 100 % |
| LLM prompt v3 | **96.9 %** | 90.9 % |
| LLM prompt v4 (agent routing) | 90.6 % | 90.9 % |

\* apps and folders only. One example is about 3 percentage points, so treat these results as preliminary.

### Prompt versions (`backend/app/assistant.py`)
* **v1**: one line, "Determine what action the user wants", plus the list of intents.
* **v2**: adds a role, a definition for every intent, rules for the target, the allowed apps and folders, "JSON only", and a note that speech recognition makes mistakes.
* **v3**: v2 plus **few-shot examples**, **saved memories**, the **last 5 conversation turns** (so "open it again" works), a rule for questions versus statements, safety rules for UNKNOWN, and a short spoken `reply`.
* **v4**: v3 plus the routing intents `ASK_DOCUMENTS` and `MULTI_STEP`, with examples, which hand requests to the agent.

### Agent evaluation (25 fixed tasks, dry-run, llama3.2:3b)

`data/agent_tasks.json` holds the fixed task suite. One unsafe task uses a document containing a prompt-injection attempt (`data/eval_documents/`).

| Category (tasks) | Single-step (v3) | Agent (v4) |
|---|---|---|
| Simple commands (5) | 100 % | 100 % |
| Questions answered from documents (7) | 0 % | 100 % |
| Questions with no answer in the documents (3) | 0 % | 100 % |
| Multi-step requests (6) | 0 % | 33 % |
| Unsafe requests incl. prompt injection (4) | 75 % | 100 % |
| **Overall (25)** | **32 %** | **84 %** |
| Average time per task | 2.0 s | 7.1 s |

A task counts only if every check passes. One task equals 4 percentage points, so this is a small test suite, not statistical evidence. Multi-step planning is the main weakness with a 3B model.

```bash
python scripts/evaluate_agent.py                                  # single-step vs agent
python scripts/evaluate_agent.py --methods agent --only T06,T13   # selected tasks
python -m pytest -q                                               # 40 tests
```

---

## 4. Project structure
```
VoicePilot/
├── backend/
│   ├── app/
│   │   ├── main.py        FastAPI endpoints, pipeline and routing guards
│   │   ├── speech.py      faster-whisper speech-to-text
│   │   ├── assistant.py   prompts v1–v4, LLM calls, JSON parsing, recall and summary
│   │   ├── agent.py       bounded agent loop, tool permissions, completion check
│   │   ├── rag.py         document loading, chunking, embeddings, search
│   │   ├── actions.py     allowlists, validation, desktop actions
│   │   ├── fallback.py    offline classifier, keyword rules, routing rules
│   │   ├── memory.py      SQLite history and memories
│   │   ├── models.py      Pydantic models
│   │   └── config.py      settings from .env
│   ├── models/intent_classifier.joblib
│   ├── requirements.txt
│   └── .env.example
├── frontend/   React + TypeScript + Vite
│   └── src/  App.tsx
│             components/  VoiceRecorder, ActionCard, AgentTrace, DocumentsPanel,
│                          MemoryPanel, ConversationHistory, StatusIndicator, DatasetRecorder
│             services/    api.ts, useRecorder.ts
│             types/
├── documents/      files the agent can search (personal/ is git-ignored)
├── data/           raw/, processed/, agent_tasks.json, eval_documents/
├── scripts/        preprocess.py, train_classifier.py, evaluate_prompts.py, evaluate_agent.py
├── results/        evaluation output
├── tests/          pytest tests + fixtures
├── docs/           screenshots
├── setup_windows.bat
└── run_windows.bat
```

API overview (try it in Swagger at http://127.0.0.1:8000/docs):
`GET /api/health` · `POST /api/command {text}` · `POST /api/voice (audio)` · `POST /api/transcribe` · `GET/DELETE /api/history` · `GET /api/memories` · `DELETE /api/memories/{id}` · `GET /api/documents` · `POST /api/documents/reindex` · `POST /api/dataset/sample` · `GET /api/dataset/count`

---

## 5. Troubleshooting
* **"Backend offline" in the UI:** start `uvicorn app.main:app` inside the `backend` folder.
* **Documents show "0 chunks" / `No module named 'pypdf'`:** activate `.venv`, run `pip install -r backend/requirements.txt`, restart the backend and select Re-index.
* **"Keyword search (TF-IDF)" instead of semantic search:** run `ollama pull nomic-embed-text`, make sure Ollama is running, then select Re-index.
* **Agent never used / backend prints a note about v4:** set `PROMPT_VERSION=v4` in `backend/.env`.
* **Microphone doesn't work:** allow microphone access and use `http://localhost:5173`, not an IP address.
* **App doesn't open on Windows:** the app must be installed. VS Code needs "Add to PATH" during install.
* **Clipboard on Linux:** install `xclip` (`sudo apt install xclip`).
* **Slow or inaccurate transcription:** set `WHISPER_MODEL=tiny` for speed or `small` for accuracy.

---

## 6. References (public code, libraries and documentation used)

Code comments in each file point to the specific source it follows.

**Agent and document search**
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. https://arxiv.org/abs/2210.03629
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. https://arxiv.org/abs/2005.11401
- Greshake et al. (2023). *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*. https://arxiv.org/abs/2302.12173
- OWASP Top 10 for LLM Applications – LLM01 Prompt Injection. https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- Ollama – embeddings API. https://docs.ollama.com/capabilities/embeddings · model: https://ollama.com/library/nomic-embed-text
- pypdf – text extraction. https://pypdf.readthedocs.io/en/stable/user/extract-text.html
- python-docx. https://python-docx.readthedocs.io/en/latest/
- scikit-learn – TF-IDF term weighting. https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting

**Speech / AI**
- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). https://arxiv.org/abs/2212.04356 · https://github.com/openai/whisper
- SYSTRAN – faster-whisper (usage example in `speech.py`). https://github.com/SYSTRAN/faster-whisper
- OpenAI Python SDK. https://github.com/openai/openai-python
- Ollama – OpenAI compatibility. https://ollama.com/blog/openai-compatibility · Llama 3.2: https://ollama.com/library/llama3.2
- Groq – OpenAI compatibility. https://console.groq.com/docs/openai
- OpenAI – Prompt engineering guide. https://platform.openai.com/docs/guides/prompt-engineering
- Brown et al. (2020). *Language Models are Few-Shot Learners*. https://arxiv.org/abs/2005.14165

**Backend**
- FastAPI docs – first steps, request files, CORS, testing, lifespan events. https://fastapi.tiangolo.com/tutorial/
- Pydantic models. https://docs.pydantic.dev/latest/concepts/models/
- Python `sqlite3` tutorial. https://docs.python.org/3/library/sqlite3.html#tutorial
- Python `subprocess`, `webbrowser`, `os.startfile`. https://docs.python.org/3/library/subprocess.html · https://docs.python.org/3/library/webbrowser.html · https://docs.python.org/3/library/os.html#os.startfile
- pyperclip (clipboard). https://github.com/asweigart/pyperclip
- python-dotenv. https://github.com/theskumar/python-dotenv

**Data / ML**
- scikit-learn – Working with text data. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- scikit-learn – FeatureUnion, train_test_split, classification_report. https://scikit-learn.org/stable/modules/compose.html
- pandas – 10 minutes to pandas. https://pandas.pydata.org/docs/user_guide/10min.html

**Frontend**
- Vite – React + TypeScript template and dev-server proxy. https://vitejs.dev/guide/ · https://vitejs.dev/config/server-options.html#server-proxy
- React docs (hooks). https://react.dev/learn
- MDN – MediaStream Recording API (`useRecorder.ts`). https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API/Using_the_MediaStream_Recording_API
- MDN – Web Speech API `SpeechSynthesis` (spoken replies). https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesis
- MDN – Fetch API / FormData. https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch

**Use of generative AI tools:** this project was developed with the help of AI coding assistants, mainly  ChatGPT for parts of the backend. All code was integrated, tested and evaluated by me.
