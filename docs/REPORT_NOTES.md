# Notes for the final report

The report (3–5 pages) must follow this order. Below, each heading lists what in the code base you can point to.

## 1–2. Implementation process, environment, tools, data, logic
* Built in MVP order: FastAPI → React ↔ FastAPI → microphone → Whisper → one intent → one action → all intents → memory → dataset/evaluation → UI polish → TTS.
* Environment: VS Code, Python venv, Node/Vite, Swagger UI (`/docs`) for API testing, pytest (17 tests).
* Computational logic: `main.run_pipeline()`. Transcribe → build prompt (v3 includes memories + last 5 turns) → LLM returns JSON → `parse_json` repairs/extracts it → `actions.validate` (allowlist) → execute → log to SQLite.
* Safety: the LLM never produces commands. App and folder names are mapped to fixed launch strings. Destructive words ("delete", "format", "shut down") are always rejected (`fallback.is_dangerous`).
* Fallback: if the LLM is unreachable or returns bad JSON, a TF-IDF + Logistic Regression classifier trained on the dataset takes over.
* Data: `preprocessing_report.json` holds the counts for each cleaning step. Describe your own recordings here (how many, which intents, which microphone).

## 3. Use of generative AI tools (must be stated)
Be honest and specific. For example: "The initial code skeleton for the backend, frontend and scripts was generated with Claude (AI assistant) from my project plan. I then … (list what *you* did: tested on Windows, recorded the dataset, tuned the prompts, changed X, fixed Y, wrote the report)." The seed rows in `commands_raw.csv` were also AI-generated paraphrases. Say so, and point out that your own recordings form the raw voice data.

## 4. Missing features / 5. Additional features
* Additional (not in the plan): spoken replies (browser TTS), offline classifier fallback, Dataset tab for recording labelled samples, dry-run mode, provider switch (Ollama/Groq/OpenAI/none).
* Possible missing items, depending on what you finish: fine-tuning was not done because the dataset is too small. A classifier baseline was used instead.

## 6. Success / failure – fill in with your own results
| Method | Intent accuracy (test) | Valid JSON |
|---|---|---|
| rules | 96.8 % | – |
| classifier | 90.3 % | – |
| LLM v1 | … | … |
| LLM v2 | … | … |
| LLM v3 | … | … |
Get these with `python scripts/evaluate_prompts.py` while Ollama/Groq is running, then copy them from `results/prompt_evaluation.md`. Typical observations to look for: v1 gives inconsistent label names and app names ("Visual Studio Code" vs "vscode"). v2 fixes the format. v3 handles memory questions, follow-ups and unsafe requests better.

## 7. What I learnt – keep a log while working
Problems, solutions, failed experiments (e.g. small models adding text around the JSON → solved with regex extraction + few-shot examples).

## 8. User guide → README.md sections 1–2

## 9. Demo video (≥ 5 min) – suggested script
1. Show the repository structure + README (30 s)
2. Installation: `setup_windows.bat`, `.env`, `ollama pull llama3.2:3b` (1.5 min)
3. Start backend + frontend, show Swagger `/docs` (30 s)
4. Voice demo: open VS Code, Downloads, search, note, remember → recall, clipboard summary, an unsafe request being rejected (2 min)
5. Dataset tab: record two samples, run `preprocess.py`, `train_classifier.py`, `evaluate_prompts.py`, show the results table (1 min)
6. Switch `PROMPT_VERSION=v1` to show a worse result (30 s)
