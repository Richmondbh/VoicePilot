"""
pandas docs: https://pandas.pydata.org/docs/user_guide/10min.html
train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.actions import normalize_app, normalize_folder  # noqa: E402
from app.fallback import clean_text, extract_target  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# Many spellings of the same label appear in the raw data -> one canonical label.
LABEL_MAP = {
    "open_app": "OPEN_APP", "openapp": "OPEN_APP", "open app": "OPEN_APP", "app": "OPEN_APP",
    "open_folder": "OPEN_FOLDER", "open folder": "OPEN_FOLDER", "folder": "OPEN_FOLDER", "open-folder": "OPEN_FOLDER",
    "web_search": "WEB_SEARCH", "web search": "WEB_SEARCH", "search": "WEB_SEARCH",
    "create_note": "CREATE_NOTE", "create note": "CREATE_NOTE", "note": "CREATE_NOTE",
    "save_memory": "SAVE_MEMORY", "save memory": "SAVE_MEMORY", "remember": "SAVE_MEMORY", "save-memory": "SAVE_MEMORY",
    "recall_memory": "RECALL_MEMORY", "recall memory": "RECALL_MEMORY", "recall": "RECALL_MEMORY",
    "summarize_clipboard": "SUMMARIZE_CLIPBOARD", "summarize": "SUMMARIZE_CLIPBOARD", "clipboard": "SUMMARIZE_CLIPBOARD",
    "unknown": "UNKNOWN", "other": "UNKNOWN", "none": "UNKNOWN",
}
NOISE = re.compile(r"^\s*(\[.*\]|\.+)?\s*$")  # "", "[inaudible]", "..."


def normalise_label(label) -> str | None:
    if pd.isna(label) or not str(label).strip():
        return None
    return LABEL_MAP.get(str(label).strip().lower(), None)


def normalise_target(row) -> str | None:
    target = row["target"] if isinstance(row["target"], str) else ""
    if row["label"] == "OPEN_APP":
        return normalize_app(target) or normalize_app(row["text"])
    if row["label"] == "OPEN_FOLDER":
        return normalize_folder(target) or normalize_folder(row["text"])
    if row["label"] in {"WEB_SEARCH", "CREATE_NOTE", "SAVE_MEMORY"}:
        # Recordings are saved without a target, so derive it from the transcription
        # ("Search Kasia on Google" -> "Kasia on Google").
        return target.strip() or (extract_target(row["label"], row["text"]) or "").strip() or None
    return ""  # intents without a target


def load_recordings() -> pd.DataFrame:
    """Transcribe the user's own recordings with faster-whisper."""
    path = RAW / "recorded_commands.csv"
    if not path.exists():
        return pd.DataFrame()
    rec = pd.read_csv(path)
    if rec.empty:
        return rec
    from app.speech import transcribe_file
    texts = []
    for f in rec["audio_file"]:
        audio = RAW / f
        try:
            texts.append(transcribe_file(audio) if audio.exists() else "")
        except Exception as exc:
            print(f"  could not transcribe {f}: {exc}")
            texts.append("")
        print(f"  {f}: {texts[-1]!r}")
    rec["text"] = texts
    rec["source"] = "recording"
    rec["collected_on"] = rec["timestamp"].astype(str).str[:10]
    return rec[["id", "source", "collected_on", "text", "label", "target", "audio_file"]]


def main():
    report = {}
    df = pd.read_csv(RAW / "commands_raw.csv", dtype={"text": str, "target": str})
    df["audio_file"] = ""
    print("Transcribing recordings (if any)...")
    rec = load_recordings()
    df = pd.concat([df, rec], ignore_index=True)
    report["1_raw_rows"] = len(df)
    report["1_raw_rows_by_source"] = df["source"].value_counts().to_dict()
    audit = df.copy()            # every raw row + why it was dropped (for the recordings report)
    audit["drop_reason"] = ""

    def mark(dropped_index, reason):
        idx = [i for i in dropped_index if audit.at[i, "drop_reason"] == ""]
        audit.loc[idx, "drop_reason"] = reason

    # 1. remove empty / noise transcriptions
    df["text"] = df["text"].fillna("")
    noise = df["text"].str.match(NOISE) | (df["text"].str.split().str.len() < 2)  # "open", "ok"
    mark(df.index[noise], "empty / noise")
    df = df[~noise]
    report["2_after_removing_empty_noise"] = len(df)

    # 2. normalise text (whitespace, case for comparison, filler words)
    df["text"] = df["text"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["clean_text"] = df["text"].apply(clean_text)

    # 3. normalise labels, drop unlabelled rows
    df["label"] = df["label"].apply(normalise_label)
    report["3_unlabelled_dropped"] = int(df["label"].isna().sum())
    mark(df.index[df["label"].isna()], "no valid label")
    df = df.dropna(subset=["label"])

    # 4. normalise + validate targets (e.g. "crome" -> chrome); drop rows whose target is missing
    df["target"] = df.apply(normalise_target, axis=1)
    report["4_invalid_target_dropped"] = int(df["target"].isna().sum())
    mark(df.index[df["target"].isna()], "target not recognised (e.g. misheard app name)")
    df = df.dropna(subset=["target"])

    # 5. remove duplicates (after normalisation "  OPEN CHROME " == "open chrome")
    before = len(df)
    mark(df.index[df.duplicated(subset=["clean_text"])], "duplicate")
    df = df.drop_duplicates(subset=["clean_text"])
    report["5_duplicates_removed"] = before - len(df)
    report["6_clean_rows"] = len(df)
    report["6_rows_per_label"] = df["label"].value_counts().to_dict()
    recs = audit[audit["source"] == "recording"]
    report["6_recordings_kept"] = f"{int((recs['drop_reason'] == '').sum())} of {len(recs)}"

    # 6. stratified split 70 / 15 / 15
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=42)
    report["7_split"] = {"train": len(train), "val": len(val), "test": len(test)}

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["id", "source", "text", "clean_text", "label", "target", "audio_file"]
    df[cols].to_csv(OUT / "dataset_clean.csv", index=False)
    train[cols].to_csv(OUT / "train.csv", index=False)
    val[cols].to_csv(OUT / "val.csv", index=False)
    test[cols].to_csv(OUT / "test.csv", index=False)
    # One row per voice recording: what Whisper heard and whether it was kept (and why not).
    if len(recs):
        rep = recs[["id", "audio_file", "label", "text", "drop_reason"]].rename(columns={"text": "whisper_heard"})
        rep["drop_reason"] = rep["drop_reason"].replace("", "kept")
        rep.to_csv(OUT / "recordings_report.csv", index=False)
    (OUT / "preprocessing_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
