"""
Offline intent understanding, used when no LLM is configured or the LLM fails.

1. If a trained classifier exists (scripts/train_classifier.py creates it), it
   predicts the intent: TF-IDF features + Logistic Regression, following the
   scikit-learn text tutorial:
   https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
2. Otherwise simple keyword rules predict the intent.
3. Targets (app name, search query, note text...) are always extracted by rules.
"""
import re

from . import config
from .actions import normalize_app, normalize_folder
from .models import IntentResult

_classifier = None
_classifier_loaded = False


def _get_classifier():
    global _classifier, _classifier_loaded
    if not _classifier_loaded:
        _classifier_loaded = True
        if config.CLASSIFIER_PATH.exists():
            try:
                import joblib
                _classifier = joblib.load(config.CLASSIFIER_PATH)
            except Exception as exc:  # e.g. made with another scikit-learn version
                print(f"[fallback] could not load classifier ({exc}); re-run scripts/train_classifier.py")
    return _classifier


def clean_text(text: str) -> str:
    """Same normalisation as scripts/preprocess.py (keep them in sync)."""
    t = text.lower().strip()
    t = re.sub(r"\b(um+|uh+|erm|hmm+|hey voicepilot|voicepilot|okay|ok)\b", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


QUESTION_START = re.compile(r"^(what|when|where|who|do you|did i|have i|which|tell me what|can you tell me)\b")


DANGEROUS = re.compile(r"\b(delete|remove|erase|format|shut ?down|restart|kill|uninstall|install|send|email|pay|buy)\b")


def rule_intent(text: str) -> str:
    t = clean_text(text)
    if "clipboard" in t or "copied" in t:
        return "SUMMARIZE_CLIPBOARD"
    if re.search(r"\b(remember|forget|keep in mind|memori[sz]e)\b", t):
        if QUESTION_START.search(t) or text.strip().endswith("?"):
            return "RECALL_MEMORY"
        return "SAVE_MEMORY"
    if QUESTION_START.search(t) and re.search(r"\b(i told you|i said|my)\b", t):
        return "RECALL_MEMORY"
    if re.search(r"\b(note|write down|jot down|jot)\b", t):
        return "CREATE_NOTE"
    if re.search(r"\b(search|google|look up|lookup|find information|find info)\b", t):
        return "WEB_SEARCH"
    if DANGEROUS.search(t):  # never map "delete my files" etc. to an app/folder
        return "UNKNOWN"
    if re.search(r"\bfolder\b", t) or (normalize_folder(t) and not normalize_app(t)):
        if normalize_folder(t):
            return "OPEN_FOLDER"
    if normalize_app(t):
        return "OPEN_APP"
    return "UNKNOWN"


def _strip(pattern: str, text: str) -> str:
    out = re.sub(pattern, "", text.strip(), count=1, flags=re.IGNORECASE)
    out = re.sub(r"\b(for me|please|thanks|thank you)\b", "", out, flags=re.IGNORECASE)
    return out.strip(" .,!?:")


def extract_target(intent: str, text: str) -> str | None:
    if intent == "OPEN_APP":
        return normalize_app(text)
    if intent == "OPEN_FOLDER":
        return normalize_folder(text)
    if intent == "WEB_SEARCH":
        return _strip(r"^.*?\b(search (google|the web|online|the internet)?\s*(for|about)?|google|look up|lookup|find (information|info) (about|on))\b\s*", text)
    if intent == "CREATE_NOTE":
        return _strip(r"^.*?\b(note|write down|jot down)\b\s*(saying that|saying|that says|that|to say|about|:)?\s*", text)
    if intent == "SAVE_MEMORY":
        return _strip(r"^.*?\b(remember|don't forget|do not forget|keep in mind)\b\s*(that)?\s*", text)
    return None


# ---------------------------------------------------------------- agent routing (offline) ---
DOC_QUESTION = re.compile(
    r"\b(according to|my cv|cv|resume|my experience|my skills|project (plan|proposal|information|description)"
    r"|assignment|grading|grade vg|grade g|requirements for|course (rules|requirements|document)"
    r"|certificat\w*|certified|my profile|linkedin|my documents?|voice pilot"
    r"|what (do|does) my \w+( \w+)?( \w+)? (say|mention))\b")
# Questions that really are about things the user asked VoicePilot to remember
MEMORY_QUESTION = re.compile(r"\b(remember|told you|i said|asked you|did i (say|tell))\b")
ACTION_GROUPS = {
    "open": r"\b(open|launch|start|show)\b",
    "search": r"\b(search|google|look up|lookup)\b",
    "note": r"\b(note|write down|jot)\b",
    "memory": r"\b(remember|keep in mind)\b",
    "clipboard": r"\b(clipboard|copied)\b",
}


def route(text: str) -> str | None:
    """Rule-based detection of the two agent intents (used when no LLM is available)."""
    t = clean_text(text)
    if DANGEROUS.search(t):
        return None
    groups = [g for g, pattern in ACTION_GROUPS.items() if re.search(pattern, t)]
    doc = bool(DOC_QUESTION.search(t)) and "folder" not in t
    if re.search(r"\b(and|then)\b", t) and (len(groups) >= 2 or (doc and groups)):
        return "MULTI_STEP"
    if doc and not groups:
        return "ASK_DOCUMENTS"
    return None


def is_dangerous(text: str) -> bool:
    return bool(DANGEROUS.search(clean_text(text)))


def understand(text: str) -> IntentResult:
    routed = route(text)
    if routed:
        return IntentResult(intent=routed, target=text.strip(), source="rules")
    clf = _get_classifier()
    if clf is not None:
        intent, source = str(clf.predict([clean_text(text)])[0]), "classifier"
    else:
        intent, source = rule_intent(text), "rules"
    return IntentResult(intent=intent, target=extract_target(intent, text), source=source)
