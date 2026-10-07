"""
Action validation + execution.

SAFETY DESIGN: the LLM only *suggests* an intent and a target. This module
decides whether that suggestion is allowed. Applications and folders must be
on an allowlist, and i never run a command string produced by the model.

References:
- subprocess:  https://docs.python.org/3/library/subprocess.html
- webbrowser:  https://docs.python.org/3/library/webbrowser.html
- os.startfile (Windows): https://docs.python.org/3/library/os.html#os.startfile
- pyperclip:   https://github.com/asweigart/pyperclip
"""
import os
import platform
import re
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from . import config, memory

SYSTEM = platform.system()  # "Windows" | "Darwin" | "Linux"

ALLOWED_INTENTS = {
    "OPEN_APP", "OPEN_FOLDER", "WEB_SEARCH", "CREATE_NOTE",
    "SAVE_MEMORY", "RECALL_MEMORY", "SUMMARIZE_CLIPBOARD",
}

# ------------------------------------------------------- applications ------
# canonical name -> launch command per operating system (fixed strings only)
APP_COMMANDS = {
    "chrome":     {"Windows": "chrome",     "Darwin": "Google Chrome",      "Linux": "google-chrome"},
    "edge":       {"Windows": "msedge",     "Darwin": "Microsoft Edge",     "Linux": "microsoft-edge"},
    "firefox":    {"Windows": "firefox",    "Darwin": "Firefox",            "Linux": "firefox"},
    "vscode":     {"Windows": "code",       "Darwin": "Visual Studio Code", "Linux": "code"},
    "calculator": {"Windows": "calc",       "Darwin": "Calculator",         "Linux": "gnome-calculator"},
    "notepad":    {"Windows": "notepad",    "Darwin": "TextEdit",           "Linux": "gedit"},
    "explorer":   {"Windows": "explorer",   "Darwin": "Finder",             "Linux": "nautilus"},
    "word":       {"Windows": "winword",    "Darwin": "Microsoft Word",     "Linux": "libreoffice --writer"},
    "excel":      {"Windows": "excel",      "Darwin": "Microsoft Excel",    "Linux": "libreoffice --calc"},
    "spotify":    {"Windows": "spotify:",   "Darwin": "Spotify",            "Linux": "spotify"},
}

APP_LABELS = {"chrome": "Google Chrome", "edge": "Microsoft Edge", "firefox": "Firefox",
              "vscode": "Visual Studio Code", "calculator": "Calculator", "notepad": "Notepad",
              "explorer": "File Explorer", "word": "Microsoft Word", "excel": "Microsoft Excel",
              "spotify": "Spotify"}

# spoken phrase -> canonical name (longest phrases are matched first)
APP_ALIASES = {
    "google chrome": "chrome", "chrome": "chrome", "browser": "chrome", "web browser": "chrome",
    "internet": "chrome", "crome": "chrome",
    "microsoft edge": "edge", "edge": "edge",
    "firefox": "firefox", "mozilla": "firefox",
    "visual studio code": "vscode", "vs code": "vscode", "vscode": "vscode", "code editor": "vscode",
    "vs coat": "vscode", "visual studio": "vscode", "code": "vscode",
    "calculator": "calculator", "calc": "calculator", "calculater": "calculator",
    "notepad": "notepad", "note pad": "notepad", "text editor": "notepad", "textedit": "notepad",
    "file explorer": "explorer", "explorer": "explorer", "finder": "explorer", "files": "explorer",
    "microsoft word": "word", "word": "word",
    "microsoft excel": "excel", "excel": "excel", "spreadsheet": "excel",
    "spotify": "spotify", "music": "spotify",
}

# ------------------------------------------------------------ folders ------
FOLDERS = {
    "downloads": "Downloads", "documents": "Documents", "desktop": "Desktop",
    "pictures": "Pictures", "music": "Music", "videos": "Videos", "home": "",
    "notes": None,  # special: VoicePilot notes folder
}
FOLDER_ALIASES = {
    "downloads": "downloads", "download": "downloads", "downloaded files": "downloads",
    "documents": "documents", "document": "documents", "docs": "documents", "my documents": "documents",
    "desktop": "desktop",
    "pictures": "pictures", "photos": "pictures", "images": "pictures", "picture": "pictures",
    "music": "music", "songs": "music",
    "videos": "videos", "video": "videos", "movies": "videos",
    "home": "home", "user folder": "home", "home folder": "home",
    "notes": "notes", "my notes": "notes", "voicepilot notes": "notes",
}


def _match_alias(text: str | None, aliases: dict) -> str | None:
    """Map free text such as 'Visual Studio Code for me' to a canonical key."""
    if not text:
        return None
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if t in aliases:
        return aliases[t]
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return aliases[alias]
    return None


def normalize_app(target: str | None) -> str | None:
    return _match_alias(target, APP_ALIASES)


def normalize_folder(target: str | None) -> str | None:
    return _match_alias(target, FOLDER_ALIASES)


# --------------------------------------------------------- validation ------
class ActionRejected(Exception):
    """Raised when an intent/target is not allowed."""


def validate(intent: str, target: str | None) -> tuple[str, str | None]:
    """Return (intent, cleaned_target) or raise ActionRejected."""
    if intent not in ALLOWED_INTENTS:
        raise ActionRejected("Sorry, I can't do that. It isn't one of my supported actions.")

    if intent == "OPEN_APP":
        app = normalize_app(target)
        if not app:
            allowed = ", ".join(sorted(APP_COMMANDS))
            raise ActionRejected(f"'{target}' is not on my list of allowed apps ({allowed}).")
        return intent, app

    if intent == "OPEN_FOLDER":
        folder = normalize_folder(target)
        if not folder:
            allowed = ", ".join(sorted(FOLDERS))
            raise ActionRejected(f"'{target}' is not an allowed folder ({allowed}).")
        return intent, folder

    if intent in {"WEB_SEARCH", "CREATE_NOTE", "SAVE_MEMORY"}:
        if not target or not target.strip():
            raise ActionRejected("I understood the action, but not what it should contain.")
        return intent, target.strip()[:500]

    return intent, target  # RECALL_MEMORY / SUMMARIZE_CLIPBOARD


# ---------------------------------------------------------- executors ------
def _open_path(path: Path) -> None:
    if config.DRY_RUN:
        return
    if SYSTEM == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif SYSTEM == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def open_app(app: str) -> str:
    command = APP_COMMANDS[app][SYSTEM if SYSTEM in ("Windows", "Darwin") else "Linux"]
    if not config.DRY_RUN:
        if SYSTEM == "Windows":
            # 'start' uses the Windows App Paths registry, so chrome/code/etc. are found.
            # `command` comes from the fixed table above, never from the LLM.
            subprocess.Popen(f'start "" {command}', shell=True)
        elif SYSTEM == "Darwin":
            subprocess.Popen(["open", "-a", command])
        else:
            subprocess.Popen(command.split())
    return f"Opening {APP_LABELS[app]}."


def open_folder(folder: str) -> str:
    if folder == "notes":
        path = config.NOTES_DIR
        path.mkdir(parents=True, exist_ok=True)
    else:
        path = Path.home() / FOLDERS[folder]
    if not path.exists():
        raise ActionRejected(f"The folder {path} does not exist on this computer.")
    _open_path(path)
    return f"Opening your {folder} folder."


def web_search(query: str) -> str:
    url = "https://www.google.com/search?q=" + quote_plus(query)
    if not config.DRY_RUN:
        webbrowser.open(url)
    return f"Searching Google for “{query}”."


def create_note(content: str) -> str:
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = config.NOTES_DIR / f"note_{stamp}.txt"
    path.write_text(content + "\n", encoding="utf-8")
    return f"Note saved to {path.name}."


def save_memory(fact: str) -> str:
    memory.add_memory(fact)
    return f"Okay, I'll remember that {fact}."


def read_clipboard() -> str:
    try:
        import pyperclip
        return pyperclip.paste() or ""
    except Exception:  # e.g. no clipboard tool on Linux
        return ""


def execute(intent: str, target: str | None) -> str:
    """Run a *validated* simple action and return a message for the user."""
    if intent == "OPEN_APP":
        return open_app(target)
    if intent == "OPEN_FOLDER":
        return open_folder(target)
    if intent == "WEB_SEARCH":
        return web_search(target)
    if intent == "CREATE_NOTE":
        return create_note(target)
    if intent == "SAVE_MEMORY":
        return save_memory(target)
    raise ActionRejected("This action needs the assistant and is handled elsewhere.")
