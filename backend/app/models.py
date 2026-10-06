"""
Pydantic models (request/response shapes and the structured intent).
Pydantic docs: https://docs.pydantic.dev/latest/concepts/models/
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Intent(str, Enum):
    OPEN_APP = "OPEN_APP"
    OPEN_FOLDER = "OPEN_FOLDER"
    WEB_SEARCH = "WEB_SEARCH"
    CREATE_NOTE = "CREATE_NOTE"
    SAVE_MEMORY = "SAVE_MEMORY"
    RECALL_MEMORY = "RECALL_MEMORY"
    SUMMARIZE_CLIPBOARD = "SUMMARIZE_CLIPBOARD"
    UNKNOWN = "UNKNOWN"


class IntentResult(BaseModel):
    """What the AI thinks the user wants (never executed directly)."""
    intent: str
    target: Optional[str] = None
    reply: Optional[str] = None      # short natural-language confirmation
    source: str = "llm"              # "llm" | "classifier" | "rules"


class CommandRequest(BaseModel):
    text: str


class CommandResponse(BaseModel):
    transcript: str
    intent: str
    target: Optional[str] = None
    status: str                      # "success" | "rejected" | "error"
    message: str                     # what the assistant says back
    interpreted_by: str              # llm / classifier / rules
    prompt_version: str
    duration_ms: int
