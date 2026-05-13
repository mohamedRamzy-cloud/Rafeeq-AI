from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User medical question"
    )

    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Conversation session id"
    )

    # ======================================================
    # VALIDATION
    # ======================================================
    @field_validator("question")
    @classmethod
    def validate_question(cls, v):

        if not v or not v.strip():
            raise ValueError("Question cannot be empty")

        return v.strip()

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, v):

        if not v:
            return None

        return v.strip()