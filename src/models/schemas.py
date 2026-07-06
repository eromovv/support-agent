from __future__ import annotations

from pydantic import BaseModel, Field

class AgentAnswer(BaseModel):

    answer: str = Field(..., description="Текстовый ответ на вопрос пользователя")
    sources: list[str] = Field(
        default_factory=list, description="Список source/chunk_id документов, использованных для ответа"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность модели в ответе от 0 до 1"
    )

class JudgeVerdict(BaseModel):

    score: int = Field(..., ge=1, le=5, description="Оценка качества ответа от 1 до 5")
    rationale: str = Field(..., description="Короткое обоснование оценки")
    is_hallucination: bool = Field(
        default=False, description="True, если ответ содержит факты, не подтверждённые источниками"
    )

class TicketCreate(BaseModel):
    title: str
    description: str
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
