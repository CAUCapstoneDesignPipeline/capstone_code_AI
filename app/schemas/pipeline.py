"""getExtractPipeline / setExtractPipeline / resetExtractPipeline 응답 (AI 쪽 제안 operation: 계약 v0.1.0 에 없다)."""

from typing import Literal

from .common import Closed, Version


class PipelineProfile(Closed):
    profileVersion: Version
    ready: bool
    concepts: str                     # 개념을 고르는 방법 (한 줄)
    relations: str                    # 관계를 고르는 방법 (한 줄)
    externalProvider: Literal["openai", "llm", "none"]       # 노트 원문이 나가는 곳: Decisions(OpenAI) · LLM 제공자 · 나가지 않음
    default: bool


class Pipeline(Closed):
    alias: Version                    # profileVersion 으로 이 값을 보내면 default 로 돈다
    default: Version | None           # 지금 alias 가 가리키는 profile (준비된 것이 없으면 null)
    source: Literal["override", "env", "auto"]               # PUT 으로 정했나 · AI_PIPELINE · 자동 (autoOrder 에서 처음 준비된 것)
    autoOrder: list[Version]
    profiles: list[PipelineProfile]
