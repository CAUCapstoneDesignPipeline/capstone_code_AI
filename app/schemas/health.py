"""getAiLiveness / getAiReadiness 응답."""

from typing import Literal

from .common import Closed, Version


class Liveness(Closed):
    status: Literal["alive"]
    schemaVersion: Literal["1.0"]


class Readiness(Closed):
    status: Literal["ready"]
    schemaVersion: Literal["1.0"]
    profileVersions: list[Version]
