"""Data model for the Agent-Pool feature (Phase 0 - no behavior yet).

Architektur-Entscheidung: eigenes Package statt Erweiterung von
tarno/ai/coding/, weil coding/ Aider-spezifisch ist (Git-Repo-Pflicht,
litellm-Single-Model). Der Pool-Orchestrator koordiniert mehrere
LLMProvider-Instanzen direkt und ruft AiderBackend nur an der
Merge-Grenze auf (siehe orchestrator.py/edit_apply.py, Phase 2) - Aider
selbst bekommt dadurch keinen neuen Pool-Code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tarno_backend.ai.pool.exceptions import InvalidPoolConfigError

PoolAgentRole = Literal["lead", "worker"]
PoolMessageKind = Literal["assign", "report", "question", "merge_decision", "status"]

_MIN_AGENTS = 2
_MAX_AGENTS = 4


@dataclass(frozen=True)
class PoolAgentSpec:
    """One agent's identity within a pool: which provider/model, which role."""

    provider: str
    model: str
    role: PoolAgentRole

    @property
    def slug(self) -> str:
        """Stable id used to address this agent in PoolMessage.from_agent/to_agent."""
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class PoolConfig:
    """A validated set of 2-4 agents with exactly one lead.

    Auto-picks the first agent as lead if none is marked - mirrors the
    UI's "optional Lead-Markierung, sonst Auto-Wahl" behavior (see plan).
    """

    agents: tuple[PoolAgentSpec, ...]

    def __post_init__(self) -> None:
        if not (_MIN_AGENTS <= len(self.agents) <= _MAX_AGENTS):
            raise InvalidPoolConfigError(
                f"Pool braucht {_MIN_AGENTS}-{_MAX_AGENTS} Agenten, hat {len(self.agents)}"
            )
        leads = [a for a in self.agents if a.role == "lead"]
        if len(leads) > 1:
            raise InvalidPoolConfigError(
                f"Pool darf nur einen Lead haben, hat {len(leads)}"
            )

    @classmethod
    def from_specs(cls, agents: list[PoolAgentSpec]) -> "PoolConfig":
        """Builds a PoolConfig, auto-promoting the first agent to lead if none is marked."""
        if agents and not any(a.role == "lead" for a in agents):
            agents = [
                PoolAgentSpec(a.provider, a.model, "lead") if i == 0
                else a
                for i, a in enumerate(agents)
            ]
        return cls(tuple(agents))

    @property
    def lead(self) -> PoolAgentSpec:
        return next(a for a in self.agents if a.role == "lead")

    @property
    def workers(self) -> tuple[PoolAgentSpec, ...]:
        return tuple(a for a in self.agents if a.role == "worker")


@dataclass
class SubTask:
    """One unit of work the Lead assigns to a specific worker agent."""

    id: str
    description: str
    assigned_to: str  # PoolAgentSpec.slug
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: str | None = None


@dataclass(frozen=True)
class PoolMessage:
    """One inter-agent communication event, streamed to the UI as-is.

    to_agent is either a specific agent slug, or "broadcast".
    """

    from_agent: str
    to_agent: str
    kind: PoolMessageKind
    text: str
    timestamp_ms: int = 0
    sub_task_id: str | None = None
