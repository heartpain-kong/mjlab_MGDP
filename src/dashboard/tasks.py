"""Typed task model for the workbench.

A task is more than an id: it declares which algorithm/scene it belongs to, how
wide the actor input is, which reward profile and docs it uses, and whether it
needs a pretrained module before training can start.  The dashboard, the CLI
and the command builder all read these declarations instead of hard-coding
per-task branches.

``TaskSpec`` is a ``Mapping`` so existing call sites that index a task like a
dict (``task["experiment"]``, ``task.get("shared")``, ``**task``) keep working.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TaskSpec(Mapping):
    alias: str
    id: str
    label: str
    algorithm: str
    scene: str
    experiment: str
    obs: int
    critic: int
    description: str
    profile: str
    docs: str = ""
    requires_pretrain: bool = False
    pretrain_command: str = ""
    pretrain_checkpoint: str = ""
    supports_curriculum: bool = True
    directed_passage: bool = False
    shared: bool = False
    resume_sources: tuple[str, ...] = field(default_factory=tuple)
    train_presets: dict[str, dict[str, int]] = field(default_factory=dict)

    # Mapping protocol -----------------------------------------------------
    def as_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key):
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    # Convenience ----------------------------------------------------------
    @property
    def needs_pretrain(self) -> bool:
        return self.requires_pretrain

    def pretrain_hint(self) -> str:
        if not self.requires_pretrain:
            return ""
        command = self.pretrain_command or "python scripts/pretrain_mgdp.py"
        return (
            f"任务 {self.alias!r} 需要先预训练感知模型：{command} "
            f"（输出 {self.pretrain_checkpoint or '默认检查点'}）"
        )


def capabilities(spec: TaskSpec) -> dict:
    """Machine-readable capability block exposed by /api/state."""
    return {
        "needs_pretrain": spec.requires_pretrain,
        "pretrain_command": spec.pretrain_command,
        "pretrain_checkpoint": spec.pretrain_checkpoint,
        "supports_curriculum": spec.supports_curriculum,
        "docs": spec.docs,
        "train_presets": spec.train_presets,
    }
