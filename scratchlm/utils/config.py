"""Dataclass configs loaded from YAML."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True


@dataclass
class TokenizerConfig:
    vocab_size: int = 4096
    special_tokens: tuple[str, ...] = ("<|endoftext|>",)


def load_config(cls: Type[T], path: str) -> T:
    """Load a YAML file into dataclass `cls`. Unknown keys raise; missing keys use defaults."""
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def save_config(cfg: Any, path: str) -> None:
    """Dump a dataclass config to YAML (written next to each run's metrics)."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
