"""Helpers for distribution-shift evaluation protocols."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class ProtocolConfig:
    """Declarative description of an evaluation protocol."""

    name: str
    history_mode: str
    eval_split: str
    requires_finetune: bool = False


@dataclass
class ShiftEvalExample:
    """A single rolling prediction example."""

    user: int
    history: List[int]
    target: int
    source_split: str


def load_item_text_map(data_dir: Path) -> Dict[int, str]:
    """Load optional item text metadata."""

    path = data_dir / "item_text.json"
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    result: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        text = str(value).strip()
        if text:
            result[idx] = text
    return result


def _build_user_timelines(records: Sequence[Mapping]) -> Dict[int, List[int]]:
    """Aggregate per-user item sequences preserving chronological order."""

    from collections import defaultdict

    events: MutableMapping[int, List[tuple[int, int]]] = defaultdict(list)
    for rec in records:
        try:
            user = int(rec.get("user", -1))
        except (TypeError, ValueError):
            continue
        if user < 0:
            continue
        items = list(rec.get("items", []))
        if not items:
            continue
        times = rec.get("times")
        if isinstance(times, Iterable):
            try:
                times = [int(t) for t in times]
            except (TypeError, ValueError):
                times = None
        else:
            times = None

        if times and len(times) == len(items):
            pairs = list(zip(times, items))
        else:
            start = len(events[user])
            pairs = [(start + idx, item) for idx, item in enumerate(items)]
        events[user].extend(pairs)

    timelines: Dict[int, List[int]] = {}
    for user, pairs in events.items():
        ordered = sorted(pairs, key=lambda pair: pair[0])
        timelines[user] = [item for _, item in ordered]
    return timelines


def build_user_timelines(splits: Mapping[str, Sequence[Mapping]]) -> Dict[str, Dict[int, List[int]]]:
    """Return ``split -> user -> sequence`` maps."""

    return {name: _build_user_timelines(records) for name, records in splits.items()}


def build_protocol_examples(
    protocol: str,
    timelines: Mapping[str, Dict[int, List[int]]],
    history_window: int,
) -> List[ShiftEvalExample]:
    """Construct evaluation examples for the requested protocol."""

    protocol = protocol.upper()
    original = timelines.get("original", {})
    finetune = timelines.get("finetune", {})
    test = timelines.get("test", {})

    examples: List[ShiftEvalExample] = []

    if protocol == "O-TAIL":
        for user, sequence in original.items():
            tail_ref = len(test.get(user, []))
            if tail_ref <= 0:
                tail_ref = min(len(sequence), history_window)
            start = max(1, len(sequence) - tail_ref)
            for idx in range(start, len(sequence)):
                history = sequence[max(0, idx - history_window) : idx]
                if not history:
                    continue
                examples.append(
                    ShiftEvalExample(user=user, history=list(history), target=sequence[idx], source_split="original")
                )
    elif protocol == "T-NOCONCAT":
        for user, test_sequence in test.items():
            if not test_sequence:
                continue
            history = original.get(user, [])[-history_window:]
            if not history:
                continue
            for item in test_sequence:
                examples.append(ShiftEvalExample(user=user, history=list(history), target=item, source_split="test"))
    elif protocol in {"T-CONCAT", "T-FINETUNE"}:
        for user, test_sequence in test.items():
            if not test_sequence:
                continue
            history = list(original.get(user, []))
            history.extend(finetune.get(user, []))
            for item in test_sequence:
                window = history[-history_window:]
                if not window:
                    history.append(item)
                    continue
                examples.append(ShiftEvalExample(user=user, history=list(window), target=item, source_split="test"))
                history.append(item)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return examples


__all__ = [
    "ProtocolConfig",
    "ShiftEvalExample",
    "build_protocol_examples",
    "build_user_timelines",
    "load_item_text_map",
]
