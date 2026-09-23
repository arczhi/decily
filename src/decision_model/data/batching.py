"""Length-aware batching (torch-free so it is locally testable)."""

from __future__ import annotations

from typing import Iterable, Iterator

from .schema import DecisionExample


def _approx_len(ex: DecisionExample) -> int:
    n = len(ex.state)
    for q in ex.questions:
        n += len(q.text) + sum(len(o.render()) for o in q.options)
    return n


def batched(
    it: Iterable[DecisionExample],
    size: int,
    bucket: bool = False,
    buffer_mult: int = 8,
) -> Iterator[list[DecisionExample]]:
    """Batch examples; optionally sort buffers by length to cut padding."""
    batch: list[DecisionExample] = []
    if not bucket:
        for ex in it:
            batch.append(ex)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch
        return

    for ex in it:
        batch.append(ex)
        if len(batch) >= size * buffer_mult:
            batch.sort(key=_approx_len)
            for i in range(0, len(batch), size):
                yield batch[i : i + size]
            batch = []
    if batch:
        batch.sort(key=_approx_len)
        for i in range(0, len(batch), size):
            yield batch[i : i + size]
