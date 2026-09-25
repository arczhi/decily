"""Build salience training examples from existing article shards (no gold labels).

Reads decision-tuple JSONL shards, takes each example's `state` as an article,
splits it into sentences, samples up to `max_candidates` sentences (biased toward
earlier ones), and emits a CHOICE question:

    "Which sentence is most important for understanding this text?"

Gold is a placeholder (first option) and is replaced by the teacher's argmax in
the precompute step (`precompute_teacher.py --argmax-gold`), so the KD/CE signal
comes from the teacher, not from a fabricated label.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_model.data.schema import (  # noqa: E402
    DecisionExample,
    Option,
    Question,
    read_jsonl,
    write_jsonl,
)

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SPLIT.split(text) if len(s.strip()) >= 30]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--max-state-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    for path in args.inputs:
        for ex in read_jsonl(path):
            if len(out) >= args.limit:
                break
            state = ex.state[: args.max_state_chars]
            sents = sentences(state)
            if len(sents) < 5:
                continue
            head = sents[: max(4, args.max_candidates // 2)]
            tail = sents[len(head):]
            rng.shuffle(tail)
            chosen = head + tail[: max(0, args.max_candidates - len(head))]
            rng.shuffle(chosen)
            options = [Option(id=f"s{i}", text=s) for i, s in enumerate(chosen)]
            out.append(
                DecisionExample(
                    task="salience_self",
                    state=state,
                    questions=[
                        Question(
                            text="Which sentence is most important for understanding this text?",
                            options=options,
                            answer_ids=[options[0].id],  # placeholder, teacher argmax later
                            type="choice",
                        )
                    ],
                    meta={"src_task": ex.task, "n_sentences": len(sents)},
                )
            )
        if len(out) >= args.limit:
            break

    n = write_jsonl(args.out, iter(out))
    print(f"[salience] wrote {n} examples -> {args.out}")


if __name__ == "__main__":
    main()
