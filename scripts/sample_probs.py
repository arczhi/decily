"""Reference inference for a single decision question (torch, for cross-checks).

Usage:
    python scripts/sample_probs.py --config configs/rlcd_v5_distill.yaml \
        --ckpt runs/rlcd_v5_distill/ckpt_step3000.pt --out /tmp/ref.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from decision_model.data.collate import RouteBCollator  # noqa: E402
from decision_model.data.schema import DecisionExample, Option, Question  # noqa: E402
from decision_model.models.cross_encoder import (  # noqa: E402
    CrossEncoderConfig,
    CrossEncoderDecisionModel,
)

STATE = (
    "I ordered a laptop three weeks ago and it still hasn't arrived. "
    "The tracking page hasn't updated in 10 days and support keeps telling me to wait."
)
QUESTION = "Which department should handle this ticket?"
OPTIONS = ["returns", "billing", "shipping", "technical support"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = {k: v for k, v in cfg["model"].items() if k not in ("route", "init_ckpt")}
    mc["dtype"] = "bfloat16"
    model = CrossEncoderDecisionModel(CrossEncoderConfig(**mc))
    state = torch.load(args.ckpt, map_location="cpu")
    state = state["state"] if isinstance(state, dict) and "state" in state else state
    model.load_state_dict(state, strict=False)
    model.to("cuda").eval()

    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    coll = RouteBCollator(tok, **cfg["data"]["collator"])

    ex = DecisionExample(
        task="_inference",
        state=STATE,
        questions=[
            Question(
                text=QUESTION,
                options=[Option(id=o, text=o) for o in OPTIONS],
                answer_ids=[OPTIONS[0]],
                type="choice",
            )
        ],
    )
    batch = coll([ex])
    for f in batch.__dataclass_fields__:
        v = getattr(batch, f)
        if isinstance(v, torch.Tensor):
            setattr(batch, f, v.to("cuda"))
    with torch.no_grad():
        logits = model(batch)["logits"][0, 0].float().cpu()
    probs = torch.softmax(logits, dim=-1)
    out = {
        "state": STATE,
        "question": QUESTION,
        "options": OPTIONS,
        "probs": [float(p) for p in probs],
    }
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
