"""MLX inference for the Stage-1 Qwen3.5-2B decision model (Apple Silicon).

Backbone uses mlx-lm's official qwen3_5 implementation; the decision head
(attention pooling + MLP) is ours. Weights come from the bundle written by
scripts/export_stage1_mlx.py:

  model.language_model.*   <- merged backbone (original HF key names)
  pool.*, head.*           <- decision head

Usage:
    python mlx/decision_mlx_q35.py --model-dir models_mlx/stage1 \\
        --state "..." --question "..." --options "a,b,c" --temperature 1.0
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx

NEG = -1e9


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0))))


class DecisionModelQ35MLX:
    def __init__(self, model_dir: str):
        from mlx_lm.models.qwen3_5 import Model, ModelArgs

        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.m = Model(ModelArgs.from_dict(self.cfg))

        raw = dict(mx.load(os.path.join(model_dir, "model.safetensors")))
        self.head: dict = {}
        tower: dict = {}
        for k, v in raw.items():
            if k.startswith("model.language_model."):
                stem = k[len("model.language_model.") :]
                tower["language_model.model." + stem] = v
            else:
                self.head[k] = v
        tower = self.m.language_model.sanitize(dict(tower))
        self.m.load_weights(list(tower.items()))

    # ---- backbone ----
    def hidden(self, ids: list[int]) -> mx.array:
        h = self.m.language_model.model(mx.array([ids]))
        mx.eval(h)
        return h[0]

    # ---- decision head ----
    def score(self, ids: list[int]) -> float:
        h = self.hidden(ids).astype(mx.float32)
        scores = (h @ self.head["pool.proj.weight"].T).squeeze(-1) + self.head["pool.proj.bias"]
        w = mx.softmax(scores, axis=-1)
        pooled = (h * w[:, None]).sum(axis=0)
        ln_w, ln_b = self.head["head.0.weight"], self.head["head.0.bias"]
        mu = pooled.mean()
        var = ((pooled - mu) ** 2).mean()
        x = (pooled - mu) / mx.sqrt(var + 1e-5) * ln_w + ln_b
        x = _gelu(x @ self.head["head.1.weight"].T + self.head["head.1.bias"])
        out = (x @ self.head["head.4.weight"].T + self.head["head.4.bias"]).item()
        return float(out)

    def decide(
        self,
        state: str,
        question: str,
        options: list[str],
        tokenizer,
        temperature: float = 1.0,
        max_state_tokens: int = 256,
        max_question_tokens: int = 96,
        max_option_tokens: int = 64,
    ) -> list[tuple[str, float]]:
        s_ids = tokenizer(state, truncation=True, max_length=max_state_tokens)["input_ids"]
        q_ids = tokenizer(question, truncation=True, max_length=max_question_tokens)["input_ids"]
        logits = []
        for opt in options:
            c_ids = tokenizer(opt, truncation=True, max_length=max_option_tokens)["input_ids"]
            logits.append(self.score(s_ids + q_ids + c_ids))
        z = mx.array(logits) / temperature
        p = mx.softmax(z, axis=-1)
        mx.eval(p)
        return sorted(zip(options, p.tolist()), key=lambda t: -t[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--options", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = DecisionModelQ35MLX(args.model_dir)
    res = model.decide(
        args.state, args.question, [o.strip() for o in args.options.split(",")], tok,
        temperature=args.temperature,
    )
    print(json.dumps({o: round(p, 4) for o, p in res}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
