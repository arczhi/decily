"""Probe candidate datasets for Stage 2 mixture (availability, size, labels)."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CANDIDATES = [
    ("mteb/amazon_counterfactual", "en", "test"),
    ("mteb/amazon_polarity", None, "test"),
    ("mteb/amazon_reviews_multi", "en", "test"),
    ("mteb/imdb", None, "test"),
    ("mteb/tweet_sentiment_extraction", None, "test"),
    ("mteb/poem_sentiment", None, "test"),
    ("mteb/sickr", None, "test"),
    ("mteb/toxic_conversations", None, "test"),
    ("mteb/mtop_domain", None, "test"),
    ("mteb/mtop_intent", None, "test"),
    ("mteb/amazon_massive_scenario", "en", "test"),
    ("mteb/tweet_topic_single", None, "test"),
    ("fancyzhx/dbpedia_14", None, "test"),
    ("cais/mmlu", "high_school_world_history", "test"),
    ("nyu-mll/glue", "rte", "validation"),
    ("nyu-mll/glue", "qqp", "validation"),
    ("nyu-mll/glue", "mrpc", "validation"),
    ("allenai/ai2_arc", "ARC-Challenge", "validation"),
]

from datasets import load_dataset  # noqa: E402

for name, config, split in CANDIDATES:
    try:
        kw = {"split": split}
        ds = load_dataset(name, config, **kw) if config else load_dataset(name, **kw)
        cols = list(ds.features.keys())
        n = len(ds)
        label_info = ""
        for col in ("label_text", "label", "intent", "answer", "answerKey", "choices"):
            if col in cols:
                if col == "label_text":
                    uniq = len(set(ds[: min(4000, n)]["label_text"]))
                    label_info = f"label_text~{uniq}"
                elif col == "label":
                    f = ds.features["label"]
                    label_info = (
                        f"ClassLabel[{f.num_classes}]"
                        if hasattr(f, "num_classes")
                        else f"Value[{getattr(f, 'dtype', '?')}]"
                    )
                else:
                    label_info = f"{col}:{getattr(ds.features[col], 'dtype', '?')}"
                break
        print(f"OK   {name}/{config or '-'} split={split} n={n} cols={cols[:6]} {label_info}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}/{config or '-'} :: {str(e)[:110]}")
