"""Probe features/splits of candidate fair-suite datasets."""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset  # noqa: E402

CANDIDATES = [
    ("google-research-datasets/paws", "labeled_final", "test"),
    ("allenai/sciq", None, "test"),
    ("CogComp/trec", None, "test"),
    ("allenai/social_i_qa", None, "validation"),
    ("tweet_eval", "irony", "test"),
    ("juletxara/xstory_cloze", "en", "test"),
    ("truthful_qa", "multiple_choice", "validation"),
    ("qiaojin/PubMedQA", "pqa_labeled", "train"),
    ("SetFit/bbc-news", None, "test"),
    ("emozilla/quality", None, "validation"),
    ("mteb/financial_phrasebank", None, "test"),
    ("databricks/databricks-dolly-15k", None, "train"),
]

for repo, cfg, split in CANDIDATES:
    try:
        kw = {"split": split}
        ds = load_dataset(repo, cfg, **kw) if cfg else load_dataset(repo, **kw)
        print(f"OK   {repo}/{cfg or '-'} split={split} n={len(ds)}")
        print(f"     cols={list(ds.features.keys())}")
        row = ds[0]
        for k in ("label", "label_text", "coarse_label", "fine_label", "answer", "correct_answer", "labels"):
            if k in row:
                print(f"     {k} = {str(row[k])[:80]}")
        if "choices" in row:
            print(f"     choices = {str(row['choices'])[:100]}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {repo}/{cfg or '-'} :: {str(e)[:100]}")
