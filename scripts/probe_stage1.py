"""Probe features of Stage-1 expansion datasets."""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset  # noqa: E402

CANDIDATES = [
    ("allenai/hellaswag", None, "validation"),
    ("ybisk/piqa", None, "validation"),
    ("allenai/openbookqa", "main", "validation"),
    ("tau/commonsense_qa", None, "validation"),
    ("allenai/winogrande", "winogrande_xl", "validation"),
    ("allenai/qasc", None, "validation"),
    ("google/boolq", None, "validation"),
    ("nyu-mll/glue", "cola", "validation"),
    ("nyu-mll/glue", "qnli", "validation"),
    ("openlifescienceai/medmcqa", None, "validation"),
    ("GBaker/MedQA-USMLE-4-options", None, "test"),
    ("microsoft/wiki_qa", None, "validation"),
    ("ehovy/race", "high", "validation"),
    ("fancyzhx/yelp_polarity", None, "test"),
    ("SetFit/sst5", None, "test"),
    ("SetFit/20_newsgroups", None, "test"),
    ("mteb/tweet_emotion", None, "test"),
    ("google/civil_comments", None, "validation"),
    ("glaiveai/glaive-function-calling-v2", None, "train"),
    ("NousResearch/hermes-function-calling-v1", None, "train"),
    ("Team-ACE/ToolACE", None, "train"),
    ("abisee/cnn_dailymail", "3.0.0", "validation"),
    ("LabHC/bias_in_bios", None, "test"),
    ("ucirvine/sms_spam", None, "test"),
]

for repo, cfg, split in CANDIDATES:
    try:
        kw = {"split": split}
        ds = load_dataset(repo, cfg, **kw) if cfg else load_dataset(repo, **kw)
        cols = list(ds.features.keys())
        print(f"OK   {repo}/{cfg or '-'} n={len(ds)} cols={cols}")
        row = ds[0]
        for k in cols[:6]:
            v = str(row[k])
            if len(v) > 70:
                v = v[:70] + "..."
            print(f"       {k} = {v}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {repo}/{cfg or '-'} :: {str(e)[:90]}")
