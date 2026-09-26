# AGENTS.md

## Working style

- For technical problems, read papers/documentation for established solutions
  before editing code; avoid blind parameter tuning. Record the sources
  (paper/URL) in DESIGN.md or next to the code.
- Training/evaluation always runs on the remote GPU: `ssh decision-gpu`
  (RTX 5090 32G). Sync code with `scripts/sync_to_gpu.sh` to
  `/root/decision-model`; Python environment `/root/venvs/decision`; run long
  jobs in tmux and write logs to /root/*.log.
- Data downloads go through hf-mirror: `HF_ENDPOINT=https://hf-mirror.com`;
  pip uses the Tsinghua mirror `-i https://pypi.tuna.tsinghua.edu.cn/simple`.

## Local checks

- Unit tests: `PYTHONPATH=src .venv/bin/python -m pytest tests -q`
- Remote smoke test: `scripts/smoke_test.sh`
- Stage 1 train+eval: `scripts/run_stage1.sh` → `runs/*/eval_report.json`

## Design notes

See DESIGN.md (design/architecture). Core: Route B (explicit candidate scorer)
+ per-question centering against representation anisotropy (Ethayarajh 2019;
Su et al. 2021 whitening). The full training/evaluation/deployment record is in
TRAIN-REPORT.md (numbering 9.x, matching the commit history). The end-to-end
reproduction recipe is in TRAINING.md; the public model card is MODEL_CARD.md.
