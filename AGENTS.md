# AGENTS.md

## 工作方式

- 遇到技术问题时，优先读论文/查文献找成熟方案，再动手改代码；
  避免纯靠反复试参的硬调。记录依据（论文/来源）到 DESIGN.md 或代码注释旁。
- 训练/评测一律在远端 GPU 跑：`ssh decision-gpu`（RTX 5090 32G），
  代码用 `scripts/sync_to_gpu.sh` 同步到 `/root/decision-model`；
  Python 环境 `/root/venvs/decision`；长任务放 tmux，日志写 /root/*.log。
- 数据下载走 hf-mirror：`HF_ENDPOINT=https://hf-mirror.com`；
  pip 用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 本地验证

- 本地单测：`PYTHONPATH=src .venv/bin/python -m pytest tests -q`
- 远端冒烟：`scripts/smoke_test.sh`
- Stage 1 训练+评测：`scripts/run_stage1.sh` → `runs/*/eval_report.json`

## 设计要点

见 DESIGN.md。核心：Route B（显式 candidate scorer）+ per-question
centering 对抗表示各向异性（Ethayarajh 2019; Su et al. 2021 whitening）。
