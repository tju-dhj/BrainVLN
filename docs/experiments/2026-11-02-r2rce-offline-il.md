# 2026-11-02 R2R-CE 离线模仿学习接线

这次把 SSE + SPN 接到 VLN-CE 的 R2R 连续环境。监督信号是 `R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz` 里已经展开好的低层专家动作，也就是 VLN-CE 的 teacher forcing。训练时不再进仿真器。这不是在线 DAgger，也不是 HM3D ObjectNav 的 34.1% 目标。

动作是 STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT，前进 0.25 m，转向 15°，成功条件是主动停止且离目标测地距离不超过 3 m。指令用数据集自带的 GloVe token，编成 32 维向量后送进 SSE 原来的目标通路。损失是 VLN-CE 的拐点加权交叉熵，系数 3.2。

数据入口：

- `data/datasets/R2R_VLNCE_v1-3_preprocessed` → `/share/home/u14004/VLN/datasets/R2R_VLNCE_v1-3_preprocessed`
- 场景仍用 `data/scene_datasets/mp3d`。抽查 train 40 条、val_unseen 20 条，网格都在。

正式三步，输出目录必须是新的：

```bash
export PYTHONPATH="$PWD"
# 采集。登录节点没有可用的 CUDA EGL，--gpu 0 会直接退出；这里用 CPU。
/share/home/u14004/.conda/envs/streamvln/bin/python -B scripts/collect_r2r_gt.py \
  --split train --gpu -1 --output storage/r2rce_gt_YYYYMMDD

# 训练。ImageNet 初始化，30 epoch，学习率 2.5e-4，batch 2。
storage/envs/spikingnav/bin/python -B scripts/train_r2r_offline_il.py \
  --data storage/r2rce_gt_YYYYMMDD/rollouts \
  --output storage/r2rce_il_YYYYMMDD

# 评估 val_unseen。streamvln 有 habitat-sim，torchvision 在 spikingnav 环境里。
PYTHONPATH="$PWD:storage/envs/spikingnav/lib/python3.9/site-packages" \
/share/home/u14004/.conda/envs/streamvln/bin/python -B scripts/eval_r2r.py \
  --checkpoint storage/r2rce_il_YYYYMMDD/ckpt_last.pt \
  --split val_unseen --gpu -1 \
  --output storage/r2rce_eval_YYYYMMDD
```

有 GPU 的计算节点把 `--gpu -1` 改成 `--gpu 0`。评估报告 SR、SPL、导航误差、nDTW 和 oracle SR。

2026-11-02 在登录节点只做了通路检查，不能当成 R2R 成绩：

- CPU 回放了 train 的 1 条回合（episode 1，场景 `7y3sRwLe3Va`，39 步，专家最后一步是 STOP），写在 `storage/r2rce_gt_smoke_cpu_20261102/`。
- 用这 39 步里的前 4 步、随机初始化、CPU、1 个 epoch 跑通训练，损失约 1.3865，等于四分类均匀分布的交叉熵。检查点在 `storage/r2rce_il_smoke_20261102/`。
- 用这个未训练检查点在同一条训练回合上只走了 5 步，用来确认评估脚本能写出 `results.json`。`storage/r2rce_eval_smoke_20261102/` 里的 SR 不是验证集结果。
- 同一场景上，起点到目标的测地距离测得 6.61 m，数据集记录是 6.43 m。
- `storage/r2rce_gt_smoke_20261102/` 是 `--gpu 0` 的失败记录：EGL 找不到 CUDA device 0。
