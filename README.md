# BrainVLN

BrainVLN 在 VLN-CE 的 R2R 上做连续视觉语言导航。冻结的视觉和语言前端把 RGB、深度和指令变成电流，后面的脉冲群体用固定延迟的突触读出动作。

仓库里还留着更早的 SpikingNav RoboTHOR PointNav / ObjectNav 代码（`main.py`、`spikingnav/habitat_objectnav/`、`scripts/train.sh`）。当前任务是下面这条 R2R 线。

## 任务

环境是 Matterport3D 上的 R2R-CE。智能体看到 224×224 的 RGB、深度和一句自然语言指令。目标坐标只用来记分，不进入模型。

动作是 `STOP`、`MOVE_FORWARD`（0.25 m）、`TURN_LEFT`、`TURN_RIGHT`（15°）。成功的定义是智能体主动停下，并且离目标的测地距离不超过 3 m。单回合最多 500 步。

要报的指标是官方 `val_unseen` 上的闭环 SR、SPL、导航误差和 nDTW。离线模仿的动作交叉熵，以及采集场景上的 teacher forcing，都只说明训练是否在拟合专家动作，不能代替这条闭环结果。

## 模型

可训练的是突触强度和把特征压到细胞通道的 1×1 卷积。延迟、兴奋和抑制的符号保持固定。编码器冻结。没有 GRU。

| 名称 | 结构 |
| --- | --- |
| BrainVLN（实验里的 v1） | 整图池化。CLIP RGB 与 DD-PPO 深度各 256 维，驱动 512 个 Spatial 神经元 |
| BrainVLN_v2 | B4 cells。RGB 保持 3×3，深度保持 4×4，各压到 32 通道。Spatial 800，Goal 512，Episodic 1280，Action 512 |
| BrainVLN_v3 | 与 v2 相同，另外在每个内部 LIF 步上对兴奋性突触做 NeuCube 风格的在线 STDP |
| BrainVLN_v4 | 与 v2 相同的细胞数。RGB 换成冻结的 Meta-SpikeFormer，深度和语言仍用 MLANet |
| BrainVLN_v5 | 保持 v2 的 MLANet 细胞布局。Episodic 使用 τ=30 的慢泄漏，并在视角变化时写入。Goal 的后 256 个神经元是 12 条子指令上的指针。另外 64 个慢神经元只读 Episodic 和 Goal，加到 STOP 的 logit 上 |

当前主线是 v2 和 v5。语言侧沿用 MLANet 的指令 LSTM 和 CLIP 子指令。

## 数据

离线模仿读 `storage/collect_data/train/`，这条目录只读。里面是已经展开的 R2R-CE 训练轨迹：61 个场景，动作与上面的四动作表一致。按场景名字典序，最后 6 个场景是内部 holdout，其余用于开发训练。这不是官方 Val-Seen 或 Val-Unseen。字段约定写在 `docs/brain_vln/data_contract.md`。

闭环评估和 DAgger 使用 VLN-CE 的 R2R 指令与 Matterport3D 网格。场景目录是 `data/scene_datasets`。预训练权重放在 `data/pre_trained_model/`。

`storage/`、`datasets/`、`third_party/`、权重和日志不进 git。

## 环境

BrainVLN 使用单独的 conda 环境，装在仓库内的 `storage/brain_vln/envs/BrainVLN`。Python 3.9.23，Habitat-Sim 0.3.3（headless、无 Bullet），PyTorch `2.6.0+cu126`，torchvision `0.21.0+cu126`。闭环评估需要这个 Habitat-Sim；只跑离线模仿时，仿真器可以先不调用。

`environment.yml` 和 `scripts/setup_env.sh` 对应更早的 `spikingnav` 环境（AllenAct / AI2-THOR）。配置 BrainVLN 时按下面的命令，不要用那份 yml。

在 Linux x86_64 上，从仓库根目录执行。`--prefix` 把环境放进本仓库，换机器时路径跟着项目走。

```bash
conda create -y -p storage/brain_vln/envs/BrainVLN \
  -c aihabitat -c conda-forge \
  python=3.9 \
  habitat-sim=0.3.3=py3.9_headless_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24 \
  matplotlib
conda activate storage/brain_vln/envs/BrainVLN

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-brainvln.txt
```

那个 Habitat-Sim 构建串就是当前环境的包。若频道里已经没有这条构建，改成同一版本的 headless 变体：

```bash
conda create -y -p storage/brain_vln/envs/BrainVLN \
  -c aihabitat -c conda-forge \
  python=3.9 habitat-sim=0.3.3 headless matplotlib
```

`habitat` 这个 Python 包来自 Habitat-Lab 0.3.3，本机是可编辑安装。BrainVLN 的 R2R 闭环走 `habitat_sim`，不依赖 Lab 的 Env。需要 Lab 时，把 0.3.3 源码放到 `third_party/` 再安装：

```bash
git clone --depth 1 --branch v0.3.3 \
  https://github.com/facebookresearch/habitat-lab.git \
  third_party/habitat-lab
pip install -e third_party/habitat-lab/habitat-lab
```

装完后核对版本，并确认能看到 GPU：

```bash
python -c "import torch, habitat_sim; print(torch.__version__, torch.cuda.is_available(), habitat_sim.__version__)"
```

期望输出里有 `2.6.0+cu126`、`True` 和 `0.3.3`。驱动需要能跑 CUDA 12.6 的轮子。

以后在仓库根目录运行，并带上 `PYTHONPATH=.`：

```bash
export PYTHONPATH="$PWD"
PY=storage/brain_vln/envs/BrainVLN/bin/python
```

不启动仿真器时，可以只跑模型和数据测试：

```bash
"$PY" -m pytest -q tests/brain_vln tests/vln
```

## 训练

v2 离线模仿。输出目录必须是新的；目录已经存在且没有 `--resume` 时，脚本会拒绝启动。

```bash
"$PY" -u scripts/brain_vln/train_il.py \
  --layout cells \
  --epochs 100 --batch-size 128 --microbatch 32 \
  --output storage/brain_vln/experiments/BrainVLN_v2
```

v1 去掉 `--layout cells`。v3 在 cells 上加 `--stdp`。v4 使用 `--layout cells --encoder metaspikeformer`。

v5 直接改 `configs/brain_vln_il.yaml`，再用 Hydra 入口。临时覆盖写在命令末尾，例如 `epochs=1`。

```bash
"$PY" -u scripts/brain_vln/train_v5.py
```

EnvDrop 和 DAgger 从一份 v2 检查点接着训。日程与 MLANet 的发布配置一致：EnvDrop 是 45 个 epoch 的 teacher forcing；DAgger 是 10 轮、每轮 4 个 epoch，每轮追加训练场景上的滚动轨迹，专家动作混入概率为 `0.5 ** (iteration + 1)`。`--dataset` 指向本机的 R2R-CE 指令数据。

```bash
"$PY" -u scripts/brain_vln/train_envdrop_dagger.py \
  --init storage/brain_vln/experiments/BrainVLN_v2/ckpt_epoch_89.pt \
  --output storage/brain_vln/experiments/BrainVLN_v2_envdrop \
  --dataset /path/to/R2R_VLNCE_FSASub \
  --scenes data/scene_datasets \
  --stage envdrop --gpus 0
```

把 `--stage envdrop` 换成 `dagger` 即进入 DAgger。采集和训练的检查点写到 `--output` 下面。

## 评估

官方 `val_unseen` 闭环评估。仿真失败或指令失败记入该条 episode，不把它算成成功。

```bash
"$PY" -u scripts/brain_vln/eval_val_unseen.py \
  --checkpoint storage/brain_vln/experiments/BrainVLN_v2/ckpt_epoch_89.pt \
  --output storage/brain_vln/experiments/BrainVLN_v2_val_unseen/update_000089 \
  --update 89 --gpu 0
```

全部 episode 跑完后，输出目录里的 `summary.json` 给出 SR、SPL、NE 和 nDTW。

## 目录

```
spikingnav/brain_vln/     脉冲模型、采集数据加载、闭环演员
spikingnav/vlnce/         R2R-CE 动作、仿真、SR / SPL / nDTW
scripts/brain_vln/        离线模仿、EnvDrop、DAgger、val_unseen
configs/brain_vln_il.yaml BrainVLN_v5 的训练配置
docs/brain_vln/           数据契约和结构记录
storage/                  采集数据、检查点、日志（不进 git）
data/                     场景网格与预训练权重（大文件不进 git）
```
