# SpikingNav 仿真复现

这是 [SpikingNav: Robust Embodied Navigation with Spiking Neural Policies](https://arxiv.org/abs/2608.05078) 的 **RoboTHOR 仿真复现**。论文没有官方代码，补充材料 S1 也未公开。本仓库按正文公式重实现 SSE + SPN，并接入 AllenAct / RobustNav 风格的 PointNav、ObjectNav 训练与视觉腐蚀评估。

**不包含** Thruster-V2 神经形态芯片部署。这里的前向是 PyTorch 上的稠密卷积（同一张图循环 `T=4`），墙钟时间主要花在 AI2-THOR 渲染和 PPO 反传上，不能当作芯片推理速度。

## 方法

- **SSE**：SEW-ResNet18 风格脉冲骨干（内部步数 `T=4`）+ 两层卷积压缩（512 → 128 → 32）+ 目标嵌入（32 维）+ 两层融合，输出 `r_t ∈ R^{1568}`。
- **SPN**：`v_t = λ u_{t-1} + W_r r_t + W_h u_{t-1}`，脉冲门控硬复位 `u_t = v_t ⊙ (1 - s_t)`，膜电位限制在 `±10ϑ`，再用线性头输出 actor / critic。
- **ANNNav**：匹配的 ResNet18 + GRU(512) 基线。
- **训练**：AllenAct PPO。`clip=0.1`，`value_loss_coef=0.5`，`entropy_coef=0.01`，`lr=3e-4` 线性衰减，`rollout=128`，`update_repeats=4`，`γ=0.99`，`gae_λ=0.95`。PointNav 7500 万步，ObjectNav 3 亿步。每 500 万步存一次检查点。

未公开超参写在 `spikingnav/config.py`：`hidden=512`，`λ=0.5`，`ϑ=1.0`，代理梯度为 arctan（`α=2`）。骨干用 ImageNet ANN 权重初始化 SEW-ResNet，而不是完整 BuSNN（BuSNN 同样无代码）。循环权重 `W_h` 用较小正交增益初始化，避免 128 步展开时膜电位溢出。

环境：RoboTHOR Locobot，RGB 400×300 再缩到 224，步长 0.25 m，转向 30°，单回合最多 500 步，水平视场 79°。ObjectNav 成功距离（可见性）1.0 m，动作含 `LookUp` / `LookDown`，12 类目标：AlarmClock、Apple、BaseballBat、BasketBall、Bowl、GarbageCan、HousePlant、Laptop、Mug、SprayBottle、Television、Vase。PointNav 成功距离 0.2 m，动作为 MoveAhead / RotateLeft / RotateRight / End。

## 目录

```
main.py                 AllenAct 入口，额外接受 -vc / -vs
spikingnav/             LIF、SSE、SPN、ANNNav、腐蚀、实验配置
projects/               AllenAct 能发现的实验类
scripts/                环境、数据、THOR 下载、训练、评估
tests/                  单测（不启动模拟器）
datasets/               数据下载位置（不进 git）
storage/                检查点、TensorBoard、报错现场（不进 git）
third_party/robustnav   RobustNav 评估集来源（不进 git）
```

## 环境

需要 Linux、已安装的 Miniconda 或 Anaconda、NVIDIA 驱动，以及和驱动主版本一致的 Vulkan（AI2-THOR CloudRendering 用它画图）。环境名是 `spikingnav`，Python 3.9。本机验证过的组合是 Python 3.9.23 与 `torch 2.8.0+cu128`。

还没有 conda 时，先装 Miniconda，装完重新打开终端：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

在仓库根目录创建环境并装上仿真栈。`moviepy` 必须是 `1.0.3`，2.x 会让 AllenAct 的 `from moviepy import editor` 失败。

```bash
cd /path/to/SNN-Nav
conda create -n spikingnav python=3.9 -y
conda activate spikingnav

# 与本机驱动匹配的 CUDA 12.8 轮子。驱动更旧时，把 cu128 换成对应的 CUDA 版本。
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e .
pip install 'ai2thor>=2.7.4' 'allenact' 'allenact_plugins[ithor]' 'moviepy==1.0.3'
pip install tensorboard
```

`requirements.txt` 里的 `torch>=1.12` 有时会把上面的 CUDA 轮子换成 PyPI 上的另一构建。装完后确认版本仍带 `+cu128`，并且能看到 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

若打印出来的不是 `2.8.0+cu128`，把 `pip install torch==2.8.0 ... cu128` 那一行再执行一次。

`environment.yml` 是当前可用的 `spikingnav` 环境重新导出的版本钉死清单，包含 Python 3.9.23、`torch==2.8.0`、`torchvision==0.23.0` 以及 CUDA 12.8 的 `nvidia-*-cu12` 轮子。文件里没有本机安装路径。换机器时：

```bash
conda env create -f environment.yml
conda activate spikingnav
pip install -e .
```

也可以用脚本一次做完。它按 `environment.yml` 创建 `spikingnav`（已存在则跳过创建），然后执行 `pip install -e .` 以及 AllenAct / AI2-THOR。GPU 训练前仍要核对 `torch.__version__` 带有 `+cu128`。

```bash
bash scripts/setup_env.sh
conda activate spikingnav
```

以后每次开新终端：

```bash
conda activate spikingnav
cd /path/to/SNN-Nav
```

只检查模型和单测、不启动模拟器时：

```bash
pytest -q
python scripts/smoke_test.py
```

## 数据与渲染器

```bash
bash scripts/download_robothor.sh all
```

会下载 AllenAct 的 RoboTHOR PointNav / ObjectNav 训练集，并浅克隆 [allenai/robustnav](https://github.com/allenai/robustnav)，把其中的 `robustnav_eval` 链接到 `datasets/robothor-{pointnav,objectnav}/robustnav_eval`。只要其中一项时，把 `all` 换成 `robothor-pointnav` 或 `robothor-objectnav`。

首次训练前还要有 AI2-THOR CloudRendering 包（约 800MB，commit `f0825767cd50d69f666c7f282e54abfe58f1e917`）。`scripts/train.sh` 会调用 `scripts/download_thor.sh`，用 `aria2c`（没有则用 `wget -c`）断点续传到 `~/.ai2thor/releases/`。不要等 AllenAct 自己下：它的环境启动超时是 300 秒，慢速下载会被杀掉。

默认无头 CloudRendering，并会清掉 `DISPLAY`。要用本机 X11 时：

```bash
SPIKINGNAV_USE_X11=1 bash scripts/train.sh objectnav spiking smoke
```

多张 GPU 时，NVIDIA Vulkan ICD 有时会让同一张卡出现两次，AI2-THOR 会在 `controller.py` 里断言失败。`scripts/write_cuda_vulkan_map.py` 会写 `~/.ai2thor/cuda-vulkan-mapping.json`，只保留每个 UUID 的第一个 Vulkan 设备。没有 Vulkan 设备的 CUDA 卡不会被拿来训练。

## 训练

在仓库根目录、已 `conda activate spikingnav` 的前提下：

```bash
bash scripts/train.sh <objectnav|pointnav> <spiking|ann> <full|smoke> [seed]
```

`seed` 默认 `12345`。

短程冒烟（1 个进程、128 步、只用 0 号卡），用来确认模拟器和反传能走通：

```bash
bash scripts/train.sh objectnav spiking smoke
```

论文设定的完整训练：

```bash
bash scripts/train.sh objectnav spiking full   # 3 亿步
bash scripts/train.sh pointnav spiking full    # 7500 万步
bash scripts/train.sh objectnav ann full
bash scripts/train.sh pointnav ann full
```

完整模式若没有设置 `CUDA_VISIBLE_DEVICES`，脚本会改用 Vulkan 能对应上的物理卡，并重新编号成 PyTorch 的 `0,1,2,...`。机器上别的任务占着卡时，只绑空闲卡，并按显存把并行环境数降下来，例如单卡 8 个环境：

```bash
CUDA_VISIBLE_DEVICES=0 SPIKINGNAV_GPU_IDS=0 SPIKINGNAV_NUM_PROCESSES=8 \
  bash scripts/train.sh objectnav spiking full
```

8 个环境、1 张 4090、ObjectNav 3 亿步，按目前的实测外推大约是大半年到一年。4 张空闲 4090、默认 60 个环境，粗估 1.5–4 个月。日志里出现 FPS 后，用 `300000000 / FPS / 86400` 估算剩余天数。

### 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `SPIKINGNAV_NUM_PROCESSES` | 60（smoke 为 1） | 并行仿真进程数 |
| `SPIKINGNAV_MAX_STEPS` | ObjectNav 3e8 / PointNav 7.5e7（smoke 为 128） | 总环境步数，所有进程合计 |
| `SPIKINGNAV_GPU_IDS` | 全部可见卡的本地编号 | 交给 AllenAct 的设备号 |
| `CUDA_VISIBLE_DEVICES` | 完整模式自动设成有 Vulkan 的卡 | 本进程能看见的物理卡 |
| `SPIKINGNAV_VIS_CHUNK` | 4 | 视觉编码的梯度检查点块大小，用来避免 24GB 卡上 PPO 反传 OOM。设为 0 则一次前向整批 |
| `SPIKINGNAV_USE_X11` | 0 | 设为 1 时用显示器而不是 CloudRendering |

### 输出位置

一次 ObjectNav Spiking 完整训练的标签是 `rnav_objectnav_spiking_full`，目录名带启动时间，例如 `2026-09-23_02-06-34`。

- 检查点：`storage/objectnav-spiking-rgb/checkpoints/Objectnav-RoboTHOR-SpikingNav-RGB-DDPPO/rnav_objectnav_spiking_full/<时间戳>/exp_...__steps_<步数>.pt`
- TensorBoard：`storage/objectnav-spiking-rgb/tb/Objectnav-RoboTHOR-SpikingNav-RGB-DDPPO/rnav_objectnav_spiking_full/<时间戳>/`
- 本次实际使用的配置副本：`storage/objectnav-spiking-rgb/used_configs/.../<时间戳>/`

PointNav、ANN 只是把路径里的 `objectnav` / `spiking` 换成对应名字。检查点大约每 500 万步写一次，同时在该检查点上跑验证集。训练日志大约每 1 万步记一次 success、SPL、reward、回合长度、PPO loss 和 FPS。

当前环境写事件用的是 `tensorboardX`。要在浏览器里看曲线，需要再装查看器：

```bash
pip install tensorboard
tensorboard --logdir storage/objectnav-spiking-rgb/tb --port 6006
```

### 断点续训

`scripts/train.sh` 不接收检查点参数。500 万步之前中断，没有 `.pt` 可以续。有检查点之后，用和开训时相同的 `CUDA_VISIBLE_DEVICES`、`SPIKINGNAV_GPU_IDS`、`SPIKINGNAV_NUM_PROCESSES`，在 `python main.py` 上加 `-c`：

```bash
CUDA_VISIBLE_DEVICES=0 SPIKINGNAV_GPU_IDS=0 SPIKINGNAV_NUM_PROCESSES=8 \
python main.py \
  -o storage/objectnav-spiking-rgb \
  -b projects/spikingnav_baselines/experiments \
  objectnav_robothor_spiking_ddppo \
  -s 12345 \
  --extra_tag rnav_objectnav_spiking_full \
  -c storage/objectnav-spiking-rgb/checkpoints/Objectnav-RoboTHOR-SpikingNav-RGB-DDPPO/rnav_objectnav_spiking_full/<时间戳>/exp_....pt
```

续训会从检查点里的步数接着走到原定总步数，并新建一个时间戳目录，不覆盖旧日志。文件名以 `error_for_exp_` 开头的是 NaN 现场，里面的权重不可用。

## 评测

干净验证集（`robustnav_eval`）：

```bash
bash scripts/eval.sh objectnav spiking path/to/checkpoint.pt
bash scripts/eval.sh pointnav ann path/to/checkpoint.pt
```

单一腐蚀，严重程度默认 5：

```bash
bash scripts/eval.sh objectnav spiking path/to/checkpoint.pt "Motion Blur" 5
```

七种腐蚀各跑一遍，最后再跑一遍干净集：

```bash
bash scripts/eval_corruptions.sh objectnav spiking path/to/checkpoint.pt 5
```

腐蚀名称与 RobustNav 一致：`Low Lighting`、`Motion Blur`、`Camera Crack`、`Defocus Blur`、`Speckle Noise`、`Lower FOV`、`Spatter`。主指标是 SR 与 SPL。结果写到 `storage/<task>-<model>-eval/`。

`main.py` 在交给 AllenAct 之前吃掉 `-vc` / `-vs`，并写成 `SPIKINGNAV_CORRUPTION`、`SPIKINGNAV_SEVERITY`。实验名是：

- `objectnav_robothor_spiking_ddppo`
- `objectnav_robothor_ann_ddppo`
- `pointnav_robothor_spiking_ddppo`
- `pointnav_robothor_ann_ddppo`

直接调用时要带上仓库根目录的 `PYTHONPATH`，例如：

```bash
export PYTHONPATH="$PWD"
python main.py \
  -o storage/objectnav-spiking-eval \
  -b projects/spikingnav_baselines/experiments \
  objectnav_robothor_spiking_ddppo \
  -c path/to/checkpoint.pt \
  --eval \
  -s 12345 \
  --extra_tag eval_objectnav_spiking \
  -vc "Speckle Noise" -vs 5
```

## Habitat HM3D ObjectNav

论文实验本身在 RoboTHOR 上。HM3D / Gibson / MP3D 的 ObjectNav 回合在 `datasets/objectnav`，网格在 `datasets/scene_datasets`。这一路用 Habitat 3（habitat-lab 与 habitat-sim **0.3.1**）跑 SpikingNav 的 SSE + SPN，不跑 PONI 的势函数。HM3D 的 6 类目标是 chair、bed、plant、toilet、tv_monitor、sofa。动作是 Habitat ObjectNav 的 stop / move_forward / turn_left / turn_right / look_up / look_down。成功判定沿用 Habitat：停在可见目标的 viewpoint 0.1 m 内。

Habitat 和上面的 RoboTHOR 不是同一个环境。RoboTHOR 用 `spikingnav`。Habitat 需要单独的 Python，里面有 **habitat-sim 0.3.1** 和能 `import habitat` 的 **habitat-lab 0.3.x**。无头服务器不需要显示器。进程最多同时打开 4 张 CUDA 设备，所以 `CUDA_VISIBLE_DEVICES` 最多列 4 个编号；不设置时脚本默认只用 `0`。`--gpu` 是这组可见设备里的序号，只露出一张物理卡时写 `--gpu 0`。`--algo` 可选 `ppo`、`il`、`il-human`，默认 `ppo`。

换一台机器时，在仓库根目录先指定解释器和 habitat-lab 源码目录。当前这台服务器上这两个路径已经能自动找到，可以不写：

```bash
export HABITAT_PYTHON=/path/to/envs/habitat/bin/python
export HABITAT_LAB=/path/to/habitat-lab
"$HABITAT_PYTHON" -c "import habitat, habitat_sim; print(habitat_sim.__version__)"
```

数据放在仓库里的这些位置（脚本按仓库根目录找，不写绝对路径）：

```
datasets/objectnav/hm3d/v1/{train,val,val_mini}/   # 回合 json.gz
datasets/scene_datasets/hm3d/{train,val}/          # *.basis.glb
datasets/objectnav_hm3d_hd/objectnav_hm3d_hd/      # 仅 il-human
```

长时间训练放进 tmux，断线或合上笔记本不会停。看到日志后按 `Ctrl-b` 再按 `d` 离开；下次 `tmux attach -t <会话名>`。同一张卡、同一个输出目录上不要再开第二个训练进程。

### 本机离线专家轨迹实验

2026-10-15 已在项目内补齐 Habitat 环境。`scripts/habitat_env.sh` 优先使用
`storage/envs/habitat/bin/python` 与 `third_party/habitat-lab-v0.3.1/habitat-lab`，
并加载项目内的 EGL 库。该环境只读复用原 `spikingnav` 环境的 PyTorch；
Habitat 使用 NumPy 1.26.4、Gym 0.23.0、Habitat-Lab/Sim 0.3.1。
运行以下命令核对实际解释器，不必重新安装 RoboTHOR：

```bash
source scripts/habitat_env.sh
"$PYTHON" -B -c 'import sys,torch,habitat,habitat_sim; print(sys.executable, torch.__version__, habitat_sim.__version__, torch.cuda.is_available())'
```

新离线路线使用 `scripts/hm3d_experiment.sh`。每个输出目录必须不存在；
脚本保存源码快照、Git 基线、工作区差异、配置及输入哈希，不执行 commit/push。
以下是小规模检查示例，完整实测结果见
[本次实验记录](docs/experiments/2026-10-15-hm3d-offline-il.md)。

```bash
RUN="storage/hm3d_manual_$(date +%Y%m%d_%H%M%S_%N)"
bash scripts/hm3d_experiment.sh collect --output "$RUN/collect" \
  --scenes 1S7LAXRdDqK --per-category 1
bash scripts/hm3d_experiment.sh train data="$RUN/collect/data/train" \
  output="$RUN/overfit" fixed=true updates=250 batch_size=4
```

轨迹每步保存动作执行前的 RGB、Depth、action、gt_action、位姿、目标距离，
另存终止观测、类别、成功状态、SPL。RGB 为 224×224 JPEG，Depth 为 224×224
归一化 uint16；保留每一步，但不是传感器原分辨率无损存档。当前模型输入为 RGB
和目标类别，Depth 用于检查和后续扩展。只有经过完整性检查的专家成功回合进入训练；
失败尝试仍在 `episodes.jsonl` 中计数，拒收轨迹使用 `.rejected` 后缀，只有校验通过后才发布训练 `.npz`。

离线训练使用 SSE+SPN、16 步监督窗口、8 步前置上下文；上下文及 padding 不计入
监督步数。BatchNorm 仅用训练图像校准，之后固定统计量并继续学习仿射参数。
`resume=` 恢复模型、优化器和随机数状态，要求数据及主要协议相同；新建输出目录，
不保证恢复到完全相同的 epoch 中间采样位置。检查点的 steps 是累计监督次数，
不是唯一轨迹帧数。旧的均匀交叉熵检查点没有 `transition_c`，按 1 对待，不能直接续到当前 c=3 的训练。

2026-10-16 起，后续离线 IL 的参数写在 `configs/hm3d_offline_il.yaml`，由 Hydra 读入。
改文件即可，不必再改 argparse。单次覆盖仍可写在命令后面，例如 `epochs=3 batch_size=16`。
Hydra 不切换工作目录，结果只写到配置里的 `output`。`data` 和 `output` 每次实验都要填。
正数 `epochs` 按完整遍历训练集覆盖 `updates`。当前默认：batch size 32、每 100 次更新存一次检查点、
学习率 3e-4、动作切换权重 3、`frozen_bn`。已完成的 203,470 帧训练集上，一个 epoch 是
203,470 个监督步、453 次更新；换数据时以启动日志里的 `EPOCH` 为准。

训练损失改为论文里的行为克隆加权：专家动作相对上一步发生变化的帧权重为 `transition_c`（默认 3），
其余为 1，再除以本批有效步数。回合第一帧没有上一步，权重为 1；窗口从中间切开时，用窗口外的真实上一步比较。
优化的是这个加权损失。同一批的普通交叉熵另外记为 `bc_loss_unweighted`。留出动作打分仍用普通交叉熵，
方便和 10 月 15 日的曲线比较。`transition_c=1` 恢复均匀交叉熵。

填写 `student_eval_list` 后，每个新检查点在 GPU 上另起进程做学生评估，训练权重一直留在 GPU 上。
评估使用固定验证回合，报告 SR 和 SPL。同一输出目录里只保留 SR 最高的 `keep_sr_checkpoints` 份权重，
默认 2 份；SR 相同留 SPL 更高的，仍相同留更晚的。还没有评完、或评估失败的检查点先留着。
其他实验目录里的历史检查点不会删。不填 `student_eval_list` 时不启动这份评估。
在线 PPO / IL 的边训边评仍用下面的 `watch_habitat_eval.sh`，两套入口不要混用。

正式基线在该配置里填写独立轨迹清单 `data` 和 `validation_data`，
或在命令行写成 `data=train_files.json validation_data=validation_files.json`，不要加 `fixed=true`。自主评估必须单独运行：

```bash
bash scripts/hm3d_experiment.sh eval --output "$RUN/student_val" --split val \
  --checkpoint /项目内路径/ckpt_steps_N.pt --episode-list /项目内路径/eval_episodes.json
```

用 `source scripts/habitat_env.sh` 后运行
`"$PYTHON" -B scripts/select_hm3d_episodes.py --output "$RUN/eval_episodes.json" --count 200`
可固定覆盖 20 个 val 场景的 200 回合清单。原始标注的 episode_id 可能重复，必须使用
Habitat 加载后的唯一编号。在线 RGB 采用和保存轨迹一致的缩放及 JPEG 预处理。

清单为 `{ "content_scene": "场景ID", "episode_id": "回合ID" }` 的 JSON 数组，
应在看模型结果之前固定。动作准确率、专家 SR、学生自主 SR 分开报告；
少量回合的 SR/SPL 只用于诊断，不能替代 HM3D val 全集 2,000 回合的结果。
200 是本轮快速诊断选择的子集数量，2,000 是当前本地 val 标注实际包含的回合总数，
两者都不是场景数或动作步数。当前自动任务仅安排 200 回合评估，完整验证需另外运行。
本轮使用新增采集脚本的 `--checkpoint` 模式评估；success/SPL 沿用 Habitat 实现。
新增脚本用途与一次性安装代码清理记录见上述实验说明。

### 查看本轮离线 IL 曲线

2026-10-15 实验的 TensorBoard 服务使用端口 **6006**。在远程 VS Code 的“端口”面板
转发 6006，打开 `http://127.0.0.1:6006`。实际服务命令与 PID 保存在当前实验目录
`active_tensorboard.txt` 指向的监控目录内的 `launch.json`。

`hm3d_tensorboard_bridge.py` 只读取实验的 JSON 日志，补写 TensorBoard event 文件，
不修改模型、采集或训练参数；每 5 秒同步，未来 `scaled_baseline` 开始后会自动出现。

- `baseline` → `train/bc_loss`：首轮普通 IL，每次更新的训练 batch 损失，共 600 点。
- `overfit_bn_consistent` → `train/bc_loss`：固定小样本过拟合，共 250 点。
- `fixed_diagnostic/loss`：同一固定诊断数据在训练前后的损失，只有首尾测量值。
- `heldout_actions/loss`、`heldout_actions/recall_stop`：训练结束后的留出场景动作指标。
- `student_val/sr`、`student_val/spl`：已完成的学生自主评估结果，当前是末次评估点。
- `collection/accepted_frames`：扩量采集进度，不能当作学生训练成功率。

训练图横轴 Step 为**优化器更新次数**；采集图横轴为尝试回合数。历史 JSON 未记录
每个点的原始时间戳，event 的 Wall time 是导入时间，应使用 Step 比较训练过程。
验证指标尚无逐更新曲线，不做插值冒充中间评估。加权对照使用
`train/bc_loss_weighted`，不能与普通交叉熵数值直接比较。2026-10-16 之后、
`transition_c` 不是 1 的离线训练，优化损失记在 `train/bc_loss_transition`，
同一批普通交叉熵记在 `train/bc_loss_unweighted`。这两条也不能和 10 月 15 日的 `train/bc_loss` 直接比数值。

### 换到另一台服务器接续

2026-09-25 本机训练已全部停下。PPO 日志停在第 643072 步，最后一次落盘是 `ckpt_steps_614400.pt`。下面三份权重要拷到新机器，文件里只有 `model` 和 `steps`，没有优化器状态：

| 任务 | 拷贝这个文件 | 步数 |
| --- | --- | --- |
| PPO | `storage/habitat-objectnav-hm3d-spiking/ckpt_steps_614400.pt` | 614400 |
| 测地线 IL | `storage/habitat-objectnav-hm3d-il-geodesic/ckpt_steps_563200.pt` | 563200 |
| 人工示范 IL | `storage/habitat-objectnav-hm3d-il-human/ckpt_steps_691200.pt` | 691200 |

新机器上克隆仓库后设置 `HABITAT_PYTHON` 和 `HABITAT_LAB`。场景网格约 91GB，回合标注约 631MB，人工示范约 84MB。权重放回各自的 `storage/` 目录。多卡脚本最多 4 张卡，每张卡一个进程；`--num-envs` 是每张卡的环境数，`--total-steps` 是所有卡合计的环境步。不要再传 `--gpu`。卡号按新机器的空闲 GPU 改。

RL 用 DD-PPO 接续。两张卡时把 `--preemption-threshold` 设为 `0.5`，较快的卡采完一轮后，较慢的卡走完至少一半 rollout 就可以提前更新。日志里应出现 `DD-PPO world 2` 和 `resume from step 614400, optimizer starts fresh`。

```bash
git clone git@github.com:tju-dhj/SpikingVLN.git
cd SpikingVLN
export HABITAT_PYTHON=/path/to/envs/habitat/bin/python
export HABITAT_LAB=/path/to/habitat-lab

tmux new -s train-ppo
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_habitat_hm3d_dist.sh \
  --algo ppo --split train --num-envs 8 --preemption-threshold 0.5 \
  --resume storage/habitat-objectnav-hm3d-spiking/ckpt_steps_614400.pt
```

IL 用同一个多卡脚本接续，没有 rollout 抢占。每张卡各自采样，反传后平均梯度。`beta` 按检查点里的步数继续。测地线：

```bash
tmux new -s train-il
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_habitat_hm3d_dist.sh \
  --algo il --num-envs 4 --beta-start 1 --beta-end 1 \
  --output-dir storage/habitat-objectnav-hm3d-il-geodesic \
  --resume storage/habitat-objectnav-hm3d-il-geodesic/ckpt_steps_563200.pt
```

人工示范：

```bash
tmux new -s train-human
CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_habitat_hm3d_dist.sh \
  --algo il-human --num-envs 4 \
  --output-dir storage/habitat-objectnav-hm3d-il-human \
  --resume storage/habitat-objectnav-hm3d-il-human/ckpt_steps_691200.pt
```

学习率（PPO）和 `beta`（IL）按文件里的步数继续；Adam 的动量从这一步重新积累。这次接续之后新写出的检查点会带上优化器，再下次 `--resume` 会一起恢复。检查点仍由第一张卡写入原来的输出目录。端口占用时设置 `MASTER_PORT`。按 `Ctrl-b` 再按 `d` 离开会话。

### PPO

奖励与 RobustNav / AllenAct 相同，测地项按本步实际位移裁剪：

```
r_t = 10 * I_success + clip(d_{t-1} - d_t, ±本步位移) - 0.01
```

PPO 为 `clip=0.1`（价值损失用同一个 ε 裁剪），价值系数 0.5，熵系数 0.01，学习率 `3e-4` 按总步数线性衰减到 0，`γ=0.99`，GAE `λ=0.95`，rollout 128，每个 rollout 更新 4 轮，梯度裁剪 0.5。总步数默认 3 亿。

单卡：

```bash
# 先用一个验证场景走通
bash scripts/train_habitat_hm3d.sh \
  --algo ppo --split val --scenes 4ok3usBNeis --num-envs 1 \
  --rollout-steps 8 --total-steps 8

# HM3D v1 训练集，默认物理 GPU 0
bash scripts/train_habitat_hm3d.sh --algo ppo --split train --num-envs 8
```

DD-PPO 用 `scripts/train_habitat_hm3d_dist.sh`。`CUDA_VISIBLE_DEVICES` 里每张卡一个进程，各自采样、各自保留膜电位，反传后按样本数平均梯度。较快的卡采完一轮后，较慢的卡只要已经走完至少一半 rollout，就可以提前更新。`--num-envs` 是每张卡的环境数，`--total-steps` 是所有卡合计的环境步。最多列 4 张卡。两张卡时把 `--preemption-threshold` 设为 `0.5`；默认 `0.6` 要等两张卡都采完。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_habitat_hm3d_dist.sh \
  --algo ppo --split train --num-envs 8 --preemption-threshold 0.5
```

日志里应出现 `DD-PPO world 2`。检查点与 TensorBoard 在 `storage/habitat-objectnav-hm3d-spiking/`（事件在其中的 `tb/`），多卡时只由第一张卡写入。检查点默认每 50 次更新存一次。`--resume` 从已有 `ckpt_steps_*.pt` 继续：恢复网络权重和文件里的步数，学习率按这个步数在总步数上接着衰减。文件里若有优化器状态也会一起恢复；只有权重和步数的旧检查点会让优化器从头开始，之后新写出的检查点会带上优化器。日志出现 `resume from step <步数>` 即表示接上。`train/success`、`train/spl` 来自训练集 rollout 中刚好结束的回合。验证集评估用下面的检查点监控。Gibson 和 MP3D 的回合目录已经在 `datasets/objectnav` 下，场景网格齐了之后可以用同一套环境入口换数据路径。

从 `ckpt_steps_614400.pt` 做多卡 DD-PPO 接续：

```bash
tmux new -s train-ppo
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_habitat_hm3d_dist.sh \
  --algo ppo --split train --num-envs 8 --preemption-threshold 0.5 \
  --resume storage/habitat-objectnav-hm3d-spiking/ckpt_steps_614400.pt
```

单卡接续把脚本换成 `scripts/train_habitat_hm3d.sh`，并去掉 `--preemption-threshold`。换机器时的数据拷贝见上面的「换到另一台服务器接续」。

### 模仿学习

模仿学习用同一套 SSE + SPN。损失是专家动作上的交叉熵。`--algo il` 的专家是测地最短路，直接用 HM3D 场景。`--algo il-human` 的专家是人工遥控动作，回合目录是 `datasets/objectnav_hm3d_hd/objectnav_hm3d_hd`。还没有这批示范时：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download axel81/pirlnav \
  --repo-type dataset --include "objectnav_hm3d_hd/*" \
  --local-dir datasets/objectnav_hm3d_hd --max-workers 1
```

`--beta-start` 到 `--beta-end` 是执行专家动作的概率。下面两条都固定为 1，智能体始终跟着专家走。`il` 省略这两项时，会在 `--total-steps` 内从 1 线性降到 0。`il-human` 在脚本里固定为 1。

```bash
# 测地最短路
CUDA_VISIBLE_DEVICES=1 bash scripts/train_habitat_hm3d.sh \
  --algo il --gpu 0 --num-envs 4 \
  --beta-start 1 --beta-end 1 \
  --output-dir storage/habitat-objectnav-hm3d-il-geodesic

# 人工示范
CUDA_VISIBLE_DEVICES=2 bash scripts/train_habitat_hm3d.sh \
  --algo il-human --gpu 0 --num-envs 4 \
  --output-dir storage/habitat-objectnav-hm3d-il-human
```

省略 `--output-dir` 时，测地线写到 `storage/habitat-objectnav-hm3d-il-geodesic/`，人工示范写到 `storage/habitat-objectnav-hm3d-il-human/`。TensorBoard 在各自的 `tb/`，标量有 `train/bc_loss`、`train/agreement`、`train/beta`、`train/success`、`train/spl`。检查点默认每 50 次更新存一次。这里的 `train/success`、`train/spl` 同样来自训练回合；`--algo il` 的动作还按 `beta` 混入专家动作。`--resume` 从该步继续，`beta` 按恢复后的步数计算。验证集评估用下面的检查点监控。

多卡接续测地线 IL。`--num-envs 4` 是每张卡 4 个环境：

```bash
tmux new -s train-il
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_habitat_hm3d_dist.sh \
  --algo il --num-envs 4 --beta-start 1 --beta-end 1 \
  --output-dir storage/habitat-objectnav-hm3d-il-geodesic \
  --resume storage/habitat-objectnav-hm3d-il-geodesic/ckpt_steps_563200.pt
```

人工示范把 `--algo` 换成 `il-human`，`--output-dir` 和 `--resume` 换成 `storage/habitat-objectnav-hm3d-il-human/ckpt_steps_691200.pt`，会话名用 `train-human`。单卡接续改用 `scripts/train_habitat_hm3d.sh`，并加上 `--gpu 0`。人工示范数据用 Hugging Face 下载；镜像不可用时去掉 `HF_ENDPOINT` 那一项。

### 多卡模仿学习

PPO 的多卡方式见上一节。模仿学习用同一个脚本，没有 rollout 抢占。从已有权重继续时带上 `--resume`，输出目录仍用原来的目录，这样新检查点写在同一处：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_habitat_hm3d_dist.sh \
  --algo il --num-envs 4 --beta-start 1 --beta-end 1 \
  --output-dir storage/habitat-objectnav-hm3d-il-geodesic \
  --resume storage/habitat-objectnav-hm3d-il-geodesic/ckpt_steps_563200.pt
```

人工示范把 `--algo` 换成 `il-human`，目录和 `--resume` 换成 `storage/habitat-objectnav-hm3d-il-human/ckpt_steps_691200.pt`。默认主端口是 `29531`，占用时设置 `MASTER_PORT`。TensorBoard 和检查点由第一张卡写入 `--output-dir`。

### 边训练边评估

离线 IL 不走这个监视脚本。在 `configs/hm3d_offline_il.yaml` 里设置 `student_eval_list` 后，
训练进程会在每次保存检查点时于 GPU 上评估，并只在该次输出目录保留 SR 最高的两份权重。见上面 2026-10-16 一节。

`scripts/watch_habitat_eval.sh` 盯着在线 PPO / IL 的检查点目录。训练每写出一个写完的 `ckpt_steps_*.pt`，它就在验证集上用贪心动作跑一轮，把 SR 和 SPL 追加到该目录的 `eval_results.tsv`。评完后只保留 SR 最高的两个权重，其余已评测的 `ckpt_steps_*.pt` 会删掉。SR 相同时留下 SPL 更高的；仍相同则留下训练步数更大的。还没评完的文件不会删，`tb/` 和 `eval_results.tsv` 也不会删。

换一张训练没用的卡。`--gpu` 仍是可见设备里的序号：

```bash
# PPO
CUDA_VISIBLE_DEVICES=3 bash scripts/watch_habitat_eval.sh \
  --ckpt-dir storage/habitat-objectnav-hm3d-spiking \
  --split val --episodes 100 --num-envs 2 --gpu 0

# 测地线模仿学习
CUDA_VISIBLE_DEVICES=3 bash scripts/watch_habitat_eval.sh \
  --ckpt-dir storage/habitat-objectnav-hm3d-il-geodesic \
  --split val --episodes 100 --num-envs 2 --gpu 0
```

人工示范把目录换成 `storage/habitat-objectnav-hm3d-il-human`。`--keep` 默认是 2。`--episodes` 是每个检查点评多少个验证回合，默认 100。`--once` 只评当前目录里还没记录的检查点，然后退出。

### 查看曲线

在服务器上启动，并监听所有网卡。端口按机器上的空闲端口改：

```bash
tensorboard --logdir storage/habitat-objectnav-hm3d-spiking/tb --host 0.0.0.0 --port 6011
```

保持这个窗口。在自己电脑的浏览器打开 `http://<服务器IP>:6011`。服务器 IP 用 `hostname -I` 查看。要同时看 RoboTHOR 与几条 HM3D 曲线时，把 `--logdir` 换成 `storage`。只看两条模仿学习时：

```bash
tensorboard --logdir_spec=geodesic:storage/habitat-objectnav-hm3d-il-geodesic/tb,human:storage/habitat-objectnav-hm3d-il-human/tb \
  --host 0.0.0.0 --port 6020
```

浏览器打开 `http://<服务器IP>:6020`。若中间有防火墙，在本机做 SSH 转发：`ssh -L 6011:127.0.0.1:6011 <用户>@<服务器IP>`，然后打开 `http://127.0.0.1:6011`。这时 TensorBoard 也可以只绑 `127.0.0.1`。

## 论文对照

| 方法 | Params | FLOPs | PointNav SR/SPL | ObjectNav SR/SPL | 腐蚀均值 SR/SPL |
| --- | --- | --- | --- | --- | --- |
| ANNNav | 14.0M | 4.21G | 98.21 / 82.13 | 31.05 / 14.26 | 8.45 / 3.59 |
| SpikingNav | 12.1M | 0.97G | 96.54 / 72.93 | 34.12 / 16.20 | 13.71 / 5.67 |

完整训练很重，本仓库保证模型与管线可跑通，不自动跑满长训练。数字可能无法逐项对齐：S1 未公开、骨干用 SEW-LIF 而非完整 BuSNN、随机种子与并行数也可能不同。不加载预训练权重时，本地统计的参数量大约是 SpikingNav 12.99M、ANNNav 15.12M。

## 引用

```
@article{zhang2026spikingnav,
  title={SpikingNav: Robust Embodied Navigation with Spiking Neural Policies},
  author={Zhang, Jiahong and Shen, Sijun and Wu, Dehua and Lin, Yifan and Xia, Xuechen and Chu, Xu and Zhang, Youhui and Li, Guoqi},
  journal={arXiv preprint arXiv:2608.05078},
  year={2026}
}
```
