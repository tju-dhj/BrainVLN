# HM3D ObjectNav 接续实验审计（2026-10-11，北京时间）

## 目标和版本

目标是在 HM3D ObjectNav 的未见场景验证集上，使 SSE + SPN 学生策略达到约 34.1% SR，同时报告 SPL。精确的“差不多”容差尚未指定，因此保留 34.1% 作为目标值，报告实际结果，不自行降低验收门槛。

本地 `SpikingNav.pdf` 第 6 页表 II 的 34.12% SR / 16.20% SPL 来自 **RoboTHOR**。迁移到 HM3D 是新基准上的实验，不是相同任务条件下的严格复现。本文没有重新验证官方代码是否公开；仓库按无官方代码的前提独立实现。

当前分支 `feature/my-experiments`，基础提交 `1f70f46a66ae9064347e6a3022234929c028ad14`。进入本次任务时已有未跟踪 `docs/`，保留原有记录。未 commit、push 或 stage。新约束写入根目录 `AGENTS.md`；本次新增预检查工具，不修改训练算法、历史日志或旧权重。

## 项目理解

- `spikingnav/models/sse.py`：SEW-ResNet18 风格脉冲视觉骨干，内部 4 个时间步、特征压缩、类别目标嵌入和融合。
- `spikingnav/models/spn.py`：循环膜电位状态、泄漏、脉冲复位，actor/critic 输出。仓库额外做膜电位裁剪。
- `spikingnav/habitat_objectnav/`：HM3D 的 PPO、测地线 IL/DAgger、人工示范 IL、离线示范训练和分布式支持。
- 当前策略输入是 RGB + 目标类别，尚不是自然语言 VLN。原 RoboTHOR/AllenAct 路径仍存在；当前实验优先 HM3D。
- ANN ImageNet 初始化、部分推断超参数和 GPU 稠密实现都需要在研究报告中注明，不能据此宣称复现论文芯片能耗。

## 已执行检查及阻塞

已运行 `scripts/preflight_hm3d.py`，退出码 **2**，表示前置条件不满足，而非训练完成。初次报告：`storage/hm3d_preflight_20261011_140644_996663/manifest.json`。报告保存分支/HEAD、源码快照、SHA-256、标注及检查点哈希、GPU、磁盘和历史结果。

用户确认的数据位置是本项目 `datasets/`。该目录约 1.5 GB，有回合标注，但递归检查没有真实 `.glb` 或 `.navmesh` 文件：

- `datasets/scene_datasets/hm3d/train` → `../../../../../daihaojie/Falcon-STG/data/scene_datasets/hm3d/train`，失效。
- `datasets/scene_datasets/hm3d/val` → `../../../../../daihaojie/Falcon-STG/data/scene_datasets/hm3d/val`，失效。
- 对 HM3D v1 标注逐场景核验：train 缺 **80/80**，val 缺 **20/20**，val_mini 引用的 **2/2** 也缺失。
- `/root/miniconda3/envs/spikingnav/bin/python` 缺少 torch、torchvision、numpy、habitat、habitat_sim、tensorboard。原 `habitat_env.sh` 的旧服务器回退路径也不存在。
- 本机检测到 RTX 5090 32 GB。GPU 枚举成功不代表 PyTorch CUDA 和 Habitat EGL 已可用。
- 检查时项目磁盘约剩 47 GB；README 记录原场景数据约 91 GB（不是本机实测大小）。完整迁移前需确认容量，或先只迁移指定诊断场景。
- 旧 `storage/il25k_20261011_125951/state.json` 的 `running` 来自旧服务器，本机没有相应训练进程。保留历史文件原样。

实际调用 128 步 IL smoke 启动脚本后，环境检查返回退出码 **1**：`Set HABITAT_LAB to a habitat-lab 0.3.x checkout.`。未进入训练、没有生成新权重。命令与输出保存在初次审计目录的 `smoke_startup.json` / `smoke_startup.log`。

## 历史结果的含义

| 路线 | 历史评估表最高 SR | 回合数 |
| --- | ---: | ---: |
| 测地线 IL | 2% | 100 |
| 人工示范 IL | 0% | 100 |
| PPO | 0% | 100 |

这些是旧 TSV 的统计，未在本机重测；TSV 没记录 split、回合 ID、动作选择方式或完整配置，不能视为经本次确认的完整验证集成绩。旧 128 步 IL smoke 的成功回合由专家带领，不能证明学生自主导航成功。

## 正式续训前需验证的实现问题

以下来自静态阅读，尚未通过运行定位为低 SR 的根因：

1. PPO `collect()` 没保存 rollout 起始膜电位，`update()` 却从零状态重算；跨 rollout 未结束回合时两者状态不一致。
2. PPO bootstrap 调用 `_policy_step()` 会更新 `self.memory`，下一轮可能再次处理同一观察。
3. CLI seed 传到环境，但没有统一设置 Python/NumPy/PyTorch 随机种子；训练/评估的 BatchNorm 状态差异需要测量。
4. 现有评估未保存逐回合 ID，也未显式避免重复回合；多环境下可能偏向较早结束的回合。应固定去重的回合清单，并记录逐回合 SR/SPL、动作和结束原因。
5. `watch_habitat_eval.py` 的目录监控模式会删除排名靠后的检查点；下一阶段使用单检查点评估入口并单独保存结果，保留全部原权重。

这些修复需配套可运行的回归检查，再进入长训练，避免混淆算法变更与单纯续训。

## 接续实验顺序

1. **E0 环境与专家检查**：真实 train/val 网格到位后，在项目内建立 Habitat 环境；验证 CUDA 张量计算、EGL 渲染、目标类别和动作映射。先用 train 场景 `1S7LAXRdDqK`、1 环境、rollout 16、128 步 IL 检查。旧机器 rollout 64/128 曾失败，不能照搬到新 GPU 直接长跑。
2. **E1 基线重测**：固定种子 12345 和独立 val 的 100 个去重回合，测已有 PPO/测地线 IL/人工 IL 权重，记录每类指标、停止动作比例、碰撞和超时。100 回合只用于诊断。
3. **E2 单场景学习检查**：对 train 场景做约 25,600 步 IL，beta=1，比较学生自主运行的训练场景结果。该结果不参与目标验收。
4. **E3 多场景 IL/DAgger**：E2 确认能学会后扩展 train 场景；以训练数据提供监督，定期评估固定 val 诊断子集。人工示范路线先核对 replay 与动作映射，不盲续已有 0% 权重。
5. **E4 PPO 微调与正式评估**：状态一致性问题修复并验证后，从较好 IL 权重初始化；在完整去重 val 上报告 SR/SPL、回合数、类别分解和置信区间，条件允许再做多随机种子。尚不能承诺达到 34.1%。

每阶段记录唯一实验 ID、源码快照/哈希、启动命令、依赖、数据和 checkpoint 哈希、seed、指标及退出状态。新运行不得覆盖旧目录。100 回合的 34% 仅表示约 34 个成功，不是精确达到 34.1% 的证据。

## 重跑预检查

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -B scripts/preflight_hm3d.py \
  --python /path/to/project-local-habitat/bin/python \
  --habitat-lab /path/to/project-local-habitat-lab
```

此工具只使用标准库，每次创建新 `storage/hm3d_preflight_*`；不会安装环境、训练或提交代码。发现模块仅表示可定位，仍需 E0 验证动态库和 GPU 兼容性。
