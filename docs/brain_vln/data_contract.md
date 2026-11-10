# P01 数据契约

采集目录 `storage/collect_data/train/` 只读。加载代码在 `spikingnav/brain_vln/data/`。本阶段没有启动训练。

2026-11-03 用项目环境 `storage/brain_vln/envs/BrainVLN` 对全部 10819 条带 `READY` 的 episode 做了 header 检查：动作表、步数、帧文件名集合、深度单位、指令 token、以及 `trajectory.json` 里 `positions` 的长度。10819 条全部通过。步数最少 21，最多 183，平均 58.3。像素解码抽查了 6 条，见下文。header 检查确认帧文件名齐全，不代替全库逐像素解码。

## 文件与时间

一条 episode 的目录是 `storage/collect_data/train/<scene>/<episode_id>/`。目录名必须等于 `metadata.json` 里的 `scene` 和 `episode_id`。源 split 字段是 `train`。

帧地址是 `rgb/{step}_0.png` 和 `depth/{step}_0.png`，`step` 从 0 到 `steps - 1`。时间由这个整数决定。目录列表的字典序不参与对齐。抽查里字典序的第 3 个 RGB 文件名是 `11_0.png` 或 `101_0.png`，加载器取的是 `2_0.png`，两者像素不同。

`rgb[t]` 和 `depth[t]` 是执行 `action[t]` 之前的观测。帧数等于动作数，没有额外终止帧。`positions` 同样是每个动作一条的动作前位姿；加载器只核对其长度，不把坐标放进样本。

## 动作、深度、指令

动作名必须严格等于：

| id | name |
| --- | --- |
| 0 | STOP |
| 1 | MOVE_FORWARD |
| 2 | TURN_LEFT |
| 3 | TURN_RIGHT |

名字重排、ObjectNav 的六动作表、0–3 以外的 id，都会拒绝。全库每条 episode 恰好有一个 STOP，并且位于最后一帧。

RGB 是 224×224 的 8-bit PNG，进入模型前变成 `float32`，除以 255，形状 `[T, 3, H, W]`。深度是 16-bit PNG。米 = 原始整数 / `depth_scale`，`depth_unit` 必须是 `meter`。原始值 0 是无效像素：`depth_valid` 为假，深度值保持 0。抽查的 6 条都声明 `depth_scale = 1000`。

指令使用整句 `instruction_tokens`，并要求与 `tokens.npy` 一致。0 只允许出现在尾部 padding。`token_mask` 在非 0 处为真。采集数据没有子指令或 FSA 字段；样本里的 `subinstruction_spans` 固定是 `None`。文件里若出现这类字段，加载器直接拒绝，不自行分段。

目标坐标、测地距离、起点位姿、`positions`、`locations`、`reference_path` 都不进入 batch。

## 样本与 batch

`BrainVLNDataset` 的 `mode="window"` 把每个窗口补到 64 步。`mode="episode"` 返回整段，由 `collate_fn` 补到 batch 内最长长度。补齐动作的值是 `-1`，不能把它读成 STOP。

batch 按时间在前、batch 在后排列，图像是 `[T, B, ...]`，token 是 `[B, L]`。字段顺序是：

`rgb`，`depth`，`depth_valid`，`tokens`，`token_mask`，`actions`，`valid_mask`，`loss_mask`，`burn_in_mask`，`episode_mask`，`reset_mask`，`detach_mask`，`episode_id`，`scene`，`instruction_text`，`window_start`，`episode_steps`。

模型输入是前五个键。`actions` 是标签。后面的 mask 交给以后的 Reservoir 和损失。`episode_id` 与 `scene` 只做索引。

一条样本只含一个 episode 片段，所以 `episode_mask` 等于 `valid_mask`。`reset_mask` 只在该样本的局部时间 0 为真：每个窗口、每个 batch 列都从清空状态开始，不把上一个样本的 Reservoir 状态继承过来。`detach_mask` 在局部时间 32、64、96……且该步仍然有效时为真，供截断 BPTT 在步前断开梯度。

## 窗口

默认 `window_size=64`，`burn_in=16`，`truncate_every=32`。约束是 `2 * burn_in <= window_size`，这样重放的重叠段不会把已经监督过的动作再监督一次。

长度不超过 64 的 episode 整段作为一个窗口，burn-in 为 0，再补齐。它从清空状态开始，前面的动作仍计算损失。

更长的 episode 用重叠窗口铺满全部动作，每个动作只在一个窗口里进入损失：

- 第一个窗口覆盖 `[0, 64)`，burn-in 为 0。
- 下一窗口从 48 开始，前 16 步是 burn-in，只用来重放状态，损失从局部时间 16 开始。
- 100 步的 episode 得到两个窗口：`(start=0, valid=64, burn=0)` 和 `(start=48, valid=52, burn=16)`。监督步依次是 0–63 和 64–99。

## 场景划分

61 个场景目录名按字典序排列。最后 6 个是内部 holdout，前 55 个是开发训练集。这个顺序不使用随机种子，`SPLIT_SEED` 是 `None`。

Holdout：`s8pcmisQ38h`，`sKLMLpTHeUy`，`sT4fr6TAbpF`，`uNb9QFRL6hY`，`ur6pFq6Qu1A`，`vyrNrziPKCB`。

这是 collected train 内部的开发划分，不是官方 Val-Seen 或 Val-Unseen。官方验证集不参与这一阶段。

## 六条抽查

选取方式是把排好序的 61 个场景等距取 6 个，每个场景取目录名排序后的第一条 `READY` episode。最后一条落在 holdout 场景里，只用于抽查，不会因此进入训练索引。

| scene | episode | steps | 非 padding token | 无效深度像素 | 深度范围（米） | 字典序第 3 个 RGB 名 |
| --- | --- | --- | --- | --- | --- | --- |
| 17DRP5sb8fy | 10154 | 50 | 28/200 | 208 | 0–9.183 | 11_0.png |
| D7G3Y4RVNrH | 10043 | 25 | 13/200 | 6456 | 0–6.196 | 11_0.png |
| S9hNv5qa7GM | 10121 | 63 | 26/200 | 58 | 0–9.718 | 11_0.png |
| ZMojNkEp431 | 10085 | 54 | 16/200 | 2019 | 0–23.347 | 11_0.png |
| mJXqzFtmKg4 | 1000 | 46 | 31/200 | 462 | 0–12.552 | 11_0.png |
| vyrNrziPKCB | 10076 | 112 | 36/200 | 681911 | 0–31.542 | 101_0.png |

六条的最后动作都是 STOP，RGB 为 `[T, 3, 224, 224]`，深度为 `[T, 1, 224, 224]`。局部第 2 帧与文件 `2_0.png` 的像素一致，并与上表字典序文件不同。

## 检查命令

```bash
cd ~/dhj/SpikingVLN
PYTHONPATH=. storage/brain_vln/envs/BrainVLN/bin/python -m pytest -q tests/brain_vln/test_p01_data.py
PYTHONPATH=. storage/brain_vln/envs/BrainVLN/bin/python -m compileall spikingnav/brain_vln
```

2026-11-03 的结果是 24 项测试通过，`spikingnav/brain_vln` 编译通过。当时这个环境里没有 pytest，已安装 pytest 8.4.2 到上述项目环境。
