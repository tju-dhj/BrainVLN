# P00: Data and Code Audit

Audit date: 2026-11-03

## Repository state and boundaries

- Repository: /home/vln/dhj/SpikingVLN, branch main; HEAD was not changed.
- Root AGENTS.md exists. It requires project-local outputs, no commits/pushes, unique experiment directories, and checks before long runs.
- User-owned paths were already untracked at audit start: configs/r2r_offline_il.yaml, data/, docs/experiments/2026-11-02-r2rce-offline-il.md, docs/spikeroute/, scripts/collect_r2r_gt.py, scripts/eval_r2r.py, scripts/train_r2r_offline_il.py, spikingnav/vlnce/, and tests/test_r2r_il.py. P00 leaves them untouched.
- CODEX_BRAIN_RULES.md was absent from the remote project at audit start; the provided kit contains it. AGENTS.md was not modified.
- storage/collect_data resolves inside this project. data/scene_datasets is a symlink to /home/vln/VLN/scene_datasets and is read-only input.

## Collected data

Observed tree: storage/collect_data/{logs,train}. No collected validation split exists. Train contains 61 scene directories, 10,819 episode directories, and 10,819 READY markers.

Inspected episode r1Q1Z4BcV1o/6913:

| Field | Observed value |
|---|---|
| metadata.steps and action count | 63 |
| Action names / IDs | 0 STOP, 1 MOVE_FORWARD, 2 TURN_LEFT, 3 TURN_RIGHT |
| STOP | exactly once, at action index 62 |
| RGB | 63 files, 0_0.png through 62_0.png; PNG, 224x224, 8-bit RGB |
| Depth | 63 files with matching names; PNG, 224x224, 16-bit grayscale |
| Depth scale | stored integer / 1000 = meters |
| Alignment declaration | observation rgb[t], depth[t] precedes action[t] |
| Positions | 63 pre-action positions, one per action |
| Instruction | one full instruction and 200 padded word-token IDs; 29 non-padding IDs |
| Subinstructions/FSA | absent in the inspected instruction record |
| Official GT | stored 63 actions exactly match v1.3 train_gt for episode 6913 |

This record is a contiguous teacher-forcing sequence: each pre-action observation has its expert label, and there is no extra terminal observation in the action-aligned frame folders. trajectory.positions is action-aligned; locations and reference_path are auxiliary paths and may have different lengths. Pose, goal coordinates, geodesic distance, and future paths are labels/analysis only and must not enter policy inference.

The inspected format contains one instruction per collected episode. FSA segmentation/subinstruction boundaries are not stored; the initial baseline must encode the full instruction unless a deterministic segmenter is separately specified.

## Existing code and evaluation

- ObjectNav code under spikingnav/habitat_objectnav has six actions and is incompatible with R2R-CE.
- Existing SNN building blocks are spikingnav/nn/lif.py, spikingnav/models/sse.py, and spikingnav/models/spn.py. SSE has an internal frame-encoding loop; SPN carries a membrane vector across navigation steps. A new reservoir must distinguish navigation step t from internal SNN step k and reset each environment at episode boundaries.
- A separate, pre-existing user-added R2R implementation is in spikingnav/vlnce/. Its RolloutDataset currently trains on RGB, actions, and tokens and omits stored depth. Its policy uses an instruction encoder plus SSE/SPN and has a separate Habitat closed-loop evaluator. P00 leaves it unchanged.
- scripts/eval_r2r.py and spikingnav/vlnce/sim.py provide simulator rollouts and success/SPL/NE code, but do not evaluate a new RGB-D BrainVLN policy.
- R2R v1.3 train GT, embeddings, val_seen/val_unseen annotations, and the inspected scene mesh are present. No collected validation split is present. Official validation episodes must remain evaluation-only.
- /home/vln/dhj/DPedPro-VLN/data/pre_trained_model/MLANet/mla_best_ppo.pth exists outside this project (432,031,797 bytes). No MLANet checkpoint was found in this project's data/ or storage/. Its presence does not establish compatibility with a new encoder.

## P01/P02 interface and blockers

- Dataset item: RGB [T,3,224,224], depth [T,1,224,224] converted by the declared scale, token IDs [L], action IDs [T], and valid mask [T]. Batched sequences use [T,B,...] with episode-boundary reset masks.
- Validate each episode's action_names; reject unknown or reordered action maps instead of truncating ObjectNav's six actions.
- Split by scene among the 61 train scenes for a development holdout. It is not official Val-Seen or Val-Unseen. Keep official Val-Unseen for closed-loop evaluation only.
- The audited records support offline IL. Full-corpus validity remains unverified until the audit runs with --max-episodes 0.
- No BrainVLN model, spikingnav/vln trainer, or B0-B5 configs existed before P00. Official CLIP ViT-B/32 and DD-PPO ResNet50 encoder weights were not found inside this project. Do not claim official MLANet reproduction or auto-download large weights.

## P00 commands

- pwd and git status --short --branch in the remote project: branch main, pre-existing untracked paths listed above.
- find storage/collect_data/train -mindepth 1 -maxdepth 1 -type d | wc -l: 61 scenes.
- find storage/collect_data/train -mindepth 2 -maxdepth 2 -type d | wc -l: 10,819 episode directories.
- find storage/collect_data/train -name READY | wc -l: 10,819 completed markers.
- file on the inspected RGB and depth files: 224x224 RGB8 and 224x224 grayscale16 PNG.
- gzip -cd train_gt.json.gz piped to jq for episode 6913: 63 official actions; first and last IDs match stored labels.
- Run bounded audit: python scripts/audit_collected_vln.py --data storage/collect_data/train --max-episodes 20.
- Run audit tests: python -m pytest -q tests/vln/test_audit.py.

## Remaining checks before P01

Run the bounded audit and inspect its JSON report. Before training, run the full read-only audit with --max-episodes 0 to establish whether every episode is usable and agrees with official train GT. Do not start long training yet.
