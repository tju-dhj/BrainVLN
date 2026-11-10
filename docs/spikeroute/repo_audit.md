# Repository audit (P0)

- Host: suzhou-vln (verified hostname: dss8440)
- Repository: /home/vln/dhj/SpikingVLN
- Branch: main, tracking origin/main
- Initial worktree: untracked data/ only; no tracked R2R changes were present.
- Present model path: spikingnav/models plus spikingnav/habitat_objectnav; current task is HM3D ObjectNav.
- Existing offline collector: scripts/collect_hm3d_expert.py. It stores HM3D ObjectNav RGB/depth trajectories and geodesic expert actions.
- Existing offline trainer: scripts/train_habitat_offline_il.py -> spikingnav.habitat_objectnav.offline_il.train_offline.
- R2R/VLN-CE files: no spikingnav/vlnce package, R2R collector, R2R trainer, R2R evaluator, or R2R config exists in this checkout.
- Dataset links: data/datasets -> /home/vln/VLN/datasets; data/scene_datasets -> /home/vln/VLN/scene_datasets. The repository itself contains no verified R2R-VLNCE v1-3-preprocessed rollout store.
- Runtime audit: non-interactive SSH did not expose python/conda; a usable Python/Habitat environment is not yet verified.

Action mapping currently documented for HM3D ObjectNav is stop, move_forward, turn_left, turn_right, look_up, look_down. This cannot be reused as the four-action R2R-CE mapping without a separate verification against the VLN-CE task config.
