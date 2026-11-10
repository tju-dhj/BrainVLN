# Known issues and blockers (P0)

1. The current checkout targets HM3D ObjectNav, while the execution plan targets R2R-CE VLN. The two trajectory formats and action spaces are different.
2. No R2R offline rollout directory was found. FSM/fast-slow IL cannot train until compatible trajectories exist.
3. Python and conda were not available on the non-interactive PATH during the audit; runtime discovery is still required.
4. No long training or GPU collection was launched. Login-node execution is intentionally avoided.
5. Existing HM3D collector may be reused only for HM3D ObjectNav experiments; it is not a substitute for R2R-CE collection.
