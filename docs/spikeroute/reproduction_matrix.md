# Reproduction matrix (P0)

| ID | Scope | Current status |
|---|---|---|
| HM3D ObjectNav IL | Existing repository path | Code exists; dataset/runtime not verified in this audit |
| E0 R2R-CE | GloVe + LSTM + SSE + SPN | BLOCKED: R2R code and rollout schema absent |
| E1-E6 | Spiking language, retrieval, event gate, fast/slow SPN, adaptive T | NOT STARTED; must follow E0 validation |

The first valid R2R experiment requires: R2R-VLNCE v1-3-preprocessed annotations, compatible Matterport3D scene assets, Habitat/VLN-CE runtime, a verified four-action mapping, and an offline trajectory collector. A collector should be run once per dataset/version and its output reused by E0-E6.
