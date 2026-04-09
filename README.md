# MetaEP-RL — Reinforcement Learning for Metasurface Exceptional-Point Design

> Active research repo: **single- and multi-agent RL for designing
> metasurfaces with engineered exceptional points (EPs)**.
>
> This is the *EP-targeted* RL line, kept intentionally separate from the
> general metasurface inverse-design work (which lives in
> [`metasurface_inverse_design`](https://github.com/starpacker/metasurface_inverse_design)
> together with its `approaches/ppo_rl/` archive). The main repo's PPO entry
> targets generic polarization control; this repo specialises on **EP
> physics** and is iterated on its own.

---

## What it does

Trains an RL agent (PPO and/or SAC) to **edit a pixel-style metasurface
geometry** so that its scattering response exhibits an **exceptional
point** at a target wavelength. The reward is built from a Lumerical
FDTD simulation of the structure (`data/jones_model_origin.fsp`), with
an optional **learned world model / judge** in the loop so the
expensive FDTD calls can be partially replaced by a fast surrogate.

The repo supports both:
- **Single-agent RL** — one agent edits the whole pattern.
- **Multi-agent RL (MAPPO)** — several agents control disjoint parts of
  the pattern in parallel.

Switching between the two modes touches a few coordinated places (see
the running notes in `PPO/yjh.md` and `PPO/ppo.md`).

## Repository layout

```
MetaEP-RL/
├── PPO/
│   ├── train/                ← PPO / MAPPO training entry points
│   ├── envs/                 ← gym-style envs wrapping the metasurface FDTD task
│   │   ├── env_core.py           ← multi-agent (MAPPO) version
│   │   ├── env_core_revised.py   ← single-agent (PPO) version
│   │   └── env_continuous.py / env_discrete.py
│   ├── algorithms/           ← PPO / MAPPO actor-critic, ACTLayer, etc.
│   ├── runner/               ← rollout workers
│   ├── judge/                ← learned world-model / reward predictor (the "judge")
│   ├── utils/                ← incl. draw.py for visualising EP structures
│   ├── run_rbr.py            ← rollout-based runner script
│   ├── run_mcts_final.py     ← MCTS-augmented run script
│   ├── config.py             ← all tunable parameters live here
│   ├── ppo.md / yjh.md       ← running developer notes (mode-switching cheatsheet, etc.)
│   ├── br.txt / mcts.txt     ← experiment notes
│   └── ...
├── SAC/
│   ├── sac_ae.py             ← SAC + autoencoder agent (pixel observations)
│   ├── train.py
│   ├── encoder.py / decoder.py ← image encoder/decoder used by SAC-AE
│   ├── envs/
│   ├── judge/
│   ├── logger.py / video.py
│   ├── config.py / args.json
│   ├── conda_env.yml         ← reproducible conda environment for SAC-AE
│   └── *.md                  ← per-file notes (sac_ae.md, train.md, utils.md, …)
└── data/
    ├── jones_model_origin.fsp ← Lumerical FDTD project (the simulator)
    ├── cal_farfield_data.lsf  ← Lumerical script for far-field calculation
    └── script.lsf             ← driver script
```

## Algorithms

| Folder | Algorithm | Notes |
|---|---|---|
| `PPO/` | **PPO** (single-agent) and **MAPPO** (multi-agent) | The two share the actor-critic / ACTLayer code; mode is selected by toggling `env_core` ↔ `env_core_revised` plus a couple of `action_dim`/`num_agents` knobs |
| `PPO/` (run_mcts_final.py) | **PPO + MCTS** | Tree-search-augmented rollout |
| `SAC/` | **SAC with image autoencoder** (pyTorch SAC-AE) | Uses an encoder/decoder pair so the agent can act on pixel-level observations of the structure |

## Running (PPO)

```bash
cd PPO
python train/train.py
```

This produces:

| File | Contents |
|---|---|
| `CD_history.txt`     | per-episode CD (chromatic dispersion / objective) trace |
| `reward_history.txt` | per-episode reward trace |
| `best_result.txt`    | best structure / metric found so far |
| `record.txt`         | history of how the best result evolved |

All knobs (learning rate, num_agents, discrete vs continuous action space,
PPO vs MAPPO, judge on/off, single- vs multi-wavelength EP) live in
`PPO/config.py` and a small set of files documented in `PPO/ppo.md`.

## Status

**Active** — this is the current EP-RL research line. The notes files
(`PPO/yjh.md`, `PPO/ppo.md`, `SAC/yjh.md`, …) are intentionally kept
in-repo as a running developer log.

## Related

- [`metasurface_inverse_design`](https://github.com/starpacker/metasurface_inverse_design)
  — the general metasurface inverse-design line. Its
  `approaches/ppo_rl/` subfolder is the *original* PPO-for-polarization
  experiment that this repo branched off from.
