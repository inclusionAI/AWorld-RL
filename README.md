# T3RL

T3RL is a BFCL-focused multi-turn tool-use RL integration project built on top of `slime`.

## Status

- Current active path: BFCL
- Current `slime` version target: **v0.2.2**
  - Release: https://github.com/THUDM/slime/releases/tag/v0.2.2

## Environment Prerequisite (Important)

Before installing this project, set up the runtime environment following official `slime` guidance.

Because dependencies such as Megatron are complex, it is strongly recommended to use the official `slime` Dockerfiles for the matching version:
- https://github.com/THUDM/slime/tree/main/docker

## Installation

1. Initialize submodules:

```bash
git submodule update --init --recursive
```

2. Install T3RL dependencies:

```bash
pip install -e .
```

`pip install -e .` installs `t3rl` and `bfcl_eval`; it does **not** install `slime`.

3. Install `slime` separately:

```bash
pip install -e 3rdparty/slime --no-deps
```

## Data Preprocess

```bash
python data/preprocess_bfcl_data.py --output_dir data/processed/bfcl
```

## Train

```bash
bash scripts/train/run_bfcl_qwen3_4b.sh
```
