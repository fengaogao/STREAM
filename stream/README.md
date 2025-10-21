# STREAM: State-space Temporal REcommendation Adaptive Memory

This repository contains a reference implementation of **STREAM** on top of two
LLM-based recommenders trained on the MovieLens-10M dataset. STREAM augments a
frozen base model with a low-rank adaptive state that reacts to distribution
shifts while respecting a KL trust region.

## Quickstart

```bash
pip install -r requirements.txt

# Offline training / setup
python -m stream.train_offline \
    --data_dir ml-10M100K \
    --out_dir runs/stream_ml10m \
    --model_type causal \
    --pretrained_name_or_path gpt2 \
    --rank_r 32 --router_k 16

# Online adaptation loop
python -m stream.run_online \
    --data_dir ml-10M100K \
    --ckpt_dir runs/stream_ml10m \
    --model_type causal \
    --epsilon_kl 0.01 --window_m 500
```

Switch `--model_type causal` to `--model_type bert` to exercise the BERT-style
recommender. The commands assume the MovieLens-10M preprocessing pipeline has
already produced the expected JSONL files inside `ml-10M100K/`.

## Repository layout

```
stream/
  config.py              # Configuration knobs
  dataio.py              # Dataset loading & tokenisation
  utils.py               # Shared helpers
  metrics.py             # Offline evaluation metrics
  subspace.py            # Subspace extraction via gradient covariance or PCA
  state_adapter.py       # STREAM state overlay (RegionStateBank + item head)
  detectors.py           # Drift detection (GLR + conformal band)
  trust_region.py        # KL-constrained solver for state updates
  consolidate.py         # Distillation and occasional LoRA merge utilities
  models/
    causal_lm_stream.py  # Causal LM integration
    bert_stream.py       # BERT-style integration
  train_offline.py       # Offline training entry point
  run_online.py          # Online adaptation loop
  distill_and_merge.py   # Periodic consolidation utilities
  tests/test_shapes.py   # Basic smoke tests
```

## Workflow overview

1. **Offline training** warms up the base recommender, extracts the STREAM
   subspace using gradient covariance, and initialises the item head `W`.
2. **Online adaptation** routes requests to regions, monitors for drifts, and
   performs trust-region constrained state updates with rollback safety.
3. **Consolidation** periodically distils accumulated state into the shared
   item head and optionally applies controlled LoRA merges.

### Config knobs

Important configuration options exposed in `config.py` include the subspace
rank `rank_r`, KL budget `epsilon_kl`, trust-region ridge `lambda_l2`, detector
window `window_m`, GLR threshold `glr_threshold`, and softmax temperature.

### Distillation and merge

`distill_and_merge.py` provides utilities to aggregate state deltas into the
shared `W` matrix at a configurable cadence and sketches a safe procedure for a
rare LoRA merge. The script operates on saved run directories produced by
`train_offline.py` and used in `run_online.py`.

## Testing

Minimal smoke tests are provided under `tests/test_shapes.py` and can be run
with `pytest`.

```bash
pytest stream/tests/test_shapes.py
```

These tests validate tensor shapes and orthonormality properties of the key
STREAM components.
