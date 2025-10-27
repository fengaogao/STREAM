# STREAM Offline Alignment & Controllability Experiments

This folder provides a runnable script that reproduces the two complementary visual studies requested for the MovieLens-10M (``ml-10M100K``) corpus after running the STREAM offline training pipeline.

## Experiments

* **Experiment A – Alignment / Decoupling**
  * Question: do the learned subspace directions align with the annotated content categories, and are the directions mutually decoupled?
  * Output: a direction-vs-category cosine heatmap together with quantitative diagnostics (diagonal energy share, mean signed/absolute diagonal response, maximum & mean cross-response).

* **Experiment B – Controllability / Causality**
  * Question: if we nudge the STREAM state along a category axis, do items of that category receive higher scores/ranks while minimally disturbing others?
  * Output: (1) dose–response curves showing Δlogit / ΔNDCG for the target category vs. others; (2) a top-k recall uplift bar chart for the target category.

## Prerequisites

1. Run the offline training script (``stream/train_offline.py``) to produce a checkpoint directory containing ``config.json``, ``subspace/subspace_U.pt``, ``item_head.pt``, and the saved model weights.
2. Ensure ``ml-10M100K`` contains ``item_ids.json``, ``item_text.json``, and ``original.jsonl`` (supplied in this repository).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m experiments.offline_vis.category_axis_probes \
    --data_dir ml-10M100K \
    --artifacts_dir ml-10M100K/bert \
    --output_dir experiments/offline_outputs \
    --model_type bert \
    --target_category "Action" \
    --alphas 0.0 0.5 1.0 1.5 2.0 \
    --topk 5 10 20
```

The command will create ``experiments/offline_outputs`` with:
* ``experiment_a_heatmap.png`` – direction vs. category cosine map.
* ``experiment_a_metrics.json`` – alignment metrics (diagonal share, signed/absolute diagonal strength, max/mean cross-response, coverage counts).
* ``experiment_b_dose_response_Action.png`` – Δlogit / ΔNDCG dose–response curves.
* ``experiment_b_topk_gain_Action.png`` – Top-k recall uplift bars for the Action axis.
* ``experiment_b_summary_Action.json`` – machine-readable statistics for Experiment B.

Set ``--model_type causal`` to analyse a causal LM checkpoint (when available) and switch ``--target_category`` / ``--alphas`` / ``--topk`` as desired.

## Reproducibility hints

* Use ``--alignment_batches`` or ``--evaluation_batches`` to limit the number of batches processed when iterating quickly. Omit them for a full pass over ``original.jsonl``.
* The script respects ``--seed`` for deterministic dataloader shuffling where applicable.
