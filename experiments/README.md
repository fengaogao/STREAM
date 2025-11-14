# STREAM MovieLens-10M Alignment Experiments

This directory contains reproducible scripts for the two requested analyses on the MovieLens-10M (ml-10M100K) corpus:

* **Experiment A (Alignment / Decoupling)** – measures how well the latent directions discovered by truncated SVD align with human genres. The script saves a direction-vs-genre cosine heatmap alongside quantitative diagnostics (diagonal energy ratio, maximum cross response, etc.).
* **Experiment B (Controllability / Causality)** – probes whether nudging logits along a genre-aligned direction boosts that genre while minimally perturbing others. It emits a dose–response curve (Δlogit and ΔNDCG) and a top-k recall improvement bar chart.

Additional tooling:

* **`category_direction_benefit.py`** – aggregates the effect of a single category-aligned direction across many evaluation samples, contrasting its impact on recall/probability metrics against random directions.

## Usage

```bash
# Ensure dependencies are available. Matplotlib/seaborn are required for plotting.
pip install -r requirements.txt

# Run both experiments (example uses the "Action" genre)
python -m experiments.genre_axis_experiments \
    --data_dir ml-10M100K \
    --output_dir experiments/outputs \
    --target_genre Action \
    --alphas 0 0.5 1.0 1.5 2.0 \
    --topk 5 10 20
```

The command will create `experiments/outputs/` with:

* `experiment_a_alignment.png` – cosine heatmap.
* `experiment_a_metrics.json` – summary statistics.
* `experiment_b_dose_response_<GENRE>.png` – Δlogit/ΔNDCG dose–response curves.
* `experiment_b_topk_gain_<GENRE>.png` – Top-k hit-rate improvements for the selected genre.
* `experiment_b_summary_<GENRE>.json` – machine-readable metrics for downstream analysis.

Pass a different `--target_genre` to inspect another category or override `--alphas` / `--topk` for alternative perturbation grids.

### Aggregated category-direction comparison

To quantify how a learnt direction reshapes ranking quality, run:

```bash
python -m experiments.category_direction_benefit \
    --data_dir /path/to/ml-10m \
    --artifacts_dir /path/to/ml-10m/bert \
    --category Drama \
    --num_samples 2048 \
    --top_k 10 \
    --alpha 0.8 \
    --random_trials 10
```

The script reports aggregated recall, target-probability, and category-mass metrics for the selected genre alongside a random-direction baseline, illustrating whether the learnt direction offers a measurable advantage.

## Implementation notes

* Latent directions are obtained via sparse truncated SVD on the user-item interaction matrix (one held-out interaction per user for evaluation).
* Genre assignment uses a Hungarian matching between SVD directions and genre indicator vectors to maximise on-diagonal cosine energy.
* Controllability probes adjust item logits by adding `α · d_genre` before ranking and compare Δlogit/ΔNDCG against baseline (`α = 0`).
* Top-k improvements are reported relative to the baseline hit rate for users whose held-out item belongs to the probed genre.

Ensure the MovieLens preprocessing files (`item_ids.json`, `item_text.json`, `finetune.jsonl`) are present under `ml-10M100K/` as provided in the repository.
