"""Genre-direction alignment and controllability experiments for MovieLens-10M."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_num_items(item_ids_path: Path) -> int:
    """Read the number of items from the metadata mapping."""

    with item_ids_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    mapping = meta.get("mid2idx", {})
    if not mapping:
        raise ValueError("item_ids.json must define a 'mid2idx' mapping")
    return max(int(idx) for idx in mapping.values()) + 1


def load_item_genres(item_text_path: Path) -> Tuple[Dict[int, List[str]], List[str]]:
    """Parse the item text metadata and extract genre annotations."""

    with item_text_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    item_to_genres: Dict[int, List[str]] = {}
    genre_set: set[str] = set()
    for idx_str, text in raw.items():
        idx = int(idx_str)
        if "Genres:" not in text:
            continue
        genre_chunk = text.split("Genres:")[-1]
        genres = [g.strip() for g in genre_chunk.split(",")]
        genres = [g for g in genres if g and g != "(no genres listed)"]
        item_to_genres[idx] = genres
        genre_set.update(genres)
    ordered_genres = sorted(genre_set)
    return item_to_genres, ordered_genres


def load_user_sequences(finetune_path: Path) -> Dict[int, List[int]]:
    """Load user interaction sequences from the finetune split."""

    user_to_items: Dict[int, List[int]] = {}
    with finetune_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user = int(record["user"])
            items: List[int] = record.get("items", [])
            if not items:
                continue
            user_to_items.setdefault(user, []).extend(items)
    return user_to_items


def build_user_item_matrix(
    user_to_items: Dict[int, Sequence[int]],
    num_items: int,
    holdout: int = 1,
) -> Tuple[coo_matrix, np.ndarray]:
    """Construct a sparse user-item matrix and hold-out targets."""

    user_ids = sorted(user_to_items.keys())
    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    test_items = np.full(len(user_ids), -1, dtype=np.int32)

    for uid, items in user_to_items.items():
        if len(items) <= holdout:
            continue
        idx = user_index[uid]
        history = items[:-holdout]
        target = items[-holdout]
        test_items[idx] = target
        for item in history:
            rows.append(idx)
            cols.append(item)
            data.append(1.0)

    matrix = coo_matrix((data, (rows, cols)), shape=(len(user_ids), num_items), dtype=np.float32)
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix, test_items


def compute_latent_factors(
    matrix: coo_matrix,
    rank: int,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run truncated SVD to obtain user and item latent factors."""

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Empty interaction matrix")
    # Use CSR for efficient svds
    csr = matrix.tocsr()
    u, s, vt = svds(csr, k=rank, which="LM", return_singular_vectors=True, random_state=random_state)
    # Sort components by descending singular value
    order = np.argsort(-s)
    s = s[order]
    u = u[:, order]
    vt = vt[order]
    user_factors = u * np.sqrt(s)
    item_factors = (vt.T * np.sqrt(s)).astype(np.float32)
    return user_factors.astype(np.float32), item_factors, vt.astype(np.float32)


# ---------------------------------------------------------------------------
# Experiment A: Alignment heatmap
# ---------------------------------------------------------------------------

def normalise_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def align_directions_with_genres(
    vt: np.ndarray,
    genre_to_items: Dict[str, np.ndarray],
    ordered_genres: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align SVD directions with genres using maximum matching."""

    directions = normalise_rows(vt.copy())
    genre_vectors = []
    for genre in ordered_genres:
        indicator = np.zeros(vt.shape[1], dtype=np.float32)
        indicator[genre_to_items[genre]] = 1.0
        genre_vectors.append(indicator)
    genre_matrix = normalise_rows(np.stack(genre_vectors, axis=0))
    cos_matrix = directions @ genre_matrix.T
    cost = -np.abs(cos_matrix)
    row_ind, col_ind = linear_sum_assignment(cost)
    # Reorder according to assignment
    ordering = np.argsort(row_ind)
    matched_rows = row_ind[ordering]
    matched_cols = col_ind[ordering]
    aligned_directions = []
    aligned_genres: List[str] = []
    aligned_cos_rows = []
    for r, c in zip(matched_rows, matched_cols):
        vec = directions[r]
        if cos_matrix[r, c] < 0:
            vec = -vec
            cos_matrix[r] = -cos_matrix[r]
        aligned_directions.append(vec)
        aligned_genres.append(ordered_genres[c])
        aligned_cos_rows.append(cos_matrix[r])
    aligned_direction_matrix = np.stack(aligned_directions, axis=0)
    aligned_cos = np.stack(aligned_cos_rows, axis=0)
    return aligned_direction_matrix, aligned_cos, aligned_genres


def compute_alignment_metrics(cos_matrix: np.ndarray) -> Dict[str, float]:
    diag = np.diag(cos_matrix)
    energy_total = float(np.sum(cos_matrix**2))
    energy_diag = float(np.sum(diag**2))
    diag_ratio = energy_diag / energy_total if energy_total > 0 else float("nan")
    off_diag = cos_matrix.copy()
    np.fill_diagonal(off_diag, 0.0)
    max_cross = float(np.max(np.abs(off_diag)))
    mean_cross = float(np.mean(np.abs(off_diag)))
    return {
        "diag_mean": float(np.mean(diag)),
        "diag_min": float(np.min(diag)),
        "diag_ratio": diag_ratio,
        "max_cross": max_cross,
        "mean_cross": mean_cross,
    }


def plot_alignment_heatmap(
    cos_matrix: np.ndarray,
    genres: Sequence[str],
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cos_matrix,
        xticklabels=genres,
        yticklabels=[f"Dir {i+1}" for i in range(len(genres))],
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        vmin=-1.0,
        vmax=1.0,
    )
    plt.title("Experiment A: Direction vs. Genre Alignment")
    plt.xlabel("Genre")
    plt.ylabel("Direction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Experiment B: Controllability curves
# ---------------------------------------------------------------------------

def ndcg_from_ranks(ranks: np.ndarray) -> np.ndarray:
    return 1.0 / np.log2(ranks + 1)


def evaluate_direction_effect(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    test_items: np.ndarray,
    direction: np.ndarray,
    alphas: Sequence[float],
    top_ks: Sequence[int],
    batch_size: int = 256,
) -> Tuple[Dict[float, np.ndarray], Dict[float, Dict[int, np.ndarray]]]:
    """Evaluate rank positions for each alpha and top-K."""

    num_users = len(test_items)
    direction = direction.astype(np.float32)
    ranks_per_alpha: Dict[float, np.ndarray] = {
        alpha: np.empty(num_users, dtype=np.int32) for alpha in alphas
    }
    hits_per_alpha: Dict[float, Dict[int, np.ndarray]] = {
        alpha: {k: np.empty(num_users, dtype=bool) for k in top_ks} for alpha in alphas
    }

    num_batches = (num_users + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_users)
        user_batch = user_factors[start:end]
        if user_batch.size == 0:
            continue
        base_scores = user_batch @ item_factors.T
        test_idx = test_items[start:end]
        for alpha in alphas:
            scores = base_scores + alpha * direction
            target_scores = scores[np.arange(end - start), test_idx]
            ranks = np.sum(scores >= target_scores[:, None], axis=1)
            ranks_per_alpha[alpha][start:end] = ranks
            for k in top_ks:
                hits = ranks <= k
                hits_per_alpha[alpha][k][start:end] = hits
    return ranks_per_alpha, hits_per_alpha


def plot_dose_response(
    alphas: Sequence[float],
    delta_logit_target: Sequence[float],
    delta_logit_other: Sequence[float],
    delta_ndcg_target: Sequence[float],
    delta_ndcg_other: Sequence[float],
    genre: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(alphas, delta_logit_target, marker="o", label=f"{genre} Δlogit", color="tab:blue")
    ax1.plot(alphas, delta_logit_other, marker="o", label="Other genres Δlogit", color="tab:blue", linestyle="--")
    ax1.set_xlabel("Perturbation strength α")
    ax1.set_ylabel("Δlogit", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(alphas, delta_ndcg_target, marker="s", label=f"{genre} ΔNDCG", color="tab:orange")
    ax2.plot(alphas, delta_ndcg_other, marker="s", label="Other genres ΔNDCG", color="tab:orange", linestyle="--")
    ax2.set_ylabel("ΔNDCG", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title(f"Experiment B: Dose–response for {genre}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_topk_improvement(
    top_ks: Sequence[int],
    improvements: Sequence[float],
    genre: str,
    alpha: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[str(k) for k in top_ks], y=improvements, color="tab:green")
    plt.xlabel("Top-K")
    plt.ylabel("Hit-rate improvement")
    plt.title(f"Experiment B: Top-K gain for {genre} at α={alpha}")
    plt.axhline(0.0, color="black", linewidth=0.8)
    for idx, val in enumerate(improvements):
        offset = 0.001 if val >= 0 else -0.001
        plt.text(idx, val + offset, f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiments(
    data_dir: Path,
    output_dir: Path,
    rank: int,
    target_genre: str,
    alphas: Sequence[float],
    top_ks: Sequence[int],
) -> None:
    num_items = load_num_items(data_dir / "item_ids.json")
    item_to_genres, genres = load_item_genres(data_dir / "item_text.json")
    for idx in range(num_items):
        item_to_genres.setdefault(idx, [])
    user_sequences = load_user_sequences(data_dir / "finetune.jsonl")
    matrix, test_items = build_user_item_matrix(user_sequences, num_items)
    if rank < len(genres):
        raise ValueError(
            f"Requested rank {rank} is smaller than the number of genres {len(genres)}; "
            "increase --rank to at least match."
        )
    alphas = sorted({float(a) for a in alphas})
    if 0.0 not in alphas:
        alphas = [0.0] + alphas
    effective_users = np.where(test_items >= 0)[0]
    if effective_users.size == 0:
        raise RuntimeError("No users with hold-out items were found.")
    matrix = matrix.tocsr()[effective_users]
    test_items = test_items[effective_users]
    user_factors, item_factors, vt = compute_latent_factors(matrix, rank=min(rank, len(genres)))

    genre_to_items_idx: Dict[str, np.ndarray] = {}
    for genre in genres:
        indices = [idx for idx, g_list in item_to_genres.items() if genre in g_list]
        genre_to_items_idx[genre] = np.array(indices, dtype=np.int32)
    aligned_dirs, aligned_cos, aligned_genres = align_directions_with_genres(vt, genre_to_items_idx, genres)

    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = output_dir / "experiment_a_alignment.png"
    plot_alignment_heatmap(aligned_cos, aligned_genres, heatmap_path)
    metrics = compute_alignment_metrics(aligned_cos)
    metrics_path = output_dir / "experiment_a_metrics.json"
    metrics_out = {"metrics": metrics, "genres": aligned_genres}
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    if target_genre not in aligned_genres:
        raise ValueError(f"Target genre '{target_genre}' not recognised.")
    genre_index = aligned_genres.index(target_genre)
    direction = aligned_dirs[genre_index]
    direction = direction - direction.mean()
    direction = direction / np.linalg.norm(direction)

    ranks, hits = evaluate_direction_effect(
        user_factors,
        item_factors,
        test_items,
        direction,
        alphas,
        top_ks,
    )
    base_ranks = ranks[alphas[0]]
    base_ndcg = ndcg_from_ranks(base_ranks)
    item_genres_lookup = {idx: item_to_genres.get(idx, []) for idx in range(num_items)}
    user_genres = [item_genres_lookup.get(item, []) for item in test_items]
    target_mask = np.array([target_genre in g_list for g_list in user_genres])
    other_mask = ~target_mask
    if not np.any(target_mask):
        raise RuntimeError(f"No evaluation users found for genre '{target_genre}'.")
    if not np.any(other_mask):
        raise RuntimeError("All evaluation users belong to the target genre; cannot form comparison group.")

    delta_logit_target = []
    delta_logit_other = []
    delta_ndcg_target = []
    delta_ndcg_other = []
    for alpha in alphas:
        direction_contrib = alpha * direction[test_items]
        delta_logit_target.append(float(direction_contrib[target_mask].mean()))
        delta_logit_other.append(float(direction_contrib[other_mask].mean()))
        ndcg = ndcg_from_ranks(ranks[alpha])
        delta = ndcg - base_ndcg
        delta_ndcg_target.append(float(delta[target_mask].mean()))
        delta_ndcg_other.append(float(delta[other_mask].mean()))

    dose_response_path = output_dir / f"experiment_b_dose_response_{target_genre}.png"
    plot_dose_response(
        alphas,
        delta_logit_target,
        delta_logit_other,
        delta_ndcg_target,
        delta_ndcg_other,
        target_genre,
        dose_response_path,
    )

    reference_alpha = alphas[-1]
    improvements = []
    for k in top_ks:
        base_hit = hits[alphas[0]][k][target_mask].mean()
        pert_hit = hits[reference_alpha][k][target_mask].mean()
        improvements.append(float(pert_hit - base_hit))
    topk_path = output_dir / f"experiment_b_topk_gain_{target_genre}.png"
    plot_topk_improvement(top_ks, improvements, target_genre, reference_alpha, topk_path)

    summary = {
        "alphas": list(alphas),
        "delta_logit_target": delta_logit_target,
        "delta_logit_other": delta_logit_other,
        "delta_ndcg_target": delta_ndcg_target,
        "delta_ndcg_other": delta_ndcg_other,
        "topk_improvements": {str(k): improvements[i] for i, k in enumerate(top_ks)},
        "target_genre": target_genre,
    }
    (output_dir / f"experiment_b_summary_{target_genre}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STREAM alignment experiments on MovieLens-10M")
    parser.add_argument("--data_dir", type=Path, default=Path("ml-10M100K"), help="Path to MovieLens-10M data directory")
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/outputs"), help="Where to store figures and summaries")
    parser.add_argument("--rank", type=int, default=19, help="Latent rank for SVD (defaults to number of genres)")
    parser.add_argument("--target_genre", type=str, default="Action", help="Genre to analyse in experiment B")
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0], help="Perturbation strengths")
    parser.add_argument("--topk", type=int, nargs="*", default=[5, 10, 20], help="Top-K thresholds for evaluation")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rank=args.rank,
        target_genre=args.target_genre,
        alphas=args.alphas,
        top_ks=args.topk,
    )


if __name__ == "__main__":
    main()
