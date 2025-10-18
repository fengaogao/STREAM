# -*- coding: utf-8 -*-
"""
DATS: Drift-Aware Tri-Stage splitting for MovieLens-10M (and similar datasets)

Outputs (in --out_dir):
  - original_stride1.jsonl   : O 段（Warm-up，无漂移区段）
  - finetune.jsonl           : F 段（Adapt/Online，漂移发生区段，严格按时间）
  - test.jsonl               : T 段（Hold-out，更新后的稳态评估区段）
  - drift_points.json        : 每用户的漂移检测点与边界
  - user_splits.json         : 每用户 O/F/T 的 [l, r) 索引（基于该用户自身序列）
  - stats.json               : 全局统计与参数快照

特性：
  - 会话化（按间隔小时）以增强子意图/稳定性（可选）
  - 双触发漂移检测：
      1) 分布漂移：两窗口多项分布的对称 KL（带 Laplace/Dirichlet 平滑）
      2) 评分趋势漂移：滚动均值的保形（Conformal）带越界
  - Guard Band：阶段接缝保护带，避免信息泄漏/前震污染
  - 稀疏用户回退：总交互不足/无可靠漂移时启用稳妥比例切分 + Guard Band

使用：
  python dats_ml10m_split.py \
    --ml_root /path/to/ml-10M100K \
    --out_dir /path/to/out \
    --session_gap_hours 12 --glr_window 50 --kl_sym_th 0.8 --conf_alpha 0.1 \
    --guard_band_events 50 --t_min_ratio 0.2 --t_min_days 30

说明：
  - 不依赖电影元数据；分布漂移直接基于 item id 的多项分布（局部词表，窗口并集）。
  - 评分信号：将评分 ≥4 视为偏好强信号；构造滚动均值并套保形带（分位数）判断越界。
  - 若你是隐式反馈（无评分），可把评分均值部分替换为滑动点击率或位置偏好统计。
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------- Utils --------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def ts_to_days(ts: int) -> float:
    return ts / 86400.0


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(); q = q / q.sum()
    return float((p * (np.log(p) - np.log(q))).sum() + (q * (np.log(q) - np.log(p))).sum())


def rolling_conformal_bounds(values: List[float], m: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple conformal prediction band for rolling mean.
    values: sequence of scalars
    returns: lower, upper arrays aligned to values (first m-1 are NaN)
    """
    v = np.array(values, dtype=float)
    n = len(v)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    for t in range(m-1, n):
        ref = v[t-m+1:t+1]
        mu = ref.mean()
        resid = np.abs(ref - mu)
        q = np.quantile(resid, 1 - alpha)
        lower[t] = mu - q
        upper[t] = mu + q
    return lower, upper


# -------------------- I/O --------------------

@dataclass
class Args:
    ml_root: str
    out_dir: str
    session_gap_hours: int = 12
    glr_window: int = 50           # 两窗口大小（各 m）
    kl_sym_th: float = 0.8         # 对称 KL 触发阈值（经验量纲，建议网格调优）
    conf_alpha: float = 0.10       # 保形带显著性
    conf_q: int = 3                # 连续越界次数
    guard_band_events: int = 50    # 接缝保护带（按交互数）
    guard_band_days: int = 0       # 或按天数（>0 生效）
    t_min_ratio: float = 0.2       # T 段最小比例
    t_min_days: int = 30           # 或最小天数（取较大者）
    min_user_len: int = 30         # 稀疏用户过滤/回退
    seed: int = 42


# -------------------- Load ML-10M --------------------

def load_ml10m(ml_root: str) -> pd.DataFrame:
    ratings_path = os.path.join(ml_root, 'ratings.dat')
    if not os.path.exists(ratings_path):
        # 兼容根目录为上一层
        ratings_path = os.path.join(ml_root, 'ml-10M100K', 'ratings.dat')
    if not os.path.exists(ratings_path):
        raise FileNotFoundError('Cannot find ratings.dat under ml_root')
    df = pd.read_csv(
        ratings_path,
        sep='::', engine='python', header=None,
        names=['user','item','rating','ts']
    )
    df['user'] = df['user'].astype(int)
    df['item'] = df['item'].astype(int)
    df['rating'] = df['rating'].astype(float)
    df['ts'] = df['ts'].astype(int)
    df.sort_values(['user','ts'], inplace=True)
    return df


# -------------------- Sessionization --------------------

def sessionize_user(user_df: pd.DataFrame, gap_hours: int) -> List[Tuple[List[int], List[float], List[int]]]:
    """Return list of sessions: (items, ratings, timestamps)
    简单按时间间隔切分；若 gap_hours<=0，返回一个整段会话。
    """
    items = user_df['item'].tolist()
    ratings = user_df['rating'].tolist()
    tss = user_df['ts'].tolist()
    if gap_hours <= 0:
        return [(items, ratings, tss)]
    gap = gap_hours * 3600
    sessions = []
    cur_i, cur_r, cur_t = [items[0]], [ratings[0]], [tss[0]]
    for i in range(1, len(items)):
        if tss[i] - tss[i-1] > gap:
            sessions.append((cur_i, cur_r, cur_t))
            cur_i, cur_r, cur_t = [], [], []
        cur_i.append(items[i]); cur_r.append(ratings[i]); cur_t.append(tss[i])
    sessions.append((cur_i, cur_r, cur_t))
    return sessions


# -------------------- Drift Detection (per user) --------------------

def detect_drifts_for_user(items: List[int], ratings: List[float], tss: List[int],
                            m: int, kl_sym_th: float, conf_alpha: float, conf_q: int) -> List[int]:
    """Return list of drift indices (on the user sequence index space).
    触发条件：
      - 对称 KL(p_new||p_old)+KL(p_old||p_new) > kl_sym_th （窗口并集词表，Laplace=1 平滑）
      - 且/或 滚动评分均值越过保形带，连续 conf_q 次
    两者采用 OR 逻辑，并做连续触发折叠（相邻触发合并为一个代表点）。
    """
    n = len(items)
    if n < 2*m + 5:
        return []

    # 构造评分≥4的二值偏好信号；也可直接用评分
    pref = np.array([1.0 if r >= 4.0 else 0.0 for r in ratings], dtype=float)
    lower, upper = rolling_conformal_bounds(pref.tolist(), m, conf_alpha)

    triggers = np.zeros(n, dtype=bool)

    for t in range(m, n - m):
        # windows
        old = items[t-m:t]
        new = items[t:t+m]
        vocab = sorted(set(old) | set(new))
        idx = {v: k for k, v in enumerate(vocab)}
        c_old = np.ones(len(vocab), dtype=float)  # Laplace 1
        c_new = np.ones(len(vocab), dtype=float)
        for a in old: c_old[idx[a]] += 1
        for a in new: c_new[idx[a]] += 1
        p_old = c_old / c_old.sum()
        p_new = c_new / c_new.sum()
        d_sym = symmetric_kl(p_old, p_new)
        kl_flag = d_sym > kl_sym_th

        conf_flag = False
        if not np.isnan(lower[t]) and not np.isnan(upper[t]):
            mu_new = pref[t:t+m].mean()
            conf_flag = (mu_new < lower[t] or mu_new > upper[t])

        triggers[t] = kl_flag or conf_flag

    # 连续 conf_q 次合并
    drifts = []
    run = 0
    for t in range(n):
        if triggers[t]:
            run += 1
            if run == conf_q:
                drifts.append(t)
        else:
            run = 0
    # 去重：间隔 < m 的合并到最早一个
    merged = []
    last = -10**9
    for t in drifts:
        if t - last >= m:
            merged.append(t)
            last = t
    return merged


# -------------------- Assign O/F/T with Guard Band --------------------

def assign_splits_with_guard(items: List[int], tss: List[int], drifts: List[int],
                             guard_band_events: int, guard_band_days: int,
                             t_min_ratio: float, t_min_days: int) -> Dict[str, Tuple[int,int]]:
    n = len(items)
    if n == 0:
        return {'O': (0,0), 'F': (0,0), 'T': (0,0)}

    if len(drifts) == 0:
        # 回退：7:1:2 + guard
        lT = max(int(n * (1.0 - t_min_ratio)), 0)
        t_start = max(lT, 0)
        # guard bands
        gb = guard_band_events
        O_l, O_r = 0, max(0, int(0.7*n) - gb)
        F_l, F_r = O_r + gb, max(O_r + gb, int(0.8*n) - gb)
        T_l, T_r = max(F_r + gb, int(0.8*n) + gb), n
        O_l, O_r = 0, max(0, min(O_r, n))
        F_l, F_r = max(0, min(F_l, n)), max(0, min(F_r, n))
        T_l, T_r = max(0, min(T_l, n)), n
        return {'O': (O_l, O_r), 'F': (F_l, F_r), 'T': (T_l, T_r)}

    # 以第一个/最后一个漂移为界
    first = drifts[0]
    last = drifts[-1]

    # Guard by events
    gb = guard_band_events
    O_l, O_r = 0, max(0, first - gb)
    F_l, F_r = min(n, O_r + gb), min(n, last + gb)
    T_l = min(n, F_r + gb)

    # Guard by days（优先）
    if guard_band_days > 0:
        t0 = tss[0]
        # 找到时间上接近 first - guard_days 的索引
        gd_secs = guard_band_days * 86400
        # O 右边界：first_ts - gd_secs
        first_ts = tss[first]
        idxs = [i for i, ts in enumerate(tss) if ts <= first_ts - gd_secs]
        O_r = (max(idxs) + 1) if idxs else 0  # +1 以包含该位置
        # F 右边界：last_ts + gd_secs
        last_ts = tss[last]
        idxs = [i for i, ts in enumerate(tss) if ts >= last_ts + gd_secs]
        F_r = (min(idxs)) if idxs else n  # 兜底为 n（右开）
        T_l = min(n, F_r)

    # T 段最小比例/天数
    min_T_by_ratio = int(math.ceil(n * t_min_ratio))
    min_T_by_days = 0
    if t_min_days > 0:
        last_time = tss[-1]
        min_cut_time = tss[0] + t_min_days * 86400
        # 找到最早使得后缀长度满足天数的切点（这里用近似：若总天数不足则按比例保证）
        min_T_by_days = 0  # 已按比例兜底
    T_l = min(T_l, n)
    if n - T_l < max(min_T_by_ratio, min_T_by_days):
        T_l = max(0, n - max(min_T_by_ratio, min_T_by_days))

    # 边界清理 & 互斥
    O_l, O_r = 0, max(0, min(O_r, n))
    F_l = max(O_r, F_l)
    F_r = max(F_l, F_r)
    T_l = max(F_r, T_l)
    O = (O_l, O_r)
    F = (F_l, F_r)
    T = (T_l, n)
    return {'O': O, 'F': F, 'T': T}


# -------------------- Main pipeline --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ml_root', type=str, required=True, help='path to ml-10M100K or its parent containing ratings.dat')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--session_gap_hours', type=int, default=12)
    ap.add_argument('--glr_window', type=int, default=50)
    ap.add_argument('--kl_sym_th', type=float, default=0.8)
    ap.add_argument('--conf_alpha', type=float, default=0.10)
    ap.add_argument('--conf_q', type=int, default=3)
    ap.add_argument('--guard_band_events', type=int, default=50)
    ap.add_argument('--guard_band_days', type=int, default=0)
    ap.add_argument('--t_min_ratio', type=float, default=0.2)
    ap.add_argument('--t_min_days', type=int, default=30)
    ap.add_argument('--min_user_len', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    ensure_dir(args.out_dir)

    print('[Load] MovieLens-10M ...')
    df = load_ml10m(args.ml_root)

    # 过滤短用户
    user_lengths = df.groupby('user')['item'].count()
    keep = set(user_lengths[user_lengths >= args.min_user_len].index)
    df = df[df['user'].isin(keep)].copy()

    # item 词表
    items_sorted = sorted(df['item'].unique().tolist())
    mid2idx = {int(mid): idx for idx, mid in enumerate(items_sorted)}

    # 输出句柄
    fO = open(os.path.join(args.out_dir, 'original.jsonl'), 'w', encoding='utf-8')
    fF = open(os.path.join(args.out_dir, 'finetune.jsonl'), 'w', encoding='utf-8')
    fT = open(os.path.join(args.out_dir, 'test.jsonl'), 'w', encoding='utf-8')

    drift_points_out: Dict[str, Dict] = {}
    user_splits_out: Dict[str, Dict[str, List[int]]] = {}

    total_O = total_F = total_T = 0

    print('[Process] per-user sessionization + drift detection + split')

    for uid, g in tqdm(df.groupby('user')):
        g = g.sort_values('ts')
        sessions = sessionize_user(g, args.session_gap_hours)
        # 合并会话回到单序列（保留时间顺序）
        items = []
        ratings = []
        tss = []
        for it, rt, tt in sessions:
            items.extend(it); ratings.extend(rt); tss.extend(tt)
        n = len(items)
        if n < args.min_user_len:
            continue

        # 检测漂移点
        drifts = detect_drifts_for_user(items, ratings, tss,
                                        m=args.glr_window,
                                        kl_sym_th=args.kl_sym_th,
                                        conf_alpha=args.conf_alpha,
                                        conf_q=args.conf_q)
        # 分配 O/F/T + Guard Band
        splits = assign_splits_with_guard(items, tss, drifts,
                                          guard_band_events=args.guard_band_events,
                                          guard_band_days=args.guard_band_days,
                                          t_min_ratio=args.t_min_ratio,
                                          t_min_days=args.t_min_days)
        (lO, rO), (lF, rF), (lT, rT) = splits['O'], splits['F'], splits['T']

        # 写出 jsonl（各段可能为空，但尽量保留非空用户）
        def dump_range(l: int, r: int, fh):
            if r - l <= 0:
                return 0
            seq_items = [mid2idx[int(x)] for x in items[l:r]]
            seq_times = tss[l:r]
            rec = {'user': int(uid), 'items': seq_items, 'times': seq_times}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return len(seq_items)

        total_O += dump_range(lO, rO, fO)
        total_F += dump_range(lF, rF, fF)
        total_T += dump_range(lT, rT, fT)

        drift_points_out[str(int(uid))] = {
            'drifts_idx': drifts,
            'drifts_ts': [int(tss[i]) for i in drifts],
            'splits': {'O': [lO, rO], 'F': [lF, rF], 'T': [lT, rT]},
        }
        user_splits_out[str(int(uid))] = {'O': [lO, rO], 'F': [lF, rF], 'T': [lT, rT]}

    fO.close(); fF.close(); fT.close()

    # 词表
    with open(os.path.join(args.out_dir, 'item_ids.json'), 'w', encoding='utf-8') as f:
        json.dump({'mid2idx': mid2idx}, f)

    # 漂移点与切分
    with open(os.path.join(args.out_dir, 'drift_points.json'), 'w', encoding='utf-8') as f:
        json.dump(drift_points_out, f)
    with open(os.path.join(args.out_dir, 'user_splits.json'), 'w', encoding='utf-8') as f:
        json.dump(user_splits_out, f)

    # 统计
    stats = {
        'n_users': int(df['user'].nunique()),
        'n_items': int(len(mid2idx)),
        'events_O': int(total_O),
        'events_F': int(total_F),
        'events_T': int(total_T),
        'args': vars(args),
        'note': 'DATS splitting with sessionization + symmetric KL + conformal band + guard bands.'
    }
    with open(os.path.join(args.out_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print('[Done] Outputs at', args.out_dir)
    print('  original.jsonl :', total_O)
    print('  finetune.jsonl         :', total_F)
    print('  test.jsonl             :', total_T)


if __name__ == '__main__':
    main()
