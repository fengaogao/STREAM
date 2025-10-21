import os, json, math, argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(); q = q / q.sum()
    return float((p * (np.log(p) - np.log(q))).sum() + (q * (np.log(q) - np.log(p))).sum())

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Jensen–Shannon divergence, symmetric & bounded in [0, ln(2)]."""
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * ((p * (np.log(p) - np.log(m))).sum() +
                  (q * (np.log(q) - np.log(m))).sum())



def rolling_conformal_bounds(values: List[float], m: int, alpha: float):
    v = np.array(values, dtype=float); n = len(v)
    lower = np.full(n, np.nan); upper = np.full(n, np.nan)
    for t in range(m-1, n):
        ref = v[t-m+1:t+1]
        mu = ref.mean()
        resid = np.abs(ref - mu)
        q = max(np.quantile(resid, 1 - alpha), 1e-6)
        lower[t] = mu - q; upper[t] = mu + q
    return lower, upper

# -------------------- Load ML-10M --------------------

def load_ratings(ml_root: str) -> pd.DataFrame:
    rp = os.path.join(ml_root, 'ratings.dat')
    if not os.path.exists(rp):
        rp = os.path.join(ml_root, 'ml-10M100K', 'ratings.dat')
    if not os.path.exists(rp):
        raise FileNotFoundError('ratings.dat not found')
    df = pd.read_csv(rp, sep='::', engine='python', header=None, names=['user','item','rating','ts'])
    df['user']=df['user'].astype(int); df['item']=df['item'].astype(int)
    df['rating']=df['rating'].astype(float); df['ts']=df['ts'].astype(int)
    df.sort_values(['user','ts'], inplace=True)
    return df


def load_movies(ml_root: str) -> pd.DataFrame:
    mp = os.path.join(ml_root, 'movies.dat')
    if not os.path.exists(mp):
        mp = os.path.join(ml_root, 'ml-10M100K', 'movies.dat')
    if not os.path.exists(mp):
        raise FileNotFoundError('movies.dat not found')
    # movies.dat: MovieID::Title::Genres
    dfm = pd.read_csv(mp, sep='::', engine='python', header=None, names=['item','title','genres'])
    dfm['item']=dfm['item'].astype(int)
    # 解析年份
    def norm_title(s):
        try:
            if s.endswith(')') and '(' in s:
                year = s[s.rfind('(')+1:-1]
                if year.isdigit():
                    base = s[:s.rfind('(')].strip()
                    return base, year
        except Exception:
            pass
        return s, ''
    rows=[]
    for _,r in dfm.iterrows():
        t,y = norm_title(str(r['title']))
        genres = str(r['genres']).replace('|', ', ')
        if y:
            txt = f"{t} ({y}) | Genres: {genres}"
        else:
            txt = f"{t} | Genres: {genres}"
        rows.append((int(r['item']), txt))
    df_txt = pd.DataFrame(rows, columns=['item','text'])
    return df_txt

def build_item_genres_map(ml_root: str):
    mp = os.path.join(ml_root, 'movies.dat')
    if not os.path.exists(mp):
        mp = os.path.join(ml_root, 'ml-10M100K', 'movies.dat')
    if not os.path.exists(mp):
        raise FileNotFoundError('movies.dat not found')

    dfm = pd.read_csv(mp, sep='::', engine='python', header=None, names=['item','title','genres'])
    dfm['item'] = dfm['item'].astype(int)

    id2genres = {}
    all_genres = set()
    for _, r in dfm.iterrows():
        gs = str(r['genres']).split('|') if pd.notnull(r['genres']) else []
        gs = [g.strip() for g in gs if g and g != '(no genres listed)']
        if not gs:
            gs = ['_UNK_']
        id2genres[int(r['item'])] = gs
        all_genres.update(gs)

    genre_list = sorted(all_genres)
    genre_index = {g:i for i,g in enumerate(genre_list)}
    if '_UNK_' not in genre_index:
        genre_list.append('_UNK_')
        genre_index['_UNK_'] = len(genre_list) - 1
    return id2genres, genre_list, genre_index


# -------------------- Sessionization + Drift --------------------

def sessionize_user(user_df: pd.DataFrame, gap_hours: int):
    items = user_df['item'].tolist(); ratings = user_df['rating'].tolist(); tss = user_df['ts'].tolist()
    if gap_hours<=0: return [(items, ratings, tss)]
    gap = gap_hours*3600; sessions=[]
    cur_i,cur_r,cur_t=[items[0]],[ratings[0]],[tss[0]]
    for i in range(1,len(items)):
        if tss[i]-tss[i-1]>gap:
            sessions.append((cur_i,cur_r,cur_t)); cur_i,cur_r,cur_t=[],[],[]
        cur_i.append(items[i]); cur_r.append(ratings[i]); cur_t.append(tss[i])
    sessions.append((cur_i,cur_r,cur_t))
    return sessions


def detect_drifts_for_user(items: List[int], ratings: List[float], m: int, kl_sym_th: float, conf_alpha: float, conf_q: int) -> List[int]:
    n=len(items)
    if n<2*m+5: return []
    pref = np.array([1.0 if r>=4.0 else 0.0 for r in ratings], dtype=float)
    lower,upper = rolling_conformal_bounds(pref.tolist(), m, conf_alpha)
    triggers = np.zeros(n, dtype=bool)
    for t in range(m, n-m):
        old = items[t-m:t]; new = items[t:t+m]
        vocab = sorted(set(old)|set(new)); idx={v:k for k,v in enumerate(vocab)}
        c_old = np.ones(len(vocab)); c_new=np.ones(len(vocab))
        for a in old: c_old[idx[a]]+=1
        for a in new: c_new[idx[a]]+=1
        p_old = c_old/c_old.sum(); p_new=c_new/c_new.sum()
        d_sym = symmetric_kl(p_old,p_new)
        kl_flag = d_sym>kl_sym_th
        conf_flag=False
        if not np.isnan(lower[t]) and not np.isnan(upper[t]):
            mu_new = pref[t:t+m].mean()
            conf_flag = (mu_new<lower[t] or mu_new>upper[t])
        triggers[t]= kl_flag or conf_flag
    drifts=[]; run=0
    for t in range(n):
        if triggers[t]:
            run+=1
            if run>=conf_q:
                drifts.append(t); run=0
        else:
            run=0
    # 合并近邻
    merged=[]; last=-10**9
    for t in drifts:
        if t-last>=m:
            merged.append(t); last=t
    return merged

def detect_drifts_for_user_genre(items: List[int], ratings: List[float], m: int,
                                          kl_sym_th: float, conf_q: int,
                                          id2genres: Dict[int, List[str]], genre_index: Dict[str, int]) -> List[int]:
    n = len(items)
    if n < 2*m + 5:
        return []

    G = len(genre_index)
    triggers = np.zeros(n, dtype=bool)

    for t in range(m, n - m):
        old_pairs = [(items[j], ratings[j]) for j in range(t - m, t)]
        new_pairs = [(items[j], ratings[j]) for j in range(t, t + m)]

        c_old = np.ones(G);
        c_new = np.ones(G)
        for a, r in old_pairs:
            if r >= 4.0:
                gs = id2genres.get(int(a), ['_UNK_'])
                w = 1.0 / max(1, len(gs))
                for g in gs:
                    c_old[genre_index.get(g, genre_index['_UNK_'])] += w
        for a, r in new_pairs:
            if r >= 4.0:
                gs = id2genres.get(int(a), ['_UNK_'])
                w = 1.0 / max(1, len(gs))
                for g in gs:
                    c_new[genre_index.get(g, genre_index['_UNK_'])] += w

        p_old = c_old / c_old.sum()
        p_new = c_new / c_new.sum()
        d_js = js_divergence(p_old, p_new)
        triggers[t] = (d_js > kl_sym_th)

    drifts, run = [], 0
    for t in range(n):
        if triggers[t]:
            run += 1
            if run >= conf_q:
                drifts.append(t); run = 0
        else:
            run = 0

    merged, last = [], -10**9
    for t in drifts:
        if t - last >= m:
            merged.append(t); last = t
    return merged




def assign_splits_with_guard(tss: List[int], drifts: List[int], n: int, guard_band_events: int, guard_band_days: int, t_min_ratio: float, t_min_days: int):
    if n==0: return {'O':(0,0),'F':(0,0),'T':(0,0)}
    if len(drifts)==0:
        # 60/20/20 + guard，单调推进
        gb=max(0,guard_band_events)
        O_l,O_r=0,min(int(0.6*n),n)
        F_l=max(O_r+gb,0); F_r=min(F_l+int(0.2*n), n)
        T_l=min(F_r+gb, n); T_r=n
        min_T=max(int(math.ceil(n*t_min_ratio)),0)
        if T_r-T_l<min_T:
            need=min_T-(T_r-T_l)
            take=min(need, max(0,F_r-F_l)); F_r-=take
            T_l=min(F_r+gb,n)
            need=max(0,need-take)
            if need>0:
                O_r=max(0,O_r-need); F_l=min(n,O_r+gb); F_r=max(F_l,F_r); T_l=min(F_r+gb,n)
        return {'O':(O_l,O_r),'F':(F_l,F_r),'T':(T_l,T_r)}
    first, last = drifts[0], drifts[-1]
    gb=max(0,guard_band_events)
    # 先按事件 guard
    O_l,O_r=0, max(0, first-gb)
    F_l,F_r=min(n,O_r+gb), min(n, last+gb)
    T_l,T_r = min(n,F_r+gb), n
    # 再按天数 guard 修正（优先级更高）
    if guard_band_days>0:
        gd=guard_band_days*86400
        first_ts=tss[first]; last_ts=tss[last]
        idxs_o=[i for i,ts in enumerate(tss) if ts<= first_ts-gd]
        O_r=(max(idxs_o)+1) if idxs_o else 0
        idxs_f=[i for i,ts in enumerate(tss) if ts>= last_ts+gd]
        F_r=(min(idxs_f)) if idxs_f else n
        F_l=min(n,O_r); T_l=min(n,F_r)
    # 统一单调+事件 guard
    F_l=min(n, max(F_l, O_r+gb)); F_r=max(F_l,F_r); T_l=min(n, max(T_l, F_r+gb))
    # T 段下限（比例与天数）
    min_T_by_ratio=int(math.ceil(n*t_min_ratio))
    min_T_by_days=0
    if t_min_days>0:
        min_tail_ts=tss[-1]-t_min_days*86400
        tail_candidates=[i for i,ts in enumerate(tss) if ts>=min_tail_ts]
        if tail_candidates:
            min_T_by_days = n - min(tail_candidates)
    min_T_needed=max(min_T_by_ratio, min_T_by_days)
    if n-T_l<min_T_needed:
        T_l=max(0, n-min_T_needed)
        F_r=min(F_r, T_l-gb); F_r=max(F_r,F_l)
    return {'O':(0,min(O_r,n)), 'F':(F_l,min(F_r,n)), 'T':(min(T_l,n), n)}

def choose_window(n: int, m_default: int, lo: int = 10, hi: int = 50, frac: float = 0.20) -> int:
    m = int(round(n * frac))
    m = max(lo, min(hi, m))
    m = min(m, m_default)
    if 2 * m + 5 > n:
        m = max(lo, min((n - 5) // 2, m))
    return m


def choose_guard(n: int, m_user: int, gb_default: int, min_gb: int = 5, max_frac: float = 0.10, m_factor: float = 0.5) -> int:
    cand = int(round(m_factor * m_user))
    gb = min(gb_default, int(max_frac * n), cand)
    return max(min_gb, gb)


# -------------------- Main --------------------
import math

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ml_root', type=str, default="ml-10M100K")
    ap.add_argument('--out_dir', type=str, default="ml-10M100K")
    ap.add_argument('--session_gap_hours', type=int, default=12)
    ap.add_argument('--glr_window', type=int, default=15)#50
    ap.add_argument('--kl_sym_th', type=float, default=0.05)
    ap.add_argument('--conf_alpha', type=float, default=0.10)
    ap.add_argument('--conf_q', type=int, default=3)
    ap.add_argument('--guard_band_events', type=int, default=0)
    ap.add_argument('--guard_band_days', type=int, default=0)
    ap.add_argument('--t_min_ratio', type=float, default=0.10)
    ap.add_argument('--t_min_days', type=int, default=7)
    ap.add_argument('--min_user_len', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--m_lo', type=int, default=5, help='自适应窗口下限')
    ap.add_argument('--m_hi', type=int, default=30, help='自适应窗口上限')
    ap.add_argument('--m_frac', type=float, default=0.20, help='窗口占用户长度比例 (20%)')
    ap.add_argument('--gb_min', type=int, default=3, help='事件隔离带下限')
    ap.add_argument('--gb_max_frac', type=float, default=0.10, help='隔离带占用户长度上限比例 (10%)')
    ap.add_argument('--gb_m_factor', type=float, default=0.50, help='隔离带约为 m 的多少倍 (0.5*m)')

    args=ap.parse_args()

    np.random.seed(args.seed); ensure_dir(args.out_dir)

    print('[Load] ratings & movies ...')
    df = load_ratings(args.ml_root)
    df = df[df['rating'] >= 4.0].copy()
    dfm = load_movies(args.ml_root)
    id2genres, genre_list, genre_index = build_item_genres_map(args.ml_root)
    title_map = {int(r["item"]): r["text"] for _,r in dfm.iterrows()}

    # 过滤短用户
    user_lengths = df.groupby('user')['item'].count()
    keep = set(user_lengths[user_lengths>=args.min_user_len].index)
    df = df[df['user'].isin(keep)].copy()

    # item vocab + text
    items_sorted = sorted(df['item'].unique().tolist())
    mid2idx = {int(mid): idx for idx, mid in enumerate(items_sorted)}
    item_text = {mid2idx[mid]: title_map.get(mid, f'Item {mid}') for mid in items_sorted}

    # 输出句柄
    fO=open(os.path.join(args.out_dir,'original.jsonl'),'w',encoding='utf-8')
    fF=open(os.path.join(args.out_dir,'finetune.jsonl'),'w',encoding='utf-8')
    fT=open(os.path.join(args.out_dir,'test.jsonl'),'w',encoding='utf-8')

    drift_points_out={}; user_splits_out={}
    total_O=total_F=total_T=0

    print('[Process] sessionize + drift detect + split')
    for uid, g in tqdm(df.groupby('user')):
        g=g.sort_values('ts')
        sessions = sessionize_user(g, args.session_gap_hours)
        items=[]; ratings=[]; tss=[]
        for it,rt,tt in sessions:
            items.extend(it); ratings.extend(rt); tss.extend(tt)
        n=len(items)
        if n<args.min_user_len: continue

        m_user = choose_window(
            n=n,
            m_default=args.glr_window,
            lo=getattr(args, 'm_lo', 5),
            hi=getattr(args, 'm_hi', 30),
            frac=getattr(args, 'm_frac', 0.20),
        )
        # gb_events = choose_guard(
        #     n=n,
        #     m_user=m_user,
        #     gb_default=args.guard_band_events,
        #     min_gb=getattr(args, 'gb_min', 3),
        #     max_frac=getattr(args, 'gb_max_frac', 0.10),
        #     m_factor=getattr(args, 'gb_m_factor', 0.50),
        # )
        gb_events = 0
        drifts = detect_drifts_for_user_genre(items, ratings, m=m_user, kl_sym_th=args.kl_sym_th, conf_q=args.conf_q, id2genres=id2genres,  genre_index=genre_index)
        splits = assign_splits_with_guard(tss, drifts, n, guard_band_events=gb_events, guard_band_days=args.guard_band_days, t_min_ratio=args.t_min_ratio, t_min_days=args.t_min_days)
        (lO,rO),(lF,rF),(lT,rT) = splits['O'], splits['F'], splits['T']
        def dump(l,r,fh):
            if r-l<=0: return 0
            seq_items=[mid2idx[int(x)] for x in items[l:r]]
            seq_times=tss[l:r]
            fh.write(json.dumps({'user':int(uid),'items':seq_items,'times':seq_times}, ensure_ascii=False)+'\n')
            return len(seq_items)
        total_O += dump(lO,rO,fO)
        total_F += dump(lF,rF,fF)
        total_T += dump(lT,rT,fT)
        drift_points_out[str(int(uid))]={'drifts_idx':drifts,'drifts_ts':[int(tss[i]) for i in drifts],'splits':{'O':[lO,rO],'F':[lF,rF],'T':[lT,rT]}}
        user_splits_out[str(int(uid))]={'O':[lO,rO],'F':[lF,rF],'T':[lT,rT]}

    fO.close(); fF.close(); fT.close()

    with open(os.path.join(args.out_dir,'item_ids.json'),'w',encoding='utf-8') as f:
        json.dump({'mid2idx': mid2idx}, f)
    with open(os.path.join(args.out_dir,'item_text.json'),'w',encoding='utf-8') as f:
        json.dump(item_text, f)
    with open(os.path.join(args.out_dir,'drift_points.json'),'w',encoding='utf-8') as f:
        json.dump(drift_points_out, f)
    with open(os.path.join(args.out_dir,'user_splits.json'),'w',encoding='utf-8') as f:
        json.dump(user_splits_out, f)

    stats={'n_users': int(df['user'].nunique()), 'n_items': int(len(mid2idx)), 'events_O':int(total_O),'events_F':int(total_F),'events_T':int(total_T), 'args': vars(args)}
    with open(os.path.join(args.out_dir,'stats.json'),'w',encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print('[Done] Outputs at', args.out_dir)

if __name__=='__main__':
    main()
