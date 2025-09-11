# mc_robust.py — Monte Carlo robustness op daily PnL (fallback: trade events)
import argparse, math, numpy as np, pandas as pd, pathlib as pl

def load_series():
    if pl.Path("daily_pnl.csv").exists():
        df = pd.read_csv("daily_pnl.csv")
        # verwacht kolommen: date, day_pnl
        s = pd.to_numeric(df["day_pnl"], errors="coerce").dropna().reset_index(drop=True)
        kind = "daily"
    elif pl.Path("trades.csv").exists():
        df = pd.read_csv("trades.csv", parse_dates=["time"])
        df = df.sort_values("time")
        # neem alleen events die PnL beïnvloeden
        s = pd.to_numeric(df["pnl"], errors="coerce").dropna().reset_index(drop=True)
        kind = "events"
    else:
        raise SystemExit("daily_pnl.csv of trades.csv niet gevonden.")
    return s, kind

def run_mc(pnl_series, start_equity, n_paths=5000, block=0, scale=1.0, fixed_cost=0.0, seed=42):
    rng = np.random.default_rng(seed)
    N = len(pnl_series)
    pnl = pnl_series.values.astype(float)

    results = []
    for _ in range(n_paths):
        if block and block > 1:
            # block bootstrap op vaste blokgrootte
            idx = []
            while len(idx) < N:
                j = rng.integers(0, N)
                block_end = min(j + block, N)
                idx.extend(range(j, block_end))
            idx = np.array(idx[:N])
        else:
            idx = rng.integers(0, N, size=N)

        # stress: schaal PnL (fricties/slippage) en trek per stap fixed cost af
        resampled = pnl[idx] * scale - fixed_cost

        equity = start_equity + np.cumsum(resampled)
        peak = np.maximum.accumulate(np.r_[start_equity, equity[:-1]])
        dd = (equity - peak) / peak
        max_dd = dd.min() if dd.size else 0.0

        tot_ret = equity[-1] / start_equity - 1.0
        neg = (resampled < 0).sum()
        pos = (resampled > 0).sum()
        pf = (resampled[resampled > 0].sum() / abs(resampled[resampled <= 0].sum())
              if (resampled <= 0).any() else np.inf)

        results.append((tot_ret, max_dd, pf, pos, neg))

    out = pd.DataFrame(results, columns=["Return","MaxDD","PF","Wins","Losses"])
    return out

def summarize(df):
    q = df.quantile([0.05,0.5,0.95]).rename(index={0.05:"P5",0.5:"P50",0.95:"P95"})
    prob_loss = (df["Return"] < 0).mean()
    return q, prob_loss

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=200.0, help="Start equity")
    ap.add_argument("--paths", type=int, default=5000, help="Aantal MC paden")
    ap.add_argument("--block", type=int, default=3, help="Blokgrootte voor block bootstrap (0=iid)")
    ap.add_argument("--scale", type=float, default=0.9, help="PNL schaalfactor voor frictiestress (bv 0.9 = -10%)")
    ap.add_argument("--fixed", type=float, default=0.0, help="Vaste extra kost per dag/event in valuta")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s, kind = load_series()
    print(f"Bron: {kind} | observaties: {len(s)}")
    res = run_mc(s, start_equity=args.equity, n_paths=args.paths,
                 block=args.block, scale=args.scale, fixed_cost=args.fixed, seed=args.seed)
    q, p_loss = summarize(res)

    # Opslaan
    res.to_csv("mc_paths.csv", index=False)
    q.to_csv("mc_summary_quantiles.csv")
    with open("mc_readme.txt","w") as f:
        f.write(
            f"Paths={args.paths}\nBlock={args.block}\nScale={args.scale}\nFixed={args.fixed}\n"
            f"P_loss={p_loss:.3f}\n"
        )

    # Console output compact
    print("\n=== Monte Carlo samenvatting ===")
    print(q)
    print(f"\nKans op verlies (Return<0): {p_loss:.1%}")
    print("Bestanden: mc_paths.csv, mc_summary_quantiles.csv, mc_readme.txt")
