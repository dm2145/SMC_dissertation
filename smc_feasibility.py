"""
SMC feasibility prototype (D1, Section 4).

Runs entirely on SIMULATED EUR/USD-like 15-minute prices, so no market data
(and no ethics approval) is needed. It checks four things:

  1. Pattern detection works: fair value gaps (FVGs) and breaks of structure (BOS).
  2. Hindsight test: detectors give the same past signals whether they see the
     data only up to time T or the full history. A deliberately "naive" swing
     detector that peeks at future bars is included to show the test catches it.
  3. Backtest + matched random entries: FVG trades vs many random-entry runs,
     with a permutation p-value. On random prices there should be NO edge.
  4. Speed and statistical power: how long 1,000 random runs take, and the
     smallest edge (in R) the real test could detect.

Usage:
    python smc_feasibility.py                 
    python smc_feasibility.py --runs 2000 --years 2 --seed 1

"""

import argparse
import os
import time
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIP = 0.0001


# 1. Simulated price data
def simulate_ohlc(years, seed, annual_vol=0.07, start_price=1.10):
    """Random-walk prices: 15 one-minute steps per 15-minute bar.
    annual_vol ~7% is roughly EUR/USD's long-run volatility. Weekends are ignored."""
    rng = np.random.default_rng(seed)
    bars = int(years * 260 * 96)               # 260 trading days, 96 bars/day
    steps_per_bar = 15
    minute_sigma = annual_vol / np.sqrt(260 * 96 * steps_per_bar)
    log_steps = rng.normal(0.0, minute_sigma, size=bars * steps_per_bar)
    minute_prices = start_price * np.exp(np.cumsum(log_steps))
    m = minute_prices.reshape(bars, steps_per_bar)
    prev_close = np.concatenate([[start_price], m[:-1, -1]])
    df = pd.DataFrame({
        "open": prev_close,
        "high": np.maximum(m.max(axis=1), prev_close),
        "low": np.minimum(m.min(axis=1), prev_close),
        "close": m[:, -1],
    })
    df.index = pd.date_range("2015-01-05", periods=bars, freq="15min")
    return df


# 2. Pattern detectors
#    Each signal records the bar that it becomes KNOWN ("avail").
def detect_fvg(df, min_gap_pips=1.0):
    """Bullish FVG at bar i: high[i-2] < low[i]. Bearish: low[i-2] > high[i].
    Known at the close of bar i (uses bars i-2..i only)."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    gap_up = l[2:] - h[:-2]
    gap_dn = l[:-2] - h[2:]
    out = []
    for i in np.where(gap_up > min_gap_pips * PIP)[0] + 2:
        out.append((i, 1, l[i - 2]))          # (avail bar, direction, stop level)
    for i in np.where(gap_dn > min_gap_pips * PIP)[0] + 2:
        out.append((i, -1, h[i - 2]))
    out.sort()
    return out


def swings_causal(df, k):
    """Swing high at bar j if high[j] beats the k bars either side.
    It can only be KNOWN at bar j+k, so avail = j+k (no hindsight)."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(h)
    res = []
    for j in range(k, n - k):
        if h[j] > h[j - k:j].max() and h[j] > h[j + 1:j + k + 1].max():
            res.append((j + k, "H", j, h[j]))
        if l[j] < l[j - k:j].min() and l[j] < l[j + 1:j + k + 1].min():
            res.append((j + k, "L", j, l[j]))
    res.sort()
    return res


def swings_naive(df, k):
    """SAME swings, but treated as known at bar j itself (avail = j).
    This is how swings look on a finished chart -- it uses future bars."""
    return sorted((j, t, j, p) for (_, t, j, p) in swings_causal(df, k))


def detect_bos(df, swings):
    """Bullish BOS: a close above the most recent available swing high.
    Bearish BOS: a close below the most recent available swing low.
    Each swing level can be broken once."""
    c = df["close"].to_numpy()
    by_avail = {}
    for s in swings:
        by_avail.setdefault(s[0], []).append(s)
    last_h = last_l = None
    out = []
    for t in range(len(c)):
        for (_, typ, _, price) in by_avail.get(t, []):
            if typ == "H":
                last_h = price
            else:
                last_l = price
        if last_h is not None and c[t] > last_h:
            out.append((t, 1, last_h)); last_h = None
        if last_l is not None and c[t] < last_l:
            out.append((t, -1, last_l)); last_l = None
    return out


# 3. Hindsight test
def truncation_test(df, detector, cut_points):
    """For each cut T: signals from data[:T] must equal the full-data signals
    that were available before T. Returns number of cut points that failed."""
    full = detector(df)
    fails = 0
    for T in cut_points:
        part = detector(df.iloc[:T])
        expected = [s for s in full if s[0] < T]
        if part != expected:
            fails += 1
    return fails


# 4. Trade simulation 
def simulate_trades(df, entry_bars, directions, risks, target_r, max_bars, spread_pips):
    """Enter at the OPEN of entry_bar. Stop = risk away, target = target_r * risk.
    If stop and target are hit in the same bar, assume the stop (conservative).
    Exit at close after max_bars if neither is hit. Spread is charged in R."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    entry_bars = np.asarray(entry_bars)
    directions = np.asarray(directions)
    risks = np.asarray(risks)
    idx = entry_bars[:, None] + np.arange(max_bars)[None, :]
    entry = o[entry_bars]
    hi, lo = h[idx], l[idx]
    fav = np.where(directions[:, None] == 1, hi - entry[:, None], entry[:, None] - lo)
    adv = np.where(directions[:, None] == 1, entry[:, None] - lo, hi - entry[:, None])
    hit_stop = adv >= risks[:, None]
    hit_tgt = fav >= target_r * risks[:, None]
    first_stop = np.where(hit_stop.any(1), hit_stop.argmax(1), max_bars)
    first_tgt = np.where(hit_tgt.any(1), hit_tgt.argmax(1), max_bars)
    final = (c[idx[:, -1]] - entry) * directions / risks
    r = np.where(first_stop <= first_tgt,
                 np.where(first_stop < max_bars, -1.0, final),
                 target_r)
    r = np.where((first_stop == max_bars) & (first_tgt == max_bars), final, r)
    return r - spread_pips * PIP / risks


def fvg_trades(df, fvgs, max_bars):
    """Turn FVG signals into trades: enter next bar, stop beyond candle 1."""
    o = df["open"].to_numpy()
    n = len(df)
    e, d, rk = [], [], []
    for (i, direction, stop) in fvgs:
        if i + 1 + max_bars >= n:
            continue
        risk = (o[i + 1] - stop) * direction
        if risk > 0.5 * PIP:                    # skip tiny stops
            e.append(i + 1); d.append(direction); rk.append(risk)
    return np.array(e), np.array(d), np.array(rk)


# 5. Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3, help="swing confirmation bars")
    ap.add_argument("--target_r", type=float, default=2.0)
    ap.add_argument("--max_bars", type=int, default=96, help="timeout (96 = 1 day)")
    ap.add_argument("--spread", type=float, default=0.5, help="spread in pips")
    a = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    # Data
    t0 = time.perf_counter()
    df = simulate_ohlc(a.years, a.seed)
    say(f"DATA: {len(df):,} simulated 15-minute bars ({a.years} years), "
        f"{df.memory_usage().sum() / 1e6:.1f} MB in memory, "
        f"built in {time.perf_counter() - t0:.2f} s")
    say(f"      Real test period 2022-2025 would be about {4 * 260 * 96:,} bars.")

    # Detection
    fvgs = detect_fvg(df)
    sw = swings_causal(df, a.k)
    bos = detect_bos(df, sw)
    say()
    say("DETECTION")
    say(f"  FVGs: {len(fvgs):,}  ({sum(1 for s in fvgs if s[1] == 1):,} bullish, "
        f"{sum(1 for s in fvgs if s[1] == -1):,} bearish)")
    say(f"  Swing points (k={a.k}): {len(sw):,}   Breaks of structure: {len(bos):,}")

    # Hindsight test
    rng = np.random.default_rng(a.seed + 1)
    test_df = df.iloc[:20000]                    # subset keeps the test quick
    cuts = sorted(rng.integers(500, len(test_df), size=100))
    tests = {
        "FVG detector": lambda d: detect_fvg(d),
        "Swing points (causal, confirmed after k bars)": lambda d: swings_causal(d, a.k),
        "Swing points (NAIVE, known at the swing bar)": lambda d: swings_naive(d, a.k),
        "BOS (causal swings, confirmed after k bars)": lambda d: detect_bos(d, swings_causal(d, a.k)),
        "BOS (NAIVE swings, known at the swing bar)": lambda d: detect_bos(d, swings_naive(d, a.k)),
    }
    say()
    say(f"HINDSIGHT (TRUNCATION) TEST: {len(cuts)} random cut points")
    for name, fn in tests.items():
        fails = truncation_test(test_df, fn, cuts)
        say(f"  {name:45s} {'PASS' if fails == 0 else 'FAIL'}  ({fails}/{len(cuts)} cut points differ)")

    # Backtest FVG trades
    e, d, rk = fvg_trades(df, fvgs, a.max_bars)
    r = simulate_trades(df, e, d, rk, a.target_r, a.max_bars, a.spread)
    smc_mean = r.mean()
    say()
    say(f"BACKTEST (FVG, target {a.target_r}R, timeout {a.max_bars} bars, spread {a.spread} pips)")
    say(f"  Trades: {len(r):,}   Mean R: {smc_mean:+.4f}   SD of R: {r.std(ddof=1):.3f}   "
        f"Win rate: {(r > 0).mean():.1%}")
    say("  Note: trades may overlap in time, so they are not fully independent;")
    say("  the power figures below are therefore optimistic.")

    # Matched random entries
    eligible = np.arange(10, len(df) - a.max_bars - 1)
    t0 = time.perf_counter()
    rand_means = np.empty(a.runs)
    for k in range(a.runs):
        re = rng.choice(eligible, size=len(e), replace=False)
        rd = rng.permutation(d)
        rr = rng.permutation(rk)
        rand_means[k] = simulate_trades(df, re, rd, rr, a.target_r, a.max_bars, a.spread).mean()
    secs = time.perf_counter() - t0
    p = (1 + np.sum(rand_means >= smc_mean)) / (a.runs + 1)
    say()
    say(f"MATCHED RANDOM ENTRIES: {a.runs:,} runs (same number, directions and stop sizes)")
    say(f"  Random mean R: {rand_means.mean():+.4f}  (2.5-97.5%: {np.percentile(rand_means, 2.5):+.4f} "
        f"to {np.percentile(rand_means, 97.5):+.4f})")
    say(f"  One-sided permutation p-value: {p:.3f}  (expected: not significant on random data)")
    say(f"  Time: {secs:.1f} s for {a.runs:,} runs -> about {secs * 10000 / a.runs / 60:.1f} min for 10,000")

    # One trade at a time (3.4): skip signals while a trade is open
    o_, h_a, l_a = (df[x].to_numpy() for x in ("open", "high", "low"))
    idx = e[:, None] + np.arange(a.max_bars)[None, :]
    ent = o_[e]
    fav = np.where(d[:, None] == 1, h_a[idx] - ent[:, None], ent[:, None] - l_a[idx])
    adv = np.where(d[:, None] == 1, ent[:, None] - l_a[idx], h_a[idx] - ent[:, None])
    hs, ht = adv >= rk[:, None], fav >= a.target_r * rk[:, None]
    fs = np.where(hs.any(1), hs.argmax(1), a.max_bars)
    ft = np.where(ht.any(1), ht.argmax(1), a.max_bars)
    exit_bar = e + np.minimum(np.minimum(fs, ft), a.max_bars - 1)
    keep, free = [], -1
    for i in range(len(e)):
        if e[i] > free:
            keep.append(i); free = exit_bar[i]
    r1 = r[np.array(keep)]
    say()
    say(f"ONE TRADE AT A TIME: {len(r1):,} trades ({len(r1) / a.years:,.0f} per year), mean R {r1.mean():+.4f}")

    # Power / minimum detectable effect
    z_a = NormalDist().inv_cdf(1 - 0.05 / 4)    # strictest Holm step, 4 tests
    z_b = NormalDist().inv_cdf(0.80)            # 80% power
    say()
    say("STATISTICAL POWER (alpha = 0.05/4 one-sided, power 80%): smallest detectable edge in R per trade")
    for label, rr in (("overlapping trades", r), ("one trade at a time", r1)):
        sd = rr.std(ddof=1)
        n4 = int(len(rr) / a.years * 4)
        mde = (z_a + z_b) * sd / np.sqrt(n4)
        say(f"  {label:20s} 4-year test, n = {n4:6,}: {mde:.3f} R   "
            f"(with a second pair, n = {2 * n4:,}: {mde / np.sqrt(2):.3f} R)")

    # Figure 1: example chart with FVGs
    start = next(i for (i, _, _) in fvgs if i > 200) - 20
    seg = df.iloc[start:start + 70]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for x, (_, row) in enumerate(seg.iterrows()):
        col = "#2a9d8f" if row.close >= row.open else "#e76f51"
        ax.vlines(x, row.low, row.high, color=col, linewidth=0.8)
        ax.add_patch(plt.Rectangle((x - 0.3, min(row.open, row.close)), 0.6,
                                   max(abs(row.close - row.open), 1e-6), color=col))
    h_, l_ = df["high"].to_numpy(), df["low"].to_numpy()
    for (i, direction, _) in fvgs:
        if start + 2 <= i < start + 70:
            x0 = i - 2 - start
            lo_, hi_ = (h_[i - 2], l_[i]) if direction == 1 else (h_[i], l_[i - 2])
            ax.add_patch(plt.Rectangle((x0, lo_), 2.5, hi_ - lo_,
                                       color="#4c78a8" if direction == 1 else "#b279a2", alpha=0.3))
    ax.set_xlim(-1, 70)
    ax.set_title("Detected fair value gaps on simulated 15-minute prices (blue = bullish, purple = bearish)", fontsize=9)
    ax.set_xlabel("Bar"); ax.set_ylabel("Price")
    fig.tight_layout(); fig.savefig("results/fig_fvg_example.png", dpi=200); plt.close(fig)

    # Figure 2: random-entry distribution
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(rand_means, bins=40, color="#9aa5b1")
    ax.axvline(smc_mean, color="#c0392b", linewidth=2, label=f"FVG trades ({smc_mean:+.3f} R)")
    ax.set_xlabel("Mean R per trade"); ax.set_ylabel("Number of random runs")
    ax.set_title(f"FVG trades vs {a.runs:,} matched random-entry runs (simulated data, p = {p:.2f})", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/fig_random_entries.png", dpi=200); plt.close(fig)

    with open("results/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved: results/summary.txt, results/fig_fvg_example.png, results/fig_random_entries.png")


if __name__ == "__main__":
    main()
