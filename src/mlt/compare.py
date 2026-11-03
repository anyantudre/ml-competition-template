"""Is run B really better than run A? A paired bootstrap over the same out-of-fold rows.

Both runs are scored on the same resampled rows each time, so the spread of the difference
reflects only the rows, not two independent draws. Reading the result:
    |delta| < 1 SE        a tie, do not trust either direction
    1 SE <= |delta| < 2   leaning one way, confirm with another seed or the leaderboard
    |delta| >= 2 SE       likely a real difference
Keep in mind that the best of many tried variants looks better than it is.
"""

from __future__ import annotations

import numpy as np

from .config import targets
from .data import TargetEncoder
from .ensemble import Member
from .metrics import get_metric, score


def compare_runs(run_a: str, run_b: str, n_boot: int = 1000, seed: int = 0) -> dict:
    a, b = Member(run_a), Member(run_b)
    if a.state["task"] != b.state["task"] or a.state["metric"] != b.state["metric"]:
        raise ValueError("runs must share task and metric")
    if a.info["folds_hash"] != b.info["folds_hash"]:
        print("warning: the runs used different folds, so OOF rows are not strictly comparable")
    task, metric = a.state["task"], a.state["metric"]
    m = get_metric(metric)
    if task == "segmentation":
        y = None
        pa, pb = a.oof["dice"].to_numpy(), b.oof["dice"].to_numpy()
        ok = ~np.isnan(pa) & ~np.isnan(pb)

        def fn(idx, p):
            return float(np.mean(p[idx]))
    else:
        enc = TargetEncoder.from_state(a.state["target"])
        y = enc.transform(a.oof[targets(a.cfg)])
        pa, pb = a.oof[a.pred_cols].to_numpy(float), b.oof[b.pred_cols].to_numpy(float)
        ok = ~np.isnan(pa).any(1) & ~np.isnan(pb).any(1)
        thr = a.cfg["submission"]["threshold"]

        def fn(idx, p):
            return score(metric, y[idx], p[idx], task, threshold=thr)

    rows = np.flatnonzero(ok)
    sign = 1.0 if m.greater_is_better else -1.0
    base_a, base_b = fn(rows, pa), fn(rows, pb)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(rows, size=len(rows), replace=True)
        try:
            deltas.append(sign * (fn(idx, pb) - fn(idx, pa)))
        except ValueError:  # a resample with one class only
            continue
    deltas = np.asarray(deltas)
    delta = sign * (base_b - base_a)
    se = float(deltas.std(ddof=1)) if len(deltas) > 1 else float("nan")
    z = abs(delta) / se if se > 0 else float("inf")
    verdict = "tie" if z < 1 else ("leaning, needs confirmation" if z < 2 else "likely real")
    better = b.name if delta > 0 else a.name
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"{a.name}: {metric} {base_a:.5f}")
    print(f"{b.name}: {metric} {base_b:.5f}")
    print(f"improvement of B over A: {delta:+.5f}  (SE {se:.5f}, 95% interval {lo:+.5f} .. {hi:+.5f})")
    print(f"B better in {100 * np.mean(deltas > 0):.1f}% of {len(deltas)} resamples -> {verdict}"
          + ("" if verdict == "tie" else f" ({better})"))
    return {"a": base_a, "b": base_b, "delta": delta, "se": se, "z": z, "verdict": verdict}
