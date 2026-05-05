"""
optuna_unified.py — Ottimizzazione unificata di scoring_weights + search_params.

Ottimizza contemporaneamente:
1. scoring_weights (come in optuna_heuristic.py)
2. search_params: k, d, i, m per ogni architettura

Usage
-----
python -m lcsajdump.ml.optuna_unified \
    --csv lcsajdump/ml/datasets/gadget_dataset_v9.csv \
    --arch x86_64 \
    --trials 200 \
    --out /tmp/unified_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    import optuna
    import pandas as pd
    from sklearn.metrics import ndcg_score as _ndcg
except ImportError as e:
    print(
        f"Missing dependency: {e}\n pip install optuna pandas scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Heuristic scoring formula (mirrors rainbowBFS.score_gadget exactly) ───────


def _score_heuristic(row: dict, weights: dict) -> float:
    """
    Reproduce the RainbowBFS scoring formula from a feature-dict row.
    Includes clobber_penalty and long_chain_penalty to mirror the fixed scorer.
    """
    s = weights["base_score"]
    s -= row["insn_count"] * weights["insn_penalty"]
    s -= row.get("clobber_count", 0) * weights.get("clobber_penalty", 60)
    s -= max(0, row["insn_count"] - 5) * weights.get("long_chain_penalty", 25)
    if row["hits_link_reg"]:
        s += weights["bonus_link_reg"]
    if row["hits_arg_reg"]:
        s += weights["bonus_arg_reg"]
    if row["hits_frame_reg"]:
        s += weights["bonus_frame_reg"]
    if row["is_trampoline_term"]:
        s += weights["bonus_trampoline"]
    if row["is_internal_call"]:
        s -= weights["penalty_internal_call"]
    # Mirror rainbowBFS exactly: syscall terminators get bonus, not penalty
    if row.get("is_syscall_term", 0):
        s += weights.get("bonus_syscall", 60)
    elif row["is_ret_terminated"] and not row["hits_link_reg"]:
        s -= weights["penalty_bad_ret"]
    if row["has_direct_call"]:
        s += weights.get("bonus_direct_call", 0)
    if row.get("is_pivot_gadget", 0):
        s += weights.get("bonus_pivot", 0)
    return float(s)


def _eval_weights(df_arch: pd.DataFrame, weights: dict, search_params: dict) -> float:
    """
    Return mean NDCG@5 considering both scoring weights and search params.

    search_params: limit (gadgets to show), d (depth), i (instructions), m (min_score)
    Note: darkness/d affect BFS traversal and cannot be evaluated on a pre-built dataset.
    Filtra i gadget che supererebbero i limiti strutturali.
    """
    limit = search_params.get("limit", 9999)
    d_limit = search_params.get("d", 20)  # noqa: F841 — structural param, not filterable offline
    i_limit = search_params.get("i", 100)
    m_limit = search_params.get("m", 0)

    # Filter by instruction count only — heuristic_score column is pre-computed with
    # old weights and cannot be used as a filter here; apply m_limit to fresh scores below.
    df_filtered = df_arch[df_arch["insn_count"] <= i_limit].copy()

    if len(df_filtered) == 0:
        return 0.0

    # Score with current weights
    scores = df_filtered.apply(
        lambda r: _score_heuristic(r.to_dict(), weights), axis=1
    ).values

    # Apply m_limit to freshly-computed scores (not stale heuristic_score column)
    keep = scores >= m_limit
    df_filtered = df_filtered[keep]
    scores = scores[keep]

    if len(df_filtered) == 0:
        return 0.0

    labels = df_filtered["label"].values
    binary_ids = df_filtered["binary_id"].values

    # Apply k limit: only top-k per group
    ndcg_vals = []
    for bid in np.unique(binary_ids):
        mask = binary_ids == bid
        tc = labels[mask]
        sc = scores[mask]

        if tc.sum() == 0 or len(tc) < 2:
            continue

        # Apply limit: sort by score, take top-limit
        if len(sc) > limit:
            top_k_idx = np.argsort(sc)[-limit:][::-1]
            tc = tc[top_k_idx]
            sc = sc[top_k_idx]

        try:
            ndcg_vals.append(_ndcg([tc], [sc], k=5))
        except Exception:
            pass

    return float(np.mean(ndcg_vals)) if ndcg_vals else 0.0


# ── Optuna objective ───────────────────────────────────────────────────────────


def make_objective(df_arch: pd.DataFrame):
    """Return an Optuna objective that tunes weights + search params."""

    def objective(trial: optuna.Trial) -> float:
        # Scoring weights (da ottimizzare)
        weights = {
            "base_score": 100,
            "insn_penalty": trial.suggest_int("insn_penalty", 1, 30),
            "bonus_link_reg": trial.suggest_int("bonus_link_reg", 5, 100),
            "bonus_arg_reg": trial.suggest_int("bonus_arg_reg", 5, 150),
            "bonus_frame_reg": trial.suggest_int("bonus_frame_reg", 5, 100),
            "penalty_internal_call": trial.suggest_int("penalty_internal_call", 50, 300),
            "bonus_trampoline": trial.suggest_int("bonus_trampoline", 5, 100),
            "penalty_bad_ret": trial.suggest_int("penalty_bad_ret", 0, 200),
            "bonus_direct_call": trial.suggest_int("bonus_direct_call", 0, 80),
            "bonus_pivot": trial.suggest_int("bonus_pivot", 5, 150),
            "bonus_syscall": trial.suggest_int("bonus_syscall", 0, 150),
            "clobber_penalty": trial.suggest_int("clobber_penalty", 20, 200),
            "long_chain_penalty": trial.suggest_int("long_chain_penalty", 5, 80),
        }

        # Search params (da ottimizzare) — nomi devono corrispondere a config.py
        search_params = {
            "limit": trial.suggest_int("limit", 10, 50),
            "darkness": trial.suggest_int("darkness", 2, 8),
            "d": trial.suggest_int("d", 3, 10),
            "i": trial.suggest_int("i", 4, 15),     # was 10..200 — caused i=128 collapse
            "m": trial.suggest_int("m", 0, 20),
        }

        ndcg = _eval_weights(df_arch, weights, search_params)

        # Anti-degeneracy: penalize weights collapsed to floor (floor=5 for bonuses)
        floor_pen = 0.02 * sum(
            1 for k, v in weights.items()
            if k.startswith("bonus_") and v <= 5
        )
        return ndcg - floor_pen

    return objective


# ── Per-arch tuning ────────────────────────────────────────────────────────────


def tune_arch(df: pd.DataFrame, arch: str, n_trials: int = 100) -> dict:
    """Run Optuna on a single architecture. Returns best config dict."""
    arch_col = f"arch_{arch}"
    if arch_col not in df.columns:
        print(
            f"[unified] Column {arch_col} not found — skipping {arch}", file=sys.stderr
        )
        return {}

    df_arch = df[df[arch_col] == 1].copy()
    if df_arch.empty:
        print(f"[unified] No rows for {arch} — skipping", file=sys.stderr)
        return {}

    n_groups = df_arch["binary_id"].nunique()
    print(
        f"[unified] {arch}: {len(df_arch)} samples, {n_groups} groups, {n_trials} trials"
    )

    # Baseline: current config
    baseline_weights = {
        "base_score": 100,
        "insn_penalty": 15,
        "bonus_link_reg": 25,
        "bonus_arg_reg": 75,
        "bonus_frame_reg": 45,
        "penalty_internal_call": 50,
        "bonus_trampoline": 20,
        "penalty_bad_ret": 0,
        "bonus_direct_call": 15,
        "bonus_pivot": 0,
        "bonus_syscall": 60,
    }
    baseline_search = {"limit": 20, "darkness": 10, "d": 10, "i": 100, "m": 0}
    baseline = _eval_weights(df_arch, baseline_weights, baseline_search)
    print(f" Baseline NDCG@5 (current config.py) = {baseline:.4f}")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(make_objective(df_arch), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_weights = {
        "base_score": 100,
        **{k: v for k, v in best.params.items() if k not in ["limit", "darkness", "d", "i", "m"]},
    }
    best_search = {k: best.params[k] for k in ["limit", "darkness", "d", "i", "m"] if k in best.params}
    best_ndcg = best.value

    print(f" Best NDCG@5 = {best_ndcg:.4f} (Δ = {best_ndcg - baseline:+.4f})")
    print(f" Best weights: {best_weights}")
    print(f" Best search_params: {best_search}")

    return {
        "scoring_weights": best_weights,
        "search_params": best_search,
        "ndcg_improvement": best_ndcg - baseline,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune config.py scoring_weights + search_params with Optuna"
    )
    parser.add_argument("--csv", required=True, help="Labelled dataset CSV")
    parser.add_argument(
        "--arch",
        default=None,
        choices=["x86_64", "x86_32", "arm64", "riscv64"],
        help="Architecture to tune (default: all four)",
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Optuna trials per arch"
    )
    parser.add_argument(
        "--out",
        default="/tmp/unified_weights.json",
        help="Output JSON with suggested config",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    archs = [args.arch] if args.arch else ["x86_64", "x86_32", "arm64", "riscv64"]

    results = {}
    for arch in archs:
        print(f"\n{'=' * 50}")
        print(f" Tuning unified config for: {arch}")
        print(f"{'=' * 50}")
        w = tune_arch(df, arch, n_trials=args.trials)
        if w:
            results[arch] = w

    print(f"\n{'=' * 50}")
    print(" SUGGESTED config.py (scoring_weights + search_params)")
    print(f"{'=' * 50}")
    print(json.dumps(results, indent=2))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[unified] Saved to {args.out}")
    print("[unified] Apply these manually to lcsajdump/core/config.py")
