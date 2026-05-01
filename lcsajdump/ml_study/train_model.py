"""
trainer.py — Train a LightGBM LambdaRank model on the gadget dataset.

Requires: lightgbm, scikit-learn, pandas, numpy
Install:  pip install lightgbm scikit-learn pandas numpy shap

Usage
-----
    # Full pipeline
    python -m lcsajdump.ml.trainer --out model.pkl

    # Or in Python:
    from lcsajdump.ml.trainer import train_pipeline
    model, report = train_pipeline(X, y, groups)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import lightgbm as lgb
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import ndcg_score
except ImportError as e:
    print(
        f"[trainer] Missing dependency: {e}\n"
        f"  pip install lightgbm scikit-learn pandas numpy shap",
        file=sys.stderr,
    )
    sys.exit(1)

from lcsajdump.ml.features import FEATURE_NAMES, ARCH_PROFILES
from lcsajdump.ml_study.build_dataset import build_dataset, ALL_SAMPLES

# ── Hyperparameters ────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # LambdaRank objective: optimises NDCG directly.
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "label_gain": [0, 1],  # gain for label 0 and 1
    # Tree structure — deeper trees to capture finer-grained gadget patterns
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 1,
    # Learning
    "learning_rate": 0.01,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "early_stopping_round": 200,
    "verbose": -1,
    "random_state": 42,
}


# ── SHAP weight interpretation ─────────────────────────────────────────────────


def interpret_as_config_weights(model, X_df: pd.DataFrame, arch: str) -> dict:
    """
    Use SHAP mean absolute values to derive config.py-compatible weight suggestions.

    The mapping is:
      insn_count        → insn_penalty        (higher SHAP = higher penalty per instruction)
      hits_link_reg     → bonus_link_reg
      hits_arg_reg      → bonus_arg_reg
      hits_frame_reg    → bonus_frame_reg
      is_internal_call  → penalty_internal_call
      is_trampoline_term→ bonus_trampoline
      is_ret_terminated → bonus_ret (or penalty if negative)

    Returns a dict ready to paste into config.py scoring_weights.
    """
    try:
        import shap
    except ImportError:
        print(
            "[trainer] shap not installed — skipping weight interpretation",
            file=sys.stderr,
        )
        return {}

    # Filter to arch-specific rows for per-arch interpretation
    arch_col = f"arch_{arch}"
    if arch_col in X_df.columns:
        X_arch = X_df[X_df[arch_col] == 1]
    else:
        X_arch = X_df

    if X_arch.empty:
        return {}

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_arch)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    feat_importance = dict(zip(FEATURE_NAMES, mean_abs))

    # Get per-feature direction (positive SHAP = higher score = good gadget)
    mean_signed = shap_vals.mean(axis=0)
    feat_direction = dict(zip(FEATURE_NAMES, mean_signed))

    def _scale(feature: str, base: int = 50) -> int:
        """Scale raw SHAP magnitude to an integer weight."""
        imp = feat_importance.get(feature, 0)
        total = sum(feat_importance.values()) or 1
        return max(1, int(round(imp / total * base * len(FEATURE_NAMES))))

    def _signed_scale(feature: str, base: int = 50) -> int:
        raw = _scale(feature, base)
        if feat_direction.get(feature, 0) < 0:
            return -raw  # penalising feature
        return raw

    current_profile = ARCH_PROFILES.get(arch, {})
    current_weights = current_profile.get("scoring_weights", {})

    suggested = {
        "base_score": current_weights.get("base_score", 100),
        "insn_penalty": abs(_signed_scale("insn_count", 30)),
        "bonus_link_reg": abs(_scale("hits_link_reg", 80)),
        "bonus_arg_reg": abs(_scale("hits_arg_reg", 80)),
        "bonus_frame_reg": abs(_scale("hits_frame_reg", 80)),
        "penalty_internal_call": abs(_scale("is_internal_call", 200)),
        "bonus_trampoline": abs(_scale("is_trampoline_term", 80)),
        "bonus_pivot": abs(_scale("is_pivot_gadget", 80)),
        "penalty_bad_ret": abs(_scale("is_ret_terminated", 40)),
    }

    print(f"\n[trainer] Suggested scoring_weights for '{arch}':")
    print(f"  Current:   {current_weights}")
    print(f"  Suggested: {suggested}")
    return suggested


# ── Training pipeline ──────────────────────────────────────────────────────────


def train_pipeline(
    X: list[dict],
    y: list[int],
    groups: list[int],
    params: Optional[dict] = None,
    val_fraction: float = 0.2,
    model_output: str = "gadget_model.pkl",
    meta: Optional[list[dict]] = None,
) -> tuple:
    """
    Train a LightGBM LambdaRank model and save it.

    Parameters
    ----------
    X : list[dict]  — feature dicts (from build_dataset)
    y : list[int]   — binary labels
    groups : list[int]  — group sizes (one per binary)
    params : dict   — LightGBM params (defaults to DEFAULT_PARAMS)
    val_fraction : float  — fraction of groups held out for validation
    model_output : str    — where to save the trained model

    Returns
    -------
    model : lgb.LGBMRanker
    report : dict  — training metrics and suggested weights per arch
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    y_arr = np.array(y, dtype=np.float32)
    groups_arr = np.array(groups, dtype=np.int32)

    # ── Group-aware train/validation split ────────────────────────────────────
    # Build group index array: each sample gets its group index
    group_idx = np.repeat(np.arange(len(groups)), groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_groups, val_groups = next(gss.split(X_df, y_arr, groups=group_idx))

    X_train = X_df.iloc[train_groups]
    y_train = y_arr[train_groups]
    g_train = _rebuild_groups(group_idx, train_groups)

    X_val = X_df.iloc[val_groups]
    y_val = y_arr[val_groups]
    g_val = _rebuild_groups(group_idx, val_groups)

    print(
        f"[trainer] Train: {len(X_train)} samples, {int(y_train.sum())} pos, "
        f"{len(g_train)} groups"
    )
    print(
        f"[trainer] Val:   {len(X_val)} samples, {int(y_val.sum())} pos, "
        f"{len(g_val)} groups"
    )

    # ── Cap oversized groups (LightGBM LambdaRank limit: 10 000 rows/query) ────
    y_train, g_train, X_train = _cap_group_sizes(y_train, g_train, X_train)
    y_val, g_val, X_val = _cap_group_sizes(y_val, g_val, X_val)

    # ── Handle degenerate groups (all-zero labels) ────────────────────────────
    # LambdaRank requires at least one positive per group.
    # If no positive in a group, fall back to the heuristic score as pseudo-label.
    y_train_clean, g_train_clean, X_train_clean = _fix_degenerate_groups(
        y_train, g_train, X_train
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model = lgb.LGBMRanker(**params)
    model.fit(
        X_train_clean,
        y_train_clean,
        group=g_train_clean,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100, verbose=True, first_metric_only=False
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    meta_val = None
    if meta is not None:
        meta_arr = np.array(meta, dtype=object)
        meta_val = meta_arr[val_groups].tolist()
    report = _evaluate(model, X_val, y_val, g_val, meta=meta_val)
    print(f"[trainer] Validation NDCG@5: {report.get('ndcg_5', 0):.4f}")

    # ── Global feature importance ─────────────────────────────────────────────
    _print_feature_importances(model, X_df, y_arr)

    # ── Per-arch weight interpretation ────────────────────────────────────────
    report["suggested_weights"] = {}
    for arch in ("x86_64", "x86_32", "arm64", "riscv64"):
        w = interpret_as_config_weights(model, X_df, arch)
        if w:
            report["suggested_weights"][arch] = w

    # ── Save ──────────────────────────────────────────────────────────────────
    model_data = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "params": params,
        "report": report,
    }
    with open(model_output, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[trainer] Model saved to {model_output}")

    return model, report


def _rebuild_groups(group_idx: np.ndarray, sample_idx: np.ndarray) -> list[int]:
    """Rebuild group sizes from a sample index subset."""
    from collections import Counter

    g_indices = group_idx[sample_idx]
    counts = Counter(g_indices)
    return [counts[i] for i in sorted(counts)]


def _cap_group_sizes(
    y: np.ndarray,
    groups: list[int],
    X: pd.DataFrame,
    max_size: int = 9000,
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    """
    Split groups larger than max_size into chunks of at most max_size rows.

    LightGBM LambdaRank raises a fatal error when a single query exceeds
    10 000 rows. This splits oversized groups by stratified chunking so
    each sub-group still contains both positive and negative examples.
    """
    new_groups: list[int] = []
    keep_idx: list[int] = []
    offset = 0

    for size in groups:
        if size <= max_size:
            new_groups.append(size)
            keep_idx.extend(range(offset, offset + size))
        else:
            # Stratified split: interleave positives and negatives so each
            # chunk gets a representative mix.
            idx = np.arange(offset, offset + size)
            pos_idx = idx[y[idx] == 1]
            neg_idx = idx[y[idx] == 0]

            # Shuffle within each class for variety across chunks
            rng = np.random.default_rng(42)
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)

            # Distribute into chunks
            n_chunks = int(np.ceil(size / max_size))
            pos_chunks = np.array_split(pos_idx, n_chunks)
            neg_chunks = np.array_split(neg_idx, n_chunks)

            for pc, nc in zip(pos_chunks, neg_chunks):
                chunk_idx = np.concatenate([pc, nc])
                chunk_idx.sort()
                new_groups.append(len(chunk_idx))
                keep_idx.extend(chunk_idx.tolist())

        offset += size

    keep = np.array(keep_idx, dtype=np.int64)
    return y[keep], new_groups, X.iloc[keep].reset_index(drop=True)


def _fix_degenerate_groups(
    y: np.ndarray, groups: list[int], X: pd.DataFrame
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    """
    For groups with no positive label, promote the highest heuristic_score
    gadget to pseudo-label=1 so LambdaRank doesn't crash.
    """
    y_out = y.copy()
    offset = 0
    for size in groups:
        chunk = y_out[offset : offset + size]
        if chunk.sum() == 0:
            # Promote highest heuristic_score in this group
            scores = X["heuristic_score"].iloc[offset : offset + size].values
            best = np.argmax(scores)
            y_out[offset + best] = 1
        offset += size
    return y_out, groups, X


def _print_feature_importances(model, X_df: pd.DataFrame, y_arr: np.ndarray):
    """Print SHAP mean-abs feature importances with direction sign."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_df)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        mean_sign = shap_vals.mean(axis=0)
        total = mean_abs.sum() or 1.0
        pairs = sorted(zip(FEATURE_NAMES, mean_abs, mean_sign), key=lambda t: -t[1])
        print("\n=== FEATURE IMPORTANCES (SHAP mean |value|) ===")
        for name, imp, sig in pairs:
            if imp < 1e-4:
                continue
            direction = "[+]" if sig >= 0 else "[-]"
            print(f"  {direction} {name:<30}: {imp / total:.4f}")
    except Exception as e:
        print(f"[trainer] SHAP feature importances unavailable: {e}")


def _evaluate(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    g_val: list[int],
    meta: Optional[list[dict]] = None,
) -> dict:
    """Compute NDCG@1,3,5,10 on the validation set, with per-group breakdown."""
    from sklearn.metrics import ndcg_score as _ndcg

    scores = model.predict(X_val)
    report = {}
    offset = 0
    ndcg_at = {1: [], 3: [], 5: [], 10: []}
    per_group: list[tuple[str, float, float, float]] = []

    for size in g_val:
        true_chunk = y_val[offset : offset + size]
        score_chunk = scores[offset : offset + size]
        if true_chunk.sum() == 0:
            offset += size
            continue
        group_ndcg = {}
        for k in ndcg_at:
            try:
                val = _ndcg([true_chunk], [score_chunk], k=k)
                ndcg_at[k].append(val)
                group_ndcg[k] = val
            except Exception:
                pass
        # derive group label from meta
        label = "?"
        if meta is not None:
            m = meta[offset]
            label = m.get("binary_id", m.get("binary", "?"))
        per_group.append(
            (label, group_ndcg.get(1, 0), group_ndcg.get(3, 0), group_ndcg.get(5, 0))
        )
        offset += size

    if per_group:
        print("\n=== VALIDATION NDCG (per group) ===")
        for label, n1, n3, n5 in per_group:
            print(f"  {label}: @1={n1:.3f} @3={n3:.3f} @5={n5:.3f}")

    for k, vals in ndcg_at.items():
        if vals:
            report[f"ndcg_{k}"] = float(np.mean(vals))

    return report


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train lcsajdump gadget ML scorer")
    parser.add_argument("--out", default="gadget_model.pkl", help="Output model path")
    parser.add_argument(
        "--csv", default=None, help="Pre-built CSV dataset (skips running lcsajdump)"
    )
    parser.add_argument(
        "--val", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument("--epochs", type=int, default=500, help="Max estimators")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        for col in FEATURE_NAMES:
            if col not in df.columns:
                print(f"[Warning] Manca la feature {col} nel CSV! Faccio padding con 0, ma dovresti ricreare il dataset.", file=sys.stderr)
                df[col] = 0
        X = df[FEATURE_NAMES].to_dict("records")
        y = df["label"].tolist()
        # Rebuild groups from binary_id column (unique per arch+binary pair)
        groups = df.groupby("binary_id", sort=False).size().tolist()
        meta = df[["binary_id", "binary", "arch", "address"]].to_dict("records")
    else:
        print("[trainer] Building dataset from all samples...")
        from lcsajdump.ml_study.build_dataset import build_dataset, ALL_SAMPLES

        X, y, groups, meta = build_dataset(ALL_SAMPLES)

    params = {**DEFAULT_PARAMS, "n_estimators": args.epochs}
    model, report = train_pipeline(
        X, y, groups, params=params, model_output=args.out, meta=meta
    )

    print("\n[trainer] Final report:")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "suggested_weights"}, indent=2
        )
    )

    if report.get("suggested_weights"):
        print("\n[trainer] Suggested config.py scoring_weights:")
        print(json.dumps(report["suggested_weights"], indent=2))
