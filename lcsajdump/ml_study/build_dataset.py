"""
dataset_builder.py — Build a labelled gadget dataset from exploit scripts.

Usage
-----
    from lcsajdump.ml.dataset_builder import build_dataset, THESIS_SAMPLES
    X, y, groups, meta = build_dataset(THESIS_SAMPLES)

    # Or from the command line:
    python -m lcsajdump.ml.dataset_builder --out dataset.csv

Positive label (y=1): gadget address was used in an exploit script.
Negative label (y=0): gadget was found by lcsajdump but not used.

Groups: per-binary group sizes for LambdaRank (LightGBM/XGBoost).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from lcsajdump.ml.features import (
    extract_features,
    FEATURE_NAMES,
    FEATURE_NAMES_WITH_LM,
    ARCH_PROFILES,
)

# ── Address extraction from exploit scripts ───────────────────────────────────

_HEX_RE = re.compile(r"\b(0x[0-9a-fA-F]{3,12})\b")

# Patterns for pwntools ELF symbol access: elf.symbols['name'] / elf.sym['name']
_ELF_SYM_RE = re.compile(
    r'elf\.(?:symbols?|sym)\s*\[[\'"]([\w]+)[\'"]\]',
    re.IGNORECASE,
)

# exe.sym / exe.symbols (common alias)
_EXE_SYM_RE = re.compile(
    r'(?:exe|binary)\.(?:symbols?|sym)\s*\[[\'"]([\w]+)[\'"]\]',
    re.IGNORECASE,
)


def _resolve_elf_symbols(binary_path: str, symbol_names: set) -> dict[str, int]:
    """
    Resolve ELF symbol names to addresses using pyelftools.
    Returns {name: address} for found symbols.
    """
    if not symbol_names:
        return {}
    try:
        from elftools.elf.elffile import ELFFile

        result = {}
        with open(binary_path, "rb") as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name(".symtab") or elf.get_section_by_name(
                ".dynsym"
            )
            if symtab:
                for sym in symtab.iter_symbols():
                    if sym.name in symbol_names and sym["st_value"]:
                        result[sym.name] = sym["st_value"]
        return result
    except Exception as e:
        print(
            f"[dataset_builder] ELF symbol resolution failed for {binary_path}: {e}",
            file=sys.stderr,
        )
        return {}


def extract_gadget_addresses(exploit_path: str, binary_path: str) -> set[int]:
    """
    Parse an exploit script and return the set of gadget addresses it uses.

    Handles:
      - Direct hex literals: pivot = 0x40100f
      - ELF symbol references: elf.symbols['gadget_set_x0'] / exe.sym['name']
      - pwntools flat({offset: addr, ...}) patterns
    """
    source = Path(exploit_path).read_text(errors="replace")

    addresses: set[int] = set()

    # 1. Direct hex literals
    for m in _HEX_RE.finditer(source):
        try:
            addresses.add(int(m.group(1), 16))
        except ValueError:
            pass

    # 2. ELF symbol references
    syms = set()
    for m in _ELF_SYM_RE.finditer(source):
        syms.add(m.group(1))
    for m in _EXE_SYM_RE.finditer(source):
        syms.add(m.group(1))

    if syms and binary_path and os.path.exists(binary_path):
        resolved = _resolve_elf_symbols(binary_path, syms)
        addresses.update(resolved.values())

    return addresses


# ── lcsajdump JSON runner ─────────────────────────────────────────────────────


def _run_lcsajdump(
    binary_path: str,
    arch: str = "auto",
    depth: int = 20,
    darkness: int = 5,
    instructions: int = 15,
    all_exec: bool = False,
) -> dict:
    """Run lcsajdump CLI and return parsed JSON output."""
    cmd = [
        sys.executable,
        "-m",
        "lcsajdump.cli",
        binary_path,
        "--json",
        "--limit",
        "999999",
        "--depth",
        str(depth),
        "--darkness",
        str(darkness),
        "--instructions",
        str(instructions),
        "--algo",
    ]
    if arch != "auto":
        cmd += ["--arch", arch]
    if all_exec:
        cmd += ["--all-exec"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"lcsajdump CLI failed (exit {result.returncode}) for {binary_path}:\n"
            f"{result.stderr[-1000:]}"
        )

    stdout = result.stdout
    start = stdout.find("{")
    if start == -1:
        raise RuntimeError(f"No JSON in lcsajdump output for {binary_path}")
    return json.loads(stdout[start:])


# ── Feature + label extraction ────────────────────────────────────────────────


def _gadgets_from_json(data: dict) -> list[dict]:
    """Flatten sequential + jump_based entries from lcsajdump JSON."""
    gadgets = []
    for gtype_key, gtype_label in (
        ("sequential", "Sequential"),
        ("jump_based", "Jump-Based"),
    ):
        for entry in data.get(gtype_key, []):
            gadgets.append(
                {
                    "address": int(entry["primary_address"], 16),
                    "duplicate_addresses": [
                        int(a, 16) for a in entry.get("duplicate_addresses", [])
                    ],
                    "type": gtype_label,
                    "score": entry.get("score", 0),
                    "instructions": entry.get("instructions", []),
                }
            )
    return gadgets


def _gadget_text(g: dict) -> str:
    """Flat instruction text for a gadget (for pattern matching)."""
    return " ; ".join(
        (i.get("mnemonic", "") + " " + i.get("op_str", "")).strip()
        for i in g.get("instructions", [])
    ).lower()


def _matches_any_pattern(gadget_text: str, patterns: list) -> bool:
    """Return True if gadget_text matches at least one compiled regex pattern."""
    for pat in patterns:
        if isinstance(pat, str):
            if pat.lower() in gadget_text:
                return True
        else:  # pre-compiled re.Pattern
            if pat.search(gadget_text):
                return True
    return False


def _build_rows_for_binary(
    binary_path: str,
    arch: str,
    exploit_paths: list[str],
    lcsaj_json: Optional[dict] = None,
    depth: int = 20,
    darkness: int = 5,
    instructions_limit: int = 15,
    all_exec: bool = False,
    patterns: list = None,
    verbose: bool = True,
    lm=None,
    max_gadgets: Optional[int] = None,
) -> list[dict]:
    """
    Build one row per gadget for a single binary.

    Positive labels come from two complementary sources (OR-combined):
      1. exploit_paths — hex addresses / ELF symbols extracted from real exploit scripts
      2. patterns      — list of strings or re.Pattern objects; any gadget whose
                         instruction text matches is labelled positive.
                         Use this for ROP Emporium / challenge-specific canonical gadgets.

    Returns a list of dicts with keys = FEATURE_NAMES + ['label', 'binary', 'address', 'arch'].
    """
    if verbose:
        print(
            f"[dataset_builder] Processing {os.path.basename(binary_path)} ({arch})..."
        )

    # Get lcsajdump gadgets
    if lcsaj_json is None:
        lcsaj_json = _run_lcsajdump(
            binary_path,
            arch=arch,
            depth=depth,
            darkness=darkness,
            instructions=instructions_limit,
            all_exec=all_exec,
        )

    gadgets = _gadgets_from_json(lcsaj_json)
    if not gadgets:
        print(
            f"[dataset_builder] WARNING: no gadgets found for {binary_path}",
            file=sys.stderr,
        )
        return []

    # Collect all addresses lcsajdump found (primary + duplicates)
    all_lcsaj_addrs: set[int] = set()
    for g in gadgets:
        all_lcsaj_addrs.add(g["address"])
        all_lcsaj_addrs.update(g["duplicate_addresses"])

    # ── Source 1: exploit script address extraction ───────────────────────────
    used_addrs: set[int] = set()
    for exp_path in exploit_paths:
        if not os.path.exists(exp_path):
            continue
        found = extract_gadget_addresses(exp_path, binary_path)
        hits = found & all_lcsaj_addrs
        if verbose:
            print(
                f"  {os.path.basename(exp_path)}: "
                f"{len(found)} hex literals → {len(hits)} match lcsajdump gadgets"
            )
        used_addrs |= hits

    # ── Source 2: pattern-based labelling ────────────────────────────────────
    pattern_matched = 0
    compiled_patterns = patterns or []

    # Build rows
    # Build rows
    rows = []
    for g in gadgets:
        g_size = sum(i.get("size", 4) for i in g.get("instructions", [])) if g.get("instructions") else 15

        feats = extract_features(
            instructions=g["instructions"],
            arch=arch,
            gadget_type=g["type"],
            heuristic_score=g["score"],
            address=g["address"],
            gadget_pool=all_lcsaj_addrs,
            lm=lm,
            binary_path=binary_path,  
            gadget_size=g_size        
        )
        by_addr = g["address"] in used_addrs or any(
            a in used_addrs for a in g["duplicate_addresses"]
        )
        by_pattern = bool(
            compiled_patterns
            and _matches_any_pattern(_gadget_text(g), compiled_patterns)
        )
        label = 1 if (by_addr or by_pattern) else 0
        if by_pattern and not by_addr:
            pattern_matched += 1

        row = dict(feats)
        row["label"] = label
        # Use parent_dir/basename to disambiguate binaries with same name
        _parent = os.path.basename(os.path.dirname(binary_path))
        _base = os.path.basename(binary_path)
        row["binary"] = f"{_parent}/{_base}" if _parent else _base
        row["address"] = g["address"]
        row["arch"] = arch
        # Track provenance for preserving ground truth during sampling
        row["pattern_match"] = int(by_pattern and not by_addr)
        row["exploit_verified"] = int(by_addr)
        rows.append(row)

    # Optional cap: stratified sample to limit group size, targeting ~40% positive.
    # This ensures LambdaRank sees a balanced positive/negative mix even in large groups.
    # PRESERVE: All pattern-matched positive samples (exploit gadgets) are kept.
    if max_gadgets is not None and len(rows) > max_gadgets:
        import random

        pos_rows = [r for r in rows if r["label"] == 1]
        neg_rows = [r for r in rows if r["label"] == 0]
        rng = random.Random(42)

        # Target: 40% positive, 60% negative (or actual ratio if very few of either)
        n_pos_target = max(1, min(len(pos_rows), int(max_gadgets * 0.40)))
        n_neg_target = max_gadgets - n_pos_target
        n_neg_target = max(1, min(len(neg_rows), n_neg_target))

        # For positive rows, prioritize exploit-verified gadgets (by_addr) over pattern matches
        # We MUST preserve all exploit-address gadgets as they are the ground truth
        exploit_rows = [r for r in pos_rows if r.get("exploit_verified", 0)]
        pattern_rows = [r for r in pos_rows if not r.get("exploit_verified", 0)]

        # Always keep exploit-verified; fill remaining slots with best pattern matches
        n_exploit = len(exploit_rows)
        n_pattern_target = max(0, n_pos_target - n_exploit)

        if len(pattern_rows) > n_pattern_target:
            pattern_rows.sort(key=lambda r: -r.get("heuristic_score", 0))
            pattern_rows = pattern_rows[:n_pattern_target]

        pos_rows = exploit_rows + pattern_rows

        neg_rows = (
            rng.sample(neg_rows, n_neg_target)
            if len(neg_rows) > n_neg_target
            else neg_rows
        )
        rows = pos_rows + neg_rows
        rng.shuffle(rows)

    # Compute binary-level context feature: majority_term_is_ret.
    # 1 if >50% of this binary's gadgets are ret-terminated; 0 for JOP/COP binaries.
    # Gives the model context to distinguish ROP binaries from JOP-only ones.
    if rows:
        ret_count = sum(1 for r in rows if r.get("is_ret_terminated", 0))
        majority_ret = 1 if ret_count / len(rows) > 0.5 else 0
        for r in rows:
            r["majority_term_is_ret"] = majority_ret

    pos = sum(r["label"] for r in rows)
    if verbose:
        src = f" ({pattern_matched} from patterns)" if pattern_matched else ""
        print(
            f"  → {len(rows)} gadgets, {pos} positive ({pos / max(len(rows), 1) * 100:.1f}%){src}"
        )

    return rows


# ── Public API ────────────────────────────────────────────────────────────────


def build_dataset(
    samples: list[dict],
    verbose: bool = True,
    lm=None,
) -> tuple:
    """
    Build the full dataset from a list of sample descriptors.

    Parameters
    ----------
    samples : list of dict, each with keys:
        'binary'   : str — path to the ELF binary
        'arch'     : str — 'x86_64', 'arm64', or 'riscv64'
        'exploits' : list[str] — paths to exploit scripts
        'depth'    : int (optional) — lcsajdump depth, default 20
        'darkness' : int (optional) — lcsajdump darkness, default 5
        'json_cache': str (optional) — path to pre-computed lcsajdump JSON
    lm : InstructionLM, optional
        If provided, appends LM embedding features (lm_emb_0…15) to each row.

    Returns
    -------
    X : list[dict]  — feature rows (use pd.DataFrame(X) for training)
    y : list[int]   — binary labels (1 = used in exploit, 0 = not used)
    groups : list[int]  — group sizes for LambdaRank (one per binary)
    meta : list[dict]   — {binary, arch, address} per row (for diagnostics)
    """
    all_rows: list[dict] = []
    groups: list[int] = []

    for sample in samples:
        binary = sample["binary"]
        arch = sample["arch"]
        exploits = sample.get("exploits", [])

        # Use per-arch defaults from config.py if not overridden in sample
        _arch_sp = (
            ARCH_PROFILES.get(arch, {}).get("search_params", {})
            if ARCH_PROFILES
            else {}
        )
        depth = sample.get("depth", _arch_sp.get("d", 20))
        darkness = sample.get("darkness", _arch_sp.get("darkness", 5))
        instr = sample.get("instructions", _arch_sp.get("i", 100))

        json_cache = sample.get("json_cache")
        all_exec = sample.get("all_exec", False)
        lcsaj_json = None
        if json_cache and os.path.exists(json_cache):
            with open(json_cache) as f:
                lcsaj_json = json.load(f)
            if verbose:
                print(f"[dataset_builder] Using cached JSON: {json_cache}")

        rows = _build_rows_for_binary(
            binary_path=binary,
            arch=arch,
            exploit_paths=exploits,
            lcsaj_json=lcsaj_json,
            depth=depth,
            darkness=darkness,
            instructions_limit=instr,
            all_exec=all_exec,
            patterns=sample.get("patterns"),
            verbose=verbose,
            lm=lm,
            max_gadgets=sample.get("max_gadgets"),
        )
        if rows:
            all_rows.extend(rows)
            groups.append(len(rows))

    feat_names = FEATURE_NAMES_WITH_LM if lm is not None else FEATURE_NAMES
    X = [{k: r[k] for k in feat_names} for r in all_rows]
    y = [r["label"] for r in all_rows]
    meta = [
        {
            "binary": r["binary"],
            "arch": r["arch"],
            "address": r["address"],
            "binary_id": r["arch"] + "::" + r["binary"],
        }
        for r in all_rows
    ]

    if verbose:
        total_pos = sum(y)
        print(
            f"\n[dataset_builder] Total: {len(X)} samples, "
            f"{total_pos} positive ({total_pos / max(len(X), 1) * 100:.1f}%), "
            f"{len(groups)} groups  [features={len(feat_names)}]"
        )

    return X, y, groups, meta


def save_csv(X: list[dict], y: list[int], meta: list[dict], path: str):
    """Write the dataset to a CSV file for inspection."""
    import csv

    rows = []
    for feat, label, m in zip(X, y, meta):
        row = {
            "binary_id": m["binary_id"],
            "binary": m["binary"],
            "arch": m["arch"],
            "address": hex(m["address"]),
            "label": label,
            # Preserve provenance fields if present
            "exploit_verified": feat.get("exploit_verified", 0),
            "pattern_match": feat.get("pattern_match", 0),
        }
        row.update(feat)
        rows.append(row)

    # Infer feature names from data (supports both base and +LM variants)
    feat_keys = list(X[0].keys()) if X else FEATURE_NAMES
    fieldnames = [
        "binary_id",
        "binary",
        "arch",
        "address",
        "label",
        "exploit_verified",
        "pattern_match",
    ] + feat_keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[dataset_builder] Saved {len(rows)} rows to {path}")


# ── Thesis sample configuration ───────────────────────────────────────────────
# Edit paths to match your local setup.

_THESIS = Path("/home/chris/thesis_ctfs/src_ctf")

# ARM64 canonical gadget patterns: gadgets a human would select in an ARM64 exploit
_ARM64_PATTERNS = [
    # Epilogue gadgets: load callee-saved regs + x29/x30 from stack then ret
    "ldp x19, x20, [sp",
    "ldp x21, x22, [sp",
    "ldp x23, x24, [sp",
    "ldp x25, x26, [sp",
    "ldp x29, x30, [sp",  # frame pointer + link register restore (paired)
    "ldr x30, [sp",  # unpaired x30 (lr) restore — also a valid epilogue gadget
    "ldr x29, [sp",  # unpaired x29 (fp) restore
    # Stack argument loads
    "ldr x0, [sp",
    "ldr x1, [sp",
    "ldr x19, [sp",
    # JOP trampolines via register
    "blr x0",
    "blr x1",
    "blr x2",
    "blr x3",
    "blr x19",
    "blr x20",
    "blr x21",
    "blr x22",
    "br x0",
    "br x1",
    "br x2",
    "br x3",
    # Indirect pointer loads (function pointer dispatch)
    "ldr x3, [x20",
    "ldr x0, [x19",
    # Syscall
    "svc #0",
]

# RISC-V canonical gadget patterns.
# Kept precise to avoid false positives in large static binaries.
# Removed 'addi sp, sp,' (matches every stack adjustment — too broad).
# Removed 'mv a0,' (matches any mv into a0 — too broad).
import re as _re

_RISCV_PATTERNS = [
    # Return-address load from stack (canonical epilogue)
    # Support both decimal (40) and hex (0x28) offsets
    _re.compile(r"\bld\s+ra,\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    _re.compile(r"\blw\s+ra,\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    # Compressed stack loads (c.ldsp, c.lwsp) - common in exploit gadgets
    _re.compile(r"\bc\.ldsp\s+ra,\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    _re.compile(r"\bc\.lwsp\s+ra,\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    _re.compile(r"\bc\.ldsp\s+a[0-7],\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    _re.compile(r"\bc\.lwsp\s+a[0-7],\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    # Argument register loads from stack (ROP frame)
    _re.compile(r"\bld\s+a[0-7],\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    _re.compile(r"\blw\s+a[0-7],\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    # Load syscall number from stack
    _re.compile(r"\bld\s+a7,\s*(?:0x)?[0-9a-f]+\(sp\)", _re.IGNORECASE),
    # Trampolines (specific: register jump)
    "jr ra",
    "jalr zero, 0(ra)",  # canonical ret form
    _re.compile(r"\bjalr\s+zero,\s*\d+\(ra\)"),
    # Syscall
    "ecall",
    # Immediate loads for syscall setup (precise: li aX, small_number)
    _re.compile(r"\bli\s+a7,\s*\d+"),
    _re.compile(r"\bli\s+a0,\s*\d+"),
]

# Common x86_64 ROP patterns for thesis binaries that only have 1-2 exploit addresses.
# Labels the standard pop-gadgets that any x86_64 ROP chain needs.
_X86_COMMON_THESIS = [
    "pop rdi",
    "pop rsi",
    "pop rdx",
    "pop rax",
    "pop rbx",
    "pop rcx",
    "syscall",
]

# JOP gadget patterns for ctf86/jop challenge (Jump-Oriented Programming).
# All useful gadgets terminate with indirect jmp via register/memory — not ret.
_JOP_PATTERNS = [
    "jmp qword ptr [rdi",
    "jmp qword ptr [rcx",
    "jmp qword ptr [rdx",
    "jmp qword ptr [rax",
    "jmp qword ptr [rsp",
    "jmp qword ptr [rbx",
    "jmp qword ptr [rsi",
]

# Full x86_64 JOP/COP pattern set — indirect jmp and call via register/memory.
# Defined here (before THESIS_SAMPLES) so ctf86/jop can reference it.
_JOP_X86_PATTERNS = [
    # Register-based jmp dispatchers (JOP)
    "jmp qword ptr [rdi",
    "jmp qword ptr [rcx",
    "jmp qword ptr [rdx",
    "jmp qword ptr [rax",
    "jmp qword ptr [rbx",
    "jmp qword ptr [rsp",
    "jmp qword ptr [rsi",
    "jmp qword ptr [r8",
    "jmp qword ptr [r9",
    "jmp qword ptr [r10",
    "jmp qword ptr [r11",
    # Scaled-index jmp
    "jmp qword ptr [rdi + rax*8",
    "jmp qword ptr [rdx + rcx*8",
    "jmp qword ptr [rdi + rsi*8",
    "jmp qword ptr [rax + rcx*8",
    # Register-based call dispatchers (COP)
    "call qword ptr [rdi",
    "call qword ptr [rsi",
    "call qword ptr [rbx",
    "call qword ptr [rcx",
    "call qword ptr [r12",
    "call qword ptr [r13",
    "call qword ptr [rax",
    "call qword ptr [r8",
    "call qword ptr [r9",
    # Scaled-index call (ret2csu + variants)
    "call qword ptr [r12 + rbx*8",
    "call qword ptr [rbx + rbp*8",
    "call qword ptr [rdi + rax*8",
    "call qword ptr [rsi + rbx*8",
    "call qword ptr [rax + rbx*8",
    "call qword ptr [rax + rcx*8",
]

ONLINE_CTF_SAMPLES = [
    # ARM64: DEF CON 2021 mra — aarch64 static stripped, ROP via syscall gadgets
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/mra/mra",
        "arch": "arm64",
        "exploits": [
            "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/mra/exploit.py"
        ],
        "patterns": _ARM64_PATTERNS,
    },
    # ARM64: Shanghai CTF 2018 baby_arm — aarch64 dynamic, ret2csu exploit
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/baby_arm/pwn",
        "arch": "arm64",
        "exploits": [
            "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/baby_arm/solve.py"
        ],
        "patterns": _ARM64_PATTERNS,
    },
    # ARM64: easy-linux-pwn 02 — overwrite ret, direct ret2win
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/easy_pwn/02-overwrite-ret",
        "arch": "arm64",
        "exploits": [
            "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/easy_pwn/02-overwrite-ret.py"
        ],
        "patterns": _ARM64_PATTERNS,
    },
    # ARM64: easy-linux-pwn 06 — system() ROP (libc ldp gadget chain)
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/easy_pwn/06-system-rop",
        "arch": "arm64",
        "exploits": [
            "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/easy_pwn/06-system-rop.py"
        ],
        "patterns": _ARM64_PATTERNS,
    },
    # ARM64: easy-linux-pwn 07 — execve ROP (ldp/mov/svc chain)
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/easy_pwn/07-execve-rop",
        "arch": "arm64",
        "exploits": [
            "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/arm64_extra/easy_pwn/07-execve-rop.py"
        ],
        "patterns": _ARM64_PATTERNS,
    },
]

# ── ROP Emporium x86_64 samples ──────────────────────────────────────────────
# Downloaded from ropemporium.com.  No exploit scripts needed — canonical
# "useful gadgets" for each challenge are identified via patterns.
# A pattern string is a case-insensitive substring match against the full
# gadget instruction text (mnem op ; mnem op ; ...).

_ROPE = Path("/home/chris/Desktop/lcsajdump-debug/rop_emporium_bins")

# Patterns shared across all x86_64 ROP Emporium challenges
_X86_COMMON = [
    "pop rdi",  # universal argument setter
    "pop rsi",
    "pop rdx",
    "pop rax",
    "pop rbx ; pop rbp",  # double-pop useful for register setup
]

# Stack-pivot specific patterns (xchg/leave/mov rsp — fixes NDCG@1=0 on pivot group)
_X86_PIVOT = [
    "xchg rax, rsp",
    "xchg rsp, rax",
    "xchg rbx, rsp",
    "xchg rsp, rbx",
    "mov rsp, rax",
    "mov rsp, rbx",
    "mov rsp, rcx",
    "mov rsp, rdx",
    "leave",
    "pop rsp",
    "add rsp,",
    "sub rsp,",
]

# ── ROP Emporium x86_32 samples ──────────────────────────────────────────────

_X86_32_COMMON = [
    "pop eax",
    "pop ebx",
    "pop ecx",
    "pop edx",
    "pop ebp",
    "pop edi",
    "int 0x80",
    "ret",
]

_X86_32_PIVOT = [
    "xchg eax, esp",
    "xchg esp, eax",
    "xchg ebx, esp",
    "xchg esp, ebx",
    "mov esp, eax",
    "mov esp, ebx",
    "mov esp, ecx",
    "mov esp, edx",
    "leave",
    "pop esp",
    "add esp,",
    "sub esp,",
]

ROP_EMPORIUM_SAMPLES = [
    # all_exec=True: forces lcsajdump to scan full text section, finding usefulGadgets.
    # Exploit scripts use only addresses that lcsajdump's LCSAJ scan actually finds.
    {
        "binary": str(_ROPE / "ret2win/ret2win"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "ret2win/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON + ["ret2win"],
    },
    {
        "binary": str(_ROPE / "split/split"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "split/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON,
    },
    {
        "binary": str(_ROPE / "callme/callme"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "callme/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON + ["pop rdi ; pop rsi ; pop rdx"],
    },
    {
        "binary": str(_ROPE / "write4/write4"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "write4/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON + ["pop r14 ; pop r15", "mov qword ptr [r14], r15"],
    },
    {
        "binary": str(_ROPE / "badchars/badchars"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "badchars/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON + ["pop r12 ; pop r13", "mov qword ptr [r13], r12", "xor byte ptr [r15], r14b"],
    },
    {
        "binary": str(_ROPE / "fluff/fluff"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "fluff/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON + ["xlatb", "bextr", "stosb", "pop rdx ; pop rcx"],
    },
    {
        "binary": str(_ROPE / "pivot/pivot"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "pivot/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON + _X86_PIVOT + ["pop rax", "xchg rsp, rax", "mov rax, qword ptr [rax]", "add rax, rbp"],
    },
    {
        "binary": str(_ROPE / "ret2csu/ret2csu"),
        "arch": "x86_64",
        "exploits": [str(_ROPE / "ret2csu/solve.py")],
        "all_exec": True,
        "patterns": _X86_COMMON + [
            "pop rbx ; pop rbp ; pop r12 ; pop r13 ; pop r14 ; pop r15",
            "mov rdx, r15",
            "mov rsi, r14",
            "call qword ptr [r12",
        ],
    },
    # ── ROP Emporium x86_32 ───────────────────────────────────────────────────
    {
        "binary": str(_ROPE / "ret2win32/ret2win32"),
        "arch": "x86_32",
        "exploits": [str(_ROPE / "ret2win32/solve.py")],
        "all_exec": True,
        "patterns": _X86_32_COMMON + ["ret2win"],
    },
    {
        "binary": str(_ROPE / "split32/split32"),
        "arch": "x86_32",
        "exploits": [str(_ROPE / "split32/solve.py")],
        "all_exec": True,
        "patterns": _X86_32_COMMON,
    },
    {
        "binary": str(_ROPE / "callme32/callme32"),
        "arch": "x86_32",
        "exploits": [str(_ROPE / "callme32/solve.py")],
        "all_exec": True,
        "patterns": _X86_32_COMMON + ["pop ebx ; pop esi ; pop edi ; pop ebp"],
    },
    {
        "binary": str(_ROPE / "write432/write432"),
        "arch": "x86_32",
        "exploits": [str(_ROPE / "write432/solve.py")],
        "all_exec": True,
        "patterns": _X86_32_COMMON + ["mov dword ptr [edi], ebp", "pop edi ; pop ebp"],
    },
    {
        "binary": str(_ROPE / "badchars32/badchars32"),
        "arch": "x86_32",
        "exploits": [str(_ROPE / "badchars32/solve.py")],
        "all_exec": True,
        "patterns": _X86_32_COMMON + ["mov dword ptr [edi], esi", "xor byte ptr [ebp", "pop ebx ; pop esi ; pop edi ; pop ebp"],
    },
    {
        "binary": str(_ROPE / "fluff32/fluff32"),
        "arch": "x86_32",
        "exploits": [str(_ROPE / "fluff32/solve.py")],
        "all_exec": True,
        "patterns": _X86_32_COMMON + ["pext", "xchg byte ptr [ecx], dl", "bswap ecx", "pop ecx"],
    },
    {
        "binary": str(_ROPE / "pivot32/pivot32"),
        "arch": "x86_32",
        "exploits": [str(_ROPE / "pivot32/solve.py")],
        "all_exec": True,
        "patterns": _X86_32_COMMON + _X86_32_PIVOT + ["pop eax", "xchg esp, eax", "mov eax, dword ptr [eax]", "add eax, ebx"],
    },
]

# ── CTFtime-sourced samples ───────────────────────────────────────────────────
# LACTF 2024 (UCLA CTF) and DiceCTF 2024 — x86_64 challenges with solve scripts.
# Source: github.com/uclaacm/lactf-archive and github.com/dicegang/dicectf-quals-2024-challenges
# Stored permanently in /home/chris/Desktop/lcsajdump-debug/ctf_binaries/

_CTFTIME = Path("/home/chris/Desktop/lcsajdump-debug/ctf_binaries")

# Shared x86_64 patterns for heap/stack pwn challenges
_X86_HEAP_COMMON = [
    "pop rdi",
    "pop rsi",
    "pop rdx",
    "pop rax",
    "pop rbp",
    "ret",
    "syscall",
    "int 0x80",
]

# PIE binaries removed (ppplot, lamp, unsafe) — runtime-computed addrs, no static GT.
# Kept: eepy (static), minceraft (non-PIE), state-change (non-PIE).
CTFTIME_SAMPLES = [
    # LACTF 2025: eepy — statically linked, rich gadget set, ROP with pop rdi
    {
        "binary": str(_CTFTIME / "lactf2025/eepy/vuln"),
        "arch": "x86_64",
        "exploits": [str(_CTFTIME / "lactf2025/eepy/solve.py")],
        "all_exec": True,
        "patterns": _X86_HEAP_COMMON + _X86_COMMON + _X86_PIVOT,
    },
    # LACTF 2025: minceraft — pwntools ROP chain (ret2system), non-PIE
    {
        "binary": str(_CTFTIME / "lactf2025/minceraft/chall"),
        "arch": "x86_64",
        "exploits": [str(_CTFTIME / "lactf2025/minceraft/solve.py")],
        "all_exec": True,
        "patterns": _X86_HEAP_COMMON + _X86_COMMON,
    },
    # LACTF 2025: state-change — rbp pivot / ret2win, non-PIE
    {
        "binary": str(_CTFTIME / "lactf2025/state-change/chall"),
        "arch": "x86_64",
        "exploits": [str(_CTFTIME / "lactf2025/state-change/solve.py")],
        "all_exec": True,
        "patterns": _X86_HEAP_COMMON + _X86_COMMON + _X86_PIVOT,
    },
]

# ── JOP / COP challenge samples ──────────────────────────────────────────────
# Custom CTF binaries compiled to contain JOP (jmp [reg]) and COP (call [reg])
# gadgets.  Each binary has a stack/heap overflow and a function-pointer
# dispatcher, mirroring real-world JOP/COP CTF challenges.
#
# Source: ctf_binaries/pivot_jop_cop/  (see Makefile there to recompile)
# Static-linked ⇒ ~14k+ gadgets per binary.  max_gadgets caps the group so
# training remains balanced.

_JOP_COP_DIR = Path(
    "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/pivot_jop_cop/bins"
)

# ARM64 JOP/BLR patterns — adds x8/x9/x16/x17 variants to the base set
_ARM64_JOP_PATTERNS = _ARM64_PATTERNS + [
    "blr x8",
    "blr x9",
    "blr x10",
    "blr x11",
    "blr x16",
    "blr x17",  # PLT-temp registers, common in trampolines
    "br x8",
    "br x9",
    "br x16",
    "br x17",
    "ldr x9, [x0]",  # load function ptr then branch
    "ldr x9, [x1]",
    "ldr x16, [x0]",
    "ldr x17, [x0]",
]

# RISC-V JOP patterns — jr/c.jr to non-ra registers and jalr via arg-regs
_RISCV_JOP_PATTERNS = list(_RISCV_PATTERNS) + [
    "jr a0",
    "jr a1",
    "jr a2",
    "jr a3",
    "jr t0",
    "jr t1",
    "jr t2",
    "jr s1",
    "jr s2",
    "c.jr a",  # matches c.jr a0…a7 but NOT c.jr ra
    "jalr zero, 0(a0)",
    "jalr zero, 0(a1)",
    "jalr zero, 0(a2)",
    "jalr zero, 0(a3)",
    "jalr ra, 0(a0)",  # COP: call via a0
    "jalr ra, 0(a1)",
    "jalr ra, 0(t0)",
    "jalr ra, 0(t1)",
]

JOP_COP_SAMPLES = [
    # x86_64 JOP: dispatch-table challenge (jmp [rdi], jmp [rdx], etc.)
    {
        "binary": str(_JOP_COP_DIR / "x86_64/jop_dispatch"),
        "arch": "x86_64",
        "exploits": [],
        "patterns": _JOP_X86_PATTERNS + _X86_COMMON,
        "max_gadgets": 600,
    },
    # x86_64 JOP: vtable-corruption challenge (jmp [rax+off], jmp [rdi+rsi*8])
    {
        "binary": str(_JOP_COP_DIR / "x86_64/jop_vtable"),
        "arch": "x86_64",
        "exploits": [],
        "patterns": _JOP_X86_PATTERNS + _X86_COMMON,
        "max_gadgets": 600,
    },
    # x86_64 COP: callback-corruption challenge (call [rsi], call [rdi+off])
    {
        "binary": str(_JOP_COP_DIR / "x86_64/cop_callback"),
        "arch": "x86_64",
        "exploits": [],
        "patterns": _JOP_X86_PATTERNS + _X86_COMMON,
        "max_gadgets": 600,
    },
    # x86_64 COP: ret2csu-style gadget chain (call [r12+rbx*8], pop-heavy epilogue)
    {
        "binary": str(_JOP_COP_DIR / "x86_64/cop_ret2csu_style"),
        "arch": "x86_64",
        "exploits": [],
        "patterns": _JOP_X86_PATTERNS + _X86_COMMON,
        "max_gadgets": 600,
    },
    # ARM64 JOP/BLR: blr xN and br xN dispatchers
    {
        "binary": str(_JOP_COP_DIR / "arm64/jop_arm64"),
        "arch": "arm64",
        "exploits": [],
        "all_exec": True,
        "patterns": _ARM64_JOP_PATTERNS,
        "max_gadgets": 600,
    },
    # RISC-V JOP: jr a0, c.jr a*, jalr zero,0(aN) dispatchers
    {
        "binary": str(_JOP_COP_DIR / "riscv64/jop_riscv"),
        "arch": "riscv64",
        "exploits": [],
        "all_exec": True,
        "patterns": _RISCV_JOP_PATTERNS,
        "max_gadgets": 600,
    },
]

# ── testCTFs samples ───────────────────────────────────────────────────
# Real RISC-V CTF challenges from testCTFs directory

TESTCTF_SAMPLES = [
    # RISC-V: testCTFs/rop/vuln (static, exploit uses 0x4618c gadget + win)
    # Uses generic RISC-V patterns + JOP (jr/jalr) so trampoline gadgets ranked
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/testCTFs/rop/vuln",
        "arch": "riscv64",
        "exploits": ["/home/chris/Desktop/lcsajdump-debug/testCTFs/rop/exploit.py"],
        "patterns": _RISCV_JOP_PATTERNS,
        "max_gadgets": 400,
    },
    # RISC-V: testCTFs/rop2/vuln2 (static, trampoline exploit)
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/testCTFs/rop2/vuln2",
        "arch": "riscv64",
        "exploits": ["/home/chris/Desktop/lcsajdump-debug/testCTFs/rop2/exploit.py"],
        "patterns": _RISCV_JOP_PATTERNS,
        "max_gadgets": 400,
    },
    # NOTE: onlineRop/onlineVuln removed - uses direct function address (test_empty2)
    # not a proper ROP gadget challenge, so excluded from dataset
]

REAL_JOP_COP_SAMPLES = [
    # x86_64: PWNDAY #01 Juujuu (JOP via `jmp [rsp-8]` dispatcher obfuscated)
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/dplastico/pwnday01/juujuu/juujuu",
        "arch": "x86_64",
        "exploits": [
            "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/dplastico/pwnday01/juujuu/xpl.py"
        ],
        "patterns": _JOP_X86_PATTERNS + _X86_COMMON,
        "max_gadgets": 600,
    },
    # ARM64: bkerler/exploit_me lvl 14 (JOP)
    {
        "binary": "/home/chris/Desktop/lcsajdump-debug/ctf_binaries/bkerler/exploit_me/bin/exploit64",
        "arch": "arm64",
        "exploits": [],
        "patterns": _ARM64_JOP_PATTERNS,
        "max_gadgets": 600,
    },
]

# v14: synthetic JOP_COP_SAMPLES removed — only real-world CTFs.
# Real JOP coverage retained via REAL_JOP_COP_SAMPLES (juujuu x86_64, bkerler arm64).
ALL_SAMPLES = (
    ONLINE_CTF_SAMPLES
    + ROP_EMPORIUM_SAMPLES
    + CTFTIME_SAMPLES
    + TESTCTF_SAMPLES
    + REAL_JOP_COP_SAMPLES
)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build lcsajdump gadget ML dataset")
    parser.add_argument(
        "--out",
        default="gadget_dataset.csv",
        help="Output CSV path (default: gadget_dataset.csv)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    X, y, groups, meta = build_dataset(ALL_SAMPLES, verbose=not args.quiet)
    save_csv(X, y, meta, args.out)
