"""
features.py — Architecture-aware feature extraction for gadget ML scoring.

Depends only on the stdlib + config.py (already a lcsajdump dep).
Can be imported inside rainbowBFS.py without pulling in ML packages.
"""

from __future__ import annotations

import re
import sys
import os

# ── Import arch profiles ──────────────────────────────────────────────────────


def _load_profiles():
    try:
        from lcsajdump.core.config import ARCH_PROFILES

        return ARCH_PROFILES
    except ImportError:
        pass
    try:
        # Running from the debug repo root
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from lcsajdump.core.config import ARCH_PROFILES

        return ARCH_PROFILES
    except ImportError:
        return {}


ARCH_PROFILES = _load_profiles()

# ── Feature names (fixed order for pandas / numpy) ───────────────────────────

FEATURE_NAMES = [
    # Size / complexity
    "insn_count",
    "clobber_count",
    "stack_slots",
    "frame_size_bytes",
    # Memory behaviour
    "has_stack_load",
    "has_mem_read",
    "has_mem_write",
    # Terminator type
    "is_ret_terminated",
    "is_indirect_jmp_terminated",  # JOP dispatcher (jmp reg/mem, blr xN, jr non-ra)
    "is_trampoline_term",
    "is_internal_call",
    # Direct-call chain features (the lcsajdump structural advantage: E8/bl imm)
    "has_direct_call",  # tail is a call/bl/jal to a literal address (not via register)
    "is_chained",  # tail target address is itself another gadget in the pool
    # Arch-relative register usage
    "hits_link_reg",
    "hits_arg_reg",
    "hits_frame_reg",
    # Stack pivot detection
    "is_pivot_gadget",
    # Gadget class
    "is_sequential",
    # Bad-byte proxy
    "addr_has_null",
    "addr_has_newline",
    # Existing heuristic (baseline feature — lets model learn residuals)
    "heuristic_score",
    # One-hot arch (categorical)
    "arch_x86_64",
    "arch_arm64",
    "arch_riscv64",
    "arch_x86_32",
    # Binary-level context: 1 if >50% of this binary's gadgets are ret-terminated.
    # Allows the model to distinguish JOP-only binaries (0) from ROP binaries (1).
    "majority_term_is_ret",
    # Semantic features
    "sm_controls_arg_reg",
    "sm_stack_pivot_size",
    "sm_writes_memory",
]


# Extended feature set with InstructionLM embeddings (lm_emb_0 … lm_emb_15).
# Built lazily so importing features.py never pulls in gensim.
def _get_feature_names_with_lm():
    from lcsajdump.ml.instruction_lm import LM_FEATURE_NAMES

    return FEATURE_NAMES + LM_FEATURE_NAMES


try:
    FEATURE_NAMES_WITH_LM = _get_feature_names_with_lm()
except Exception:
    FEATURE_NAMES_WITH_LM = FEATURE_NAMES  # fallback if gensim not installed

# ── Helpers ────────────────────────────────────────────────────────────────────

_STORE_MNEMS = frozenset(
    {
        # x86
        "mov",
        "movl",
        "movq",  # can be stores (dst=[mem]) — handled below
        "push",
        # ARM64
        "str",
        "stp",
        "stm",
        "stmia",
        "stmdb",
        # RISC-V
        "sw",
        "sd",
        "sh",
        "sb",
        "c.sw",
        "c.sd",
        "c.swsp",
        "c.sdsp",
        # x86 float
        "fst",
        "movnti",
    }
)

# Instructions that definitively write memory (not regs)
_DEFINITE_STORES = frozenset(
    {
        "str",
        "stp",
        "stm",
        "stmia",
        "stmdb",
        "sw",
        "sd",
        "sh",
        "sb",
        "c.sw",
        "c.sd",
        "c.swsp",
        "c.sdsp",
        "fst",
        "movnti",
        "push",
    }
)

_MEM_READ_MNEMS = frozenset(
    {
        # ARM64
        "ldr",
        "ldur",
        "ldrh",
        "ldrb",
        "ldrsw",
        "ldp",
        # RISC-V
        "ld",
        "lw",
        "lh",
        "lb",
        "lhu",
        "lbu",
        "lwu",
        "c.ld",
        "c.lw",
        # x86
        "pop",  # implicit load from [rsp]
    }
)

_REG_RE = re.compile(r"^([a-z][a-z0-9]{0,4})$", re.IGNORECASE)

# Direct-call detection: operand is a literal address, not a register.
# Matches: 0x401163, 401163, #0x401163, #401163 (ARM imm prefix)
_DIRECT_ADDR_RE = re.compile(r"^#?0x[0-9a-fA-F]+$|^#?\d+$")

# Mnemonics that can be direct calls (as opposed to indirect register branches)
_DIRECT_CALL_MNEMS = {
    "x86_64": frozenset({"call"}),  # call rel32 (E8)
    "x86_32": frozenset({"call"}),  # call rel32 (E8) — same on 32-bit
    "arm64": frozenset({"bl"}),  # bl imm26 (0x94xxxxxx)
    "riscv64": frozenset({"jal"}),  # jal ra, offset (but not jalr)
}

# Matches [sp, #imm] or [sp] or imm(sp) for all three ISAs
_SP_LOAD_RE = re.compile(
    r"\bsp\b",
    re.IGNORECASE,
)

# LDP pattern: ldp r1, r2, [sp, #off] / [sp], #off
_LDP_SP_RE = re.compile(
    r"^(\w+),\s*(\w+),\s*\[sp(?:,\s*#?(-?\w+))?\](?:,\s*#?(-?\w+))?$", re.IGNORECASE
)

# LDR from SP pattern
_LDR_SP_RE = re.compile(
    r"^(\w+),\s*\[sp(?:,\s*#?(-?\w+))?\](?:,\s*#?(-?\w+))?$", re.IGNORECASE
)

# RISC-V: ld/lw reg, imm(sp)
_RISCV_SP_RE = re.compile(r"^(\w+),\s*(-?\w+)\(sp\)", re.IGNORECASE)

# RISC-V compressed stack-pointer loads: c.ldsp rd, imm(sp) / c.lwsp rd, imm(sp)
_RISCV_C_SP_RE = re.compile(r"c\.(?:ldsp|lwsp)\s+(\w+),\s*(-?\w+)\(sp\)", re.IGNORECASE)

# x86: pop reg
_POP_RE = re.compile(r"^\{?(\w+)", re.IGNORECASE)


def _parse_int(s):
    if s is None:
        return 0
    s = s.strip()
    try:
        return int(s, 16) if ("0x" in s or "0X" in s) else int(s)
    except ValueError:
        return 0


def _insn_dict(insn) -> dict:
    """Normalise capstone Instruction objects or plain dicts to {'mnemonic':..., 'op_str':...}."""
    if isinstance(insn, dict):
        return insn
    return {"mnemonic": insn.mnemonic, "op_str": insn.op_str}


def clobbered_registers(instructions: list) -> set:
    """Set of registers written by the instruction sequence (heuristic)."""
    written = set()
    for raw in instructions:
        insn = _insn_dict(raw)
        mnem = insn["mnemonic"].lower()
        if mnem in _DEFINITE_STORES:
            continue
        op_str = insn.get("op_str", "")
        if not op_str:
            continue
        # x86 store heuristic: mov [mem], reg / mov [mem], imm
        if mnem in ("mov", "movl", "movq") and op_str.lstrip().startswith("["):
            continue
        first = op_str.split(",")[0].strip().lstrip("*").lower()
        first = first.lstrip("[").split("[")[-1].strip()
        if _REG_RE.match(first):
            written.add(first)
    return written


def stack_frame_layout(instructions: list) -> tuple[int, int]:
    """Return (stack_slots, frame_size_bytes) from stack-load instructions."""
    slots = 0
    max_offset = 0
    sp_offset = 0  # x86 pop tracking

    for raw in instructions:
        insn = _insn_dict(raw)
        mnem = insn["mnemonic"].lower()
        op = insn.get("op_str", "")

        if mnem == "ldp":
            m = _LDP_SP_RE.match(op)
            if m:
                pre = _parse_int(m.group(3))
                post = _parse_int(m.group(4))
                slots += 2
                base = pre if pre != 0 else 0
                max_offset = max(max_offset, base + 8)
                if post:
                    max_offset = max(max_offset, post)

        elif mnem in ("ldr", "ldur"):
            m = _LDR_SP_RE.match(op)
            if m:
                slots += 1
                off = _parse_int(m.group(2))
                max_offset = max(max_offset, off + 8)

        elif mnem in ("ld", "lw", "lh", "lb", "lwu", "lhu", "lbu", "c.ld", "c.lw"):
            m = _RISCV_SP_RE.match(op)
            if m:
                slots += 1
                off = _parse_int(m.group(2))
                max_offset = max(max_offset, off + 8)

        # RISC-V compressed stack loads: c.ldsp rd, off(sp) / c.lwsp rd, off(sp)
        elif mnem in ("c.ldsp", "c.lwsp"):
            m = _RISCV_C_SP_RE.match(f"{mnem} {op}")
            if m:
                slots += 1
                off = _parse_int(m.group(2))
                max_offset = max(max_offset, off + 8)

        elif mnem == "pop":
            m = _POP_RE.match(op)
            if m:
                slots += 1
                sp_offset += 8
                max_offset = max(max_offset, sp_offset)

    return slots, max_offset


# ── Main entry point ──────────────────────────────────────────────────────────


def extract_features(
    instructions: list,
    arch: str,
    gadget_type: str = "Sequential",
    heuristic_score: int = 0,
    address: int = 0,
    gadget_pool: set = None,
    lm=None,
    majority_term_is_ret: int = 0,
    binary_path: str = None,
    gadget_size: int = 0,
) -> dict:
    """
    Extract a fixed-length feature vector from a gadget.

    Parameters
    ----------
    instructions : list
        List of instruction dicts {'mnemonic': ..., 'op_str': ...} OR
        capstone Instruction objects (both accepted).
    arch : str
        One of 'x86_64', 'x86_32', 'arm64', 'riscv64'.
    gadget_type : str
        'Sequential' or 'Jump-Based'.
    heuristic_score : int
        The score already assigned by rainbowBFS heuristic (used as a feature).
    address : int
        Primary address of the gadget (for bad-byte detection).
    gadget_pool : set of int, optional
        Set of all gadget addresses in the current binary.
        Required to compute is_chained (direct-call target is another gadget).
    """
    profile = ARCH_PROFILES.get(arch, {})
    insn_dicts = [_insn_dict(i) for i in instructions]

    insn_count = len(insn_dicts)
    last_mnem = insn_dicts[-1]["mnemonic"].lower() if insn_dicts else ""

    # ── Clobber analysis ────────────────────────────────────────────────────
    written = clobbered_registers(insn_dicts)

    # ── Stack frame layout ──────────────────────────────────────────────────
    slots, frame_size = stack_frame_layout(insn_dicts)

    # ── Memory behaviour ─────────────────────────────────────────────────────
    has_stack_load = any(
        _insn_dict(i)["mnemonic"].lower() in _MEM_READ_MNEMS
        and _SP_LOAD_RE.search(_insn_dict(i).get("op_str", ""))
        for i in instructions
    )
    has_mem_read = any(
        _insn_dict(i)["mnemonic"].lower() in _MEM_READ_MNEMS for i in instructions
    )
    has_mem_write = any(
        _insn_dict(i)["mnemonic"].lower() in _DEFINITE_STORES for i in instructions
    )

    # ── Terminator analysis ──────────────────────────────────────────────────
    ret_mnems = profile.get("ret_mnems", set())
    trampoline_mnems = profile.get("trampoline_mnems", set())
    call_mnems = profile.get("call_mnems", set())

    is_ret_term = last_mnem in ret_mnems
    is_trampoline_term = last_mnem in trampoline_mnems

    # RISC-V: refine is_ret_term — jalr/jr/c.jr/c.jalr to non-ra registers are
    # JOP dispatchers, not returns. Only jalr x0,0(ra) / c.jr ra / ret is a true ret.
    _riscv_jop = False
    if (
        arch == "riscv64"
        and insn_dicts
        and last_mnem in ("jalr", "jr", "c.jalr", "c.jr")
    ):
        last_op = insn_dicts[-1].get("op_str", "").strip().lower()
        # Canonical ret: "ret", "c.jr ra", "jr ra", "jalr zero, 0(ra)"
        # Non-ret: "jr a0", "c.jr a0", "jalr zero, 0(a0)", "jalr ra, 0(...)"
        _is_ra_target = (
            last_mnem == "ret"
            or last_op == "ra"  # c.jr ra, jr ra
            or "(ra)" in last_op  # jalr zero, 0(ra)
            or last_op.startswith("ra,")  # jalr ra, 0(...) - this is a call-via-reg
        )
        if not _is_ra_target:
            is_ret_term = False
            _riscv_jop = True

    # Indirect-jump terminator (JOP dispatcher):
    # - x86: jmp/call via register or memory (not a direct address)
    # - ARM64: blr xN or br xN (always register-based)
    # - RISC-V: jalr/jr/c.jalr/c.jr to non-ra register (detected above)
    is_indirect_jmp = 0
    if insn_dicts:
        last_op = insn_dicts[-1].get("op_str", "").strip().lower()
        if arch in ("x86_64", "x86_32"):
            if last_mnem in ("jmp", "call") and not _DIRECT_ADDR_RE.match(last_op):
                is_indirect_jmp = 1
        elif arch == "arm64":
            if last_mnem in ("blr", "br"):
                is_indirect_jmp = 1
        elif arch == "riscv64":
            if _riscv_jop:
                is_indirect_jmp = 1
    is_internal_call = any(
        _insn_dict(i)["mnemonic"].lower() in call_mnems for i in insn_dicts[:-1]
    )

    # ── Direct-call chain detection ───────────────────────────────────────────
    # has_direct_call: tail is call/bl/jal to a literal address (not via register).
    # This is the lcsajdump structural advantage — E8 rel32 on x86, bl imm26 on ARM64.
    direct_call_mnems = _DIRECT_CALL_MNEMS.get(arch, frozenset())
    has_direct_call = 0
    direct_call_target = None
    if insn_dicts and last_mnem in direct_call_mnems:
        op = insn_dicts[-1].get("op_str", "").strip()
        if _DIRECT_ADDR_RE.match(op):
            has_direct_call = 1
            try:
                op_clean = op.lstrip("#")
                direct_call_target = (
                    int(op_clean, 16) if "0x" in op_clean.lower() else int(op_clean)
                )
            except ValueError:
                pass

    # is_chained: direct call target is itself a known gadget in this binary.
    # Requires gadget_pool to be passed; falls back to 0 if not available.
    is_chained = 0
    if has_direct_call and direct_call_target is not None and gadget_pool:
        is_chained = int(direct_call_target in gadget_pool)

    # ── Arch-relative register hits ──────────────────────────────────────────
    link_reg = profile.get("link_reg", set())
    link_set = {link_reg} if isinstance(link_reg, str) else set(link_reg)
    primary_arg = profile.get("primary_arg_reg", "")
    frame_reg = profile.get("frame_reg", "")

    # For x86_64, link_reg includes 'rsp' which appears almost everywhere.
    # We check op_str text directly (same as rainbowBFS reg_in_op).
    all_op_strs = " ".join(d.get("op_str", "") for d in insn_dicts).lower()
    hits_link = bool(link_set and any(r.lower() in all_op_strs for r in link_set))
    hits_arg = bool(primary_arg and primary_arg.lower() in written)
    hits_frame = bool(frame_reg and frame_reg.lower() in written)

    # ── Pivot gadget detection ──────────────────────────────────────────────
    pivot_always = profile.get("pivot_always_mnems", frozenset())
    pivot_sp_mnems = profile.get("pivot_sp_mnems", frozenset())
    sp_reg = profile.get("stack_pointer_reg", "")
    is_pivot = any(
        d["mnemonic"].lower() in pivot_always
        or (
            d["mnemonic"].lower() in pivot_sp_mnems
            and sp_reg
            and sp_reg in d.get("op_str", "").lower()
        )
        for d in insn_dicts
    )

    # ── Address bad bytes ────────────────────────────────────────────────────
    if address > 0:
        byte_len = (address.bit_length() + 7) // 8
        addr_bytes = address.to_bytes(byte_len, "little")
        addr_has_null = int(b"\x00" in addr_bytes)
        addr_has_newline = int(b"\n" in addr_bytes)
    else:
        addr_has_null = addr_has_newline = 0

    # ── Semantic features ────────────────────────────────────────────────────
    sm_controls = 0
    sm_pivot = 0
    sm_writes = 0
    
    if binary_path and os.path.exists(binary_path):
        from lcsajdump.ml.semantic_features import extract_semantic_features
        sem_feats = extract_semantic_features(binary_path, address, gadget_size, arch)
        sm_controls = sem_feats.get("sm_controls_arg_reg", 0)
        sm_pivot = sem_feats.get("sm_stack_pivot_size", 0)
        sm_writes = sem_feats.get("sm_writes_memory", 0)
    
    # ── Assemble feature dict ────────────────────────────────────────────────
    feats = {
        "insn_count": insn_count,
        "clobber_count": len(written),
        "stack_slots": slots,
        "frame_size_bytes": frame_size,
        "has_stack_load": int(has_stack_load),
        "has_mem_read": int(has_mem_read),
        "has_mem_write": int(has_mem_write),
        "is_ret_terminated": int(is_ret_term),
        "is_indirect_jmp_terminated": is_indirect_jmp,
        "is_trampoline_term": int(is_trampoline_term),
        "is_internal_call": int(is_internal_call),
        "has_direct_call": has_direct_call,
        "is_chained": is_chained,
        "hits_link_reg": int(hits_link),
        "hits_arg_reg": int(hits_arg),
        "hits_frame_reg": int(hits_frame),
        "is_pivot_gadget": int(is_pivot),
        "is_sequential": int(gadget_type == "Sequential"),
        "addr_has_null": addr_has_null,
        "addr_has_newline": addr_has_newline,
        "heuristic_score": heuristic_score,
        "arch_x86_64": int(arch == "x86_64"),
        "arch_arm64": int(arch == "arm64"),
        "arch_riscv64": int(arch == "riscv64"),
        "arch_x86_32": int(arch == "x86_32"),
        "majority_term_is_ret": majority_term_is_ret,
        "sm_controls_arg_reg": sm_controls,
        "sm_stack_pivot_size": sm_pivot,
        "sm_writes_memory": sm_writes,
    }

    # Append LM embeddings if available
    if lm is not None:
        import numpy as np

        emb = lm.encode(
            " ".join(
                f"{d['mnemonic']} {d.get('op_str', '')}".strip() for d in insn_dicts
            )
        )
        for idx, val in enumerate(emb):
            feats[f"lm_emb_{idx}"] = float(val)

    return feats
