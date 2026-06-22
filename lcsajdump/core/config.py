import capstone

ARM64_ARCH = (
    capstone.CS_ARCH_AARCH64
    if hasattr(capstone, "CS_ARCH_AARCH64")
    else capstone.CS_ARCH_ARM64
)
RISCV64_MODE = (
    capstone.CS_MODE_RISCV_C
    if hasattr(capstone, "CS_MODE_RISCV_C")
    else capstone.CS_MODE_RISCVC
)

ARCH_PROFILES = {
    "riscv64": {
        "name": "RISC-V 64-bit",
        "cs_arch": capstone.CS_ARCH_RISCV,
        "cs_mode": capstone.CS_MODE_RISCV64 | RISCV64_MODE,
        "step": 2,
        "jump_mnems": {
            "j",
            "jal",
            "c.j",
            "c.jal",
            "jr",
            "jalr",
            "c.jr",
            "c.jalr",
            "ret",
            "ecall",
        },
        "unconditional_jumps": {
            "j",
            "jal",
            "c.j",
            "c.jal",
            "ret",
            "jr",
            "c.jr",
            "jalr",
            "c.jalr",
            "ecall",
        },
        "ret_mnems": {"ret", "c.jr", "jr", "jalr", "c.jalr", "ecall"},
        "syscall_mnems": frozenset({"ecall"}),
        "branch_prefixes": ("b", "c.b"),
        "call_mnems": {"jal", "c.jal", "jalr", "c.jalr"},
        "link_reg": "ra",
        "primary_arg_reg": "a0",
        "frame_reg": "s0",
        "trampoline_mnems": {"j", "c.j", "jal", "c.jal"},
        "pivot_always_mnems": frozenset(),
        "pivot_sp_mnems": frozenset({"mv", "c.mv", "addi", "c.addi"}),
        "stack_pointer_reg": "sp",
        # Optuna unified, dataset v5, 2 groups, 150 trials (NDCG@5 improved +0.3095)
        "scoring_weights": {
          "base_score": 100,
          "insn_penalty": 28,
          "bonus_link_reg": 20,
          "bonus_arg_reg": 135,
          "bonus_frame_reg": 95,
          "penalty_internal_call": 73,
          "bonus_trampoline": 58,
          "penalty_bad_ret": 146,
          "bonus_direct_call": 45,
          "bonus_pivot": 33,
          "bonus_syscall": 43,
          "clobber_penalty": 56,
          "long_chain_penalty": 8,
          "max_clobber": 4
        },
        "search_params": {
          "limit": 12,
          "darkness": 7,
          "d": 3,
          "i": 11,
          "m": 16
        },
    },
    "x86_64": {
        "name": "x86-64",
        "cs_arch": capstone.CS_ARCH_X86,
        "cs_mode": capstone.CS_MODE_64,
        "step": 1,
        "jump_mnems": {
            "jmp",
            "call",
            "ret",
            "retf",
            "iret",
            "syscall",
            "sysenter",
            "int",
        },
        "unconditional_jumps": {
            "jmp",
            "call",
            "ret",
            "retf",
            "iret",
            "syscall",
            "sysenter",
            "int",
        },
        "ret_mnems": {"ret", "retn", "retf", "iret", "syscall", "int", "sysenter"},
        "syscall_mnems": frozenset({"syscall", "sysenter", "int"}),
        "branch_prefixes": ("j", "loop"),
        "call_mnems": {"call"},
        "link_reg": {"rip", "rsp"},
        "primary_arg_reg": "rdi",
        "frame_reg": "rbp",
        "trampoline_mnems": {"jmp", "call"},
        "pivot_always_mnems": frozenset({"leave"}),
        "pivot_sp_mnems": frozenset(
            {"xchg", "pop"}
        ),  # Rimosso 'mov' per evitare falsi positivi su prologhi funzione
        "stack_pointer_reg": "rsp",
        # Optuna unified, dataset v5, 12 groups, 150 trials (NDCG@5 improved +0.6425)
        "scoring_weights": {
          "base_score": 100,
          "insn_penalty": 21,
          "bonus_link_reg": 48,
          "bonus_arg_reg": 60,
          "bonus_frame_reg": 62,
          "penalty_internal_call": 57,
          "bonus_trampoline": 11,
          "penalty_bad_ret": 105,
          "bonus_direct_call": 14,
          "bonus_pivot": 91,
          "bonus_syscall": 12,
          "clobber_penalty": 165,
          "long_chain_penalty": 32,
          "max_clobber": 2
        },
        "search_params": {
            "limit": 17,
            "darkness": 4,
            "d": 3,
            "i": 6,
            "m": 12
        },
    },
    "x86_32": {
        "name": "x86-32",
        "cs_arch": capstone.CS_ARCH_X86,
        "cs_mode": capstone.CS_MODE_32,
        "step": 1,
        "jump_mnems": {
            "jmp",
            "call",
            "ret",
            "retf",
            "iret",
            "int",
        },
        "unconditional_jumps": {
            "jmp",
            "call",
            "ret",
            "retf",
            "iret",
            "int",
        },
        "ret_mnems": {"ret", "retn", "retf", "iret", "int"},
        "syscall_mnems": frozenset({"int"}),  # int 0x80
        "branch_prefixes": ("j", "loop"),
        "call_mnems": {"call"},
        "link_reg": {"eip", "esp"},
        "primary_arg_reg": "eax",
        "frame_reg": "ebp",
        "trampoline_mnems": {"jmp", "call"},
        "pivot_always_mnems": frozenset({"leave"}),
        "pivot_sp_mnems": frozenset({"xchg", "pop"}),
        "stack_pointer_reg": "esp",
        # Optuna unified, dataset v5, 7 groups, 150 trials (NDCG@5 improved +0.5870)
        "scoring_weights": {
          "base_score": 100,
          "insn_penalty": 19,
          "bonus_link_reg": 91,
          "bonus_arg_reg": 36,
          "bonus_frame_reg": 90,
          "penalty_internal_call": 226,
          "bonus_trampoline": 15,
          "penalty_bad_ret": 24,
          "bonus_direct_call": 7,
          "bonus_pivot": 134,
          "bonus_syscall": 135,
          "clobber_penalty": 142,
          "long_chain_penalty": 50,
          "max_clobber": 3
        },
        "search_params": {
          "limit": 26,
          "darkness": 4,
          "d": 4,
          "i": 11,
          "m": 10
        },
    },
    "arm64": {
        "name": "ARM64 (AArch64)",
        "cs_arch": ARM64_ARCH,
        "cs_mode": capstone.CS_MODE_ARM,
        "step": 4,
        "jump_mnems": {"b", "bl", "br", "blr", "ret", "svc"},
        "unconditional_jumps": {"b", "bl", "br", "blr", "ret", "svc"},
        "ret_mnems": {"ret", "svc"},
        "syscall_mnems": frozenset({"svc"}),
        "branch_prefixes": ("b.", "cbz", "cbnz", "tbz", "tbnz"),
        "call_mnems": {"bl", "blr"},
        "link_reg": {"x30", "lr"},
        "primary_arg_reg": "x0",
        "frame_reg": "x29",
        "trampoline_mnems": {"b", "bl", "br", "blr"},
        "pivot_always_mnems": frozenset(),
        "pivot_sp_mnems": frozenset({"mov", "add", "sub"}),
        "stack_pointer_reg": "sp",
        # Optuna unified, dataset v5, 6 groups, 150 trials (NDCG@5 improved +0.4339)
        "scoring_weights": {
          "base_score": 100,
          "insn_penalty": 15,
          "bonus_link_reg": 80,
          "bonus_arg_reg": 116,
          "bonus_frame_reg": 12,
          "penalty_internal_call": 162,
          "bonus_trampoline": 94,
          "penalty_bad_ret": 196,
          "bonus_direct_call": 74,
          "bonus_pivot": 135,
          "bonus_syscall": 49,
          "clobber_penalty": 57,
          "long_chain_penalty": 12,
          "max_clobber": 4
        },
        "search_params": {
          "limit": 44,
          "darkness": 4,
          "d": 4,
          "i": 4,
          "m": 0
        },

    },
}
