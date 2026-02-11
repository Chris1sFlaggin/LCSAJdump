# 🔬 LCSAJdump — RISC-V ROP/JOP Gadget Finder

**The gadgets you can't see are the ones that get you.**

LCSAJdump discovers *non-contiguous* ROP/JOP gadgets in RISC-V binaries by reconstructing the control-flow graph through **LCSAJ** (Linear Code Sequence and Jump) analysis. While traditional scanners like ROPgadget or Ropper slide a linear window over raw bytes, LCSAJdump follows jumps and branches to uncover **Shadow Gadgets** — exploitable instruction chains that span multiple, non-adjacent code blocks and are invisible to sequential methods.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **RISC-V 64-bit + Compressed (C)** | Full support for 32-bit and 16-bit (compressed) instruction encoding — without the C extension you lose up to 50% of usable gadgets. |
| **LCSAJ Control-Flow Engine** | Segments `.text` into basic blocks at every jump, branch, and return, then rebuilds edges (fallthrough + direct targets) into a directed graph via NetworkX. |
| **Rainbow BFS Search** | Custom backward BFS from every `ret` / `jr` / `jalr` tail, walking the reverse graph to compose multi-block gadget chains. |
| **Score-Based Ranking** | Intelligent heuristic scoring: rewards `ra` / `a0` register loads and jump trampolines, penalizes length and non-`ra` indirect jumps. |
| **Gadget Classification** | Every gadget is tagged as **LINEAR** (single block), **TRAMPOLINE** (jump-based), **CONDITIONAL** (branch-based), or **FALLTHROUGH** — split into *Sequential* and *Jump-Based* categories for quick triage. |
| **Tunable Pruning** | The *Darkness* parameter caps per-node visits, keeping analysis tractable on large binaries without sacrificing coverage. |

---

## 🧠 Why LCSAJ?

Classic ROP scanners walk backward byte-by-byte from a `ret`, collecting only straight-line instruction sequences. If a *bad byte* (e.g. a null `\x00`) or a useless instruction sits in the middle, the gadget is discarded.

LCSAJ analysis changes the game:

```
Traditional (linear)          LCSAJ (graph-aware)
──────────────────            ──────────────────────────
│ insn A              │       │ Block 1: insn A         │
│ ❌ bad byte          │       │          j  Block 2 ──────┐
│ insn B              │       │                         │  │
│ ret                 │       │ Block 2: insn B ◄───────┘  │
└─────────────────────┘       │          ret              │
⚠ gadget broken               └──────────────────────────┘
                               ✅ shadow gadget found!
```

By following jumps and branches, LCSAJdump **bypasses bad bytes**, skips over useless instructions, and chains together blocks that a linear scanner would never connect. The result: a richer, more resilient gadget set for exploit development.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Chris1sFlaggin/LCSAJdump.git
cd LCSAJdump/LCSAJdump
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
```

### Usage

```bash
# Basic scan
python LCSAJdump.py <binary>

# Custom depth, pruning, and output limit
python LCSAJdump.py -d 15 -k 100 -l 20 -s 50 --verbose <binary>
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `-d, --depth` | 12 | Max search depth in LCSAJ blocks |
| `-k, --darkness` | 50 | Pruning threshold (max visits per node) |
| `-l, --limit` | 10 | Number of gadgets to display |
| `-s, --min-score` | 0 | Minimum score filter |
| `-v, --verbose` | off | Show detailed gadget disassembly |

Results are printed to the console and saved to `gadgets_found.txt`.

---

## ⚙️ Architecture

```
ELF Binary
    │
    ▼
┌────────────────────┐
│   BinaryLoader     │  pyelftools → .text extraction
│   (loader.py)      │  Capstone   → RV64GC disassembly
└────────┬───────────┘
         │  list[CsInsn]
         ▼
┌────────────────────┐
│   LCSAJGraph       │  Instruction stream → LCSAJ blocks
│   (graph.py)       │  Fallthrough + Jump edges → DiGraph
└────────┬───────────┘
         │  reverse_graph
         ▼
┌────────────────────┐
│   RainbowFinder    │  Backward BFS from ret/jr/jalr tails
│   (rainbowBFS.py)  │  Score, classify, rank gadgets
└────────────────────┘
```

---

## 📊 Performance

| Binary | Instructions | Gadgets Found | Time |
|---|---|---|---|
| RISC-V libc | ~300 000 | 14 000+ | < 6 s |

Benchmarked on a standard workstation. The *Darkness* pruning parameter is the primary knob for trading speed vs. coverage.

---

## 📄 Paper

For a detailed theoretical explanation of the LCSAJ-based approach, see the [research paper](https://github.com/Chris1sFlaggin/LCSAJdump/blob/main/PAPER.md).

---

## 📝 License

Released under the [MIT License](LICENSE). © 2026 Chris1sFlaggin.
