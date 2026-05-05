<div align="center">

<a href='https://chris1sflaggin.it/LCSAJdump'><img src="_images/LOGO.png" width="60%" alt="LCSAJdump Logo"/></a>

# [LCSAJdump](https://chris1sflaggin.it/LCSAJdump)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/lcsajdump?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/lcsajdump)

### *Universal Graph-Based Framework for Automated Gadget Discovery*

[![Status](https://img.shields.io/badge/status-Universal_Framework-orange?style=for-the-badge)](https://github.com/Chris1sFlaggin/LCSAJdump)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

<div align="center">
  <img src="_images/LCSAJdump1.gif" width="100%">
</div>
</br>

**LCSAJdump** is a static analysis framework designed to discover Return-Oriented Programming (ROP) and Jump-Oriented Programming (JOP) gadgets. Unlike traditional scanners, LCSAJdump is **architecture-agnostic** and employs a graph-based approach to uncover vulnerabilities invisible to common linear tools.

</div>

---

## Why LCSAJdump?

Common ROP scanners use a linear "sliding-window" approach over the binary's executable bytes. This method systematically fails to identify **Shadow Gadgets**: execution chains that traverse non-contiguous memory blocks connected by unconditional jumps or conditional branches.

LCSAJdump overcomes this limitation by reconstructing the **Control-Flow Graph (CFG)** through **LCSAJ (Linear Code Sequence and Jump)** analysis. By modeling the binary as a directed graph of basic blocks, the tool identifies:

1.  **Contiguous Gadgets:** Standard linear sequences terminating in a control-flow transfer.
2.  **Shadow Gadgets (Non-Contiguous):** Complex chains that bypass "bad bytes" (e.g., null bytes) by utilizing instructions that would otherwise be unreachable via linear scanning.

---

## Key Features

* **Multi-Architecture Support:** Native support for RISC-V (64GC), x86-64, and ARM64, easily extendable to other architectures via modular profiles.
* **Graph-Based Analysis:** Segments the `.text` section into LCSAJ basic blocks and reconstructs flow relationships using `NetworkX`.
* **Rainbow BFS Algorithm:** Proprietary backward Breadth-First Search starting from control-flow sinks. Now features an **O(1) Early-Drop Uniqueness Filter** and **Hard-Cap Instruction Limits** to prevent state explosion and ensure ultra-fast analysis even on dense CISC binaries.
* **Lazy Graph Build:** Graph construction retains only nodes reachable from gadget tails within `--depth` hops, drastically reducing memory and build time on large binaries (e.g., `libc`) while producing **identical results**.
* **Two-Stage Ranking Engine:** Combines a hyper-fast heuristic baseline (Bayesian-optimized via Optuna) with a deep-learning **LightGBM ML model** that refines gadget quality using structural and semantic features.
* **Zero-Overhead Inference:** The ML model is integrated natively and runs by default, processing tens of thousands of nodes in seconds. It acts as a highly effective filter, rejecting noisy jumps and returning clean, highly controllable gadget chains. Hosted on [Hugging Face](https://huggingface.co/chris1sflaggin/chainfinder_v4_hybrid).
* **Pruning Parameters:** Configurable "Darkness" factor to balance analysis depth and performance, preventing infinite loops in cyclic graphs.

---

## Supported Architectures

> (see [Benchmarks](https://chris1sflaggin.it/LCSAJdump#benchmarks)).

LCSAJdump is designed to be universal. Currently supported:

* **RISC-V 64-bit (RV64GC):** Full support for compressed 16-bit instructions.
* **x86-64:** Handles variable-length overlapping instructions. Safely navigates dense graphs without memory explosion.
* **ARM64:** Handles 32-bit instructions and deeply filters out bloated gadgets via strict heuristic penalties.
* **Other Architectures:** Can be easily implemented by defining new profiles in `config.py`.

---

## Installation

### Via Pip (Recommended)

```bash
pip install lcsajdump

```

### From Source (Development)

```bash
git clone [https://github.com/Chris1sFlaggin/LCSAJdump.git](https://github.com/Chris1sFlaggin/LCSAJdump.git)
cd LCSAJdump
pip install -r requirements.txt

```

---

## Usage

LCSAJdump offers a powerful CLI for precise binary analysis:

**Standard Analysis (Default RISC-V):**

```bash
python LCSAJdump.py <path_to_binary>

```

**Advanced Analysis (Specifying Architecture and Output File):**

```bash
lcsajdump -a riscv64 -d 15 -k 10 -l 20 -o gadgets.txt <path_to_binary>

```

**Export as JSON with bad-char filter:**

```bash
lcsajdump -a x86_64 -d 20 -k 5 -b "000a0d" --json -o gadgets.json <path_to_binary>
```

> **Note:** Use `-o` after `--json` to save JSON to file. Without `--json`, `-o` saves plain text.

**Save plain text output:**

```bash
lcsajdump -a riscv64 -d 15 -k 10 -l 20 -o gadgets.txt <path_to_binary>
```

**Analyze all executable sections:**

```bash
lcsajdump --all-exec -d 25 -k 10 -l 30 <path_to_binary>

```

**Force strictly algorithmic ranking (bypass ML):**

```bash
lcsajdump --algo <path_to_binary>

```

### CLI Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `-a, --arch` | TEXT | `auto` | Target architecture (`auto`, `riscv64`, `x86_64`, `arm64`). Auto-detected from ELF header. |
| `-d, --depth` | INTEGER | `20` | Max search depth in LCSAJ blocks. Controls chain length. |
| `-k, --darkness` | INTEGER | `5` | Pruning threshold — max visits per node. Higher = more gadgets, slower scan. |
| `-l, --limit` | INTEGER | `10` | Max number of gadgets to display in the output. |
| `-s, --min-score` | INTEGER | `0` | Minimum heuristic score for a gadget to appear in results. |
| `-i, --instructions` | INTEGER | `15` | Max number of instructions contained in a single LCSAJ node. |
| `-v, --verbose` | FLAG | — | Enable verbose output for detailed per-gadget results. |
| `-o, --output` | PATH | — | Write output to file. Plain text by default; use with `--json` for JSON output. |
| `-b, --bad-chars` | TEXT | — | Hex bytes to filter from gadget addresses (e.g. `"000a0d"`). |
| `--json` | FLAG | — | Output gadgets as structured JSON. Combine with `-o` to save to file. |
| `--all-exec` | FLAG | — | Analyze all executable sections, not just `.text`. |
| `-al, --algo` | FLAG | — | Use strictly the algorithmic ranking (bypass ML). |
| `--version` | FLAG | — | Show the installed version and exit. |
| `--help` | FLAG | — | Show help message and exit. |

---

---

## 📊 Accuracy & Benchmarks

LCSAJdump is backed by a rigorous, incrementally validated test suite located in the `benchmarkTests/` directory.

Through 14 major iterations of semantic feature engineering, the hybrid model has learned to discriminate gadgets based on actual memory side-effects (extracted via `angr` symbolic execution) rather than purely syntactic heuristics. 

When evaluated on monolithic, real-world executables like `libc.so.6`, the engine achieves a mathematically near-perfect **NDCG@1 = 0.9833** and **NDCG@10 = 0.9656**. The Two-Stage engine successfully prioritizes clean stack-popping sequences and `ret2csu`-like calls, while heavily penalizing crash-prone fixed-offset jumps that deceive traditional static scanners.

---

## 🧠 Developer & ML Guide

The repository is structured to support both end-users and ML researchers.

* **Production Engine:** The core CLI seamlessly integrates the inference engine using models hosted on [Hugging Face](https://huggingface.co/chris1sflaggin/chainfinder_hybrid), requiring no manual model loading.
* **ML Pipeline:** The `lcsajdump/ml_study/` directory contains the complete pipeline used to train the models:
  * `build_dataset.py`: Extracts structural and semantic features from a corpus of CTF binaries.
  * `train_model.py`: Trains the LightGBM LambdaRank model and outputs the `.pkl` models.
  * `kfold_cv.py`: Validates the dataset using K-Fold Cross Validation.

## Contributing (Open for Forks!)

The framework is open to new implementations. To add a new architecture:

1. **Fork** the repository.
2. Open `lcsajdump/core/config.py`.
3. Add a new profile to the `ARCH_PROFILES` dictionary, defining jump mnemonics, return mnemonics, and registers for the desired architecture (e.g., x86_64).
4. Submit a **Pull Request**.

---

## License

This project is released under the **MIT** license. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

## Project Link
Visit the project web page: [LCSAJdump web page](https://chris1sflaggin.it/LCSAJdump)

---

<div align="center">
Made by <b>Chris1sflaggin</b> as a research project for Automated Gadget Discovery.
</div>
