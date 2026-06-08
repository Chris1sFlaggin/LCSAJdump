import click
import sys
from .core.loader import BinaryLoader
from .core.graph import LCSAJGraph
from .core.rainbowBFS import RainbowFinder
from .core.config import ARCH_PROFILES
from elftools.elf.elffile import ELFFile


def auto_detect_env(binary_path):
    """Legge l'header ELF e deduce l'architettura."""
    try:
        with open(binary_path, "rb") as f:
            elf = ELFFile(f)
            arch_str = elf.get_machine_arch()

            if arch_str in ["x64", "EM_X86_64"]:
                return "x86_64"
            elif arch_str in ["x86", "EM_386", "i386"]:
                return "x86_32"
            elif arch_str in ["AArch64", "EM_AARCH64"]:
                return "arm64"
            elif arch_str in ["RISC-V", "EM_RISCV"]:
                return "riscv64"
            else:
                return "riscv64"
    except Exception:
        return "riscv64"


@click.command()
@click.argument("binary_path", type=click.Path(exists=True))
@click.option(
    "--depth", "-d", default=None, type=int, help="Max search depth (LCSAJ blocks)."
)
@click.option(
    "--darkness",
    "-k",
    default=None,
    type=int,
    help="Pruning threshold (Max visits per node).",
)
@click.option(
    "--limit", "-l", default=None, type=int, help="Desired number of gadgets to show."
)
@click.option(
    "--min-score",
    "-s",
    default=None,
    type=int,
    help="Min score for a gadget to be shown.",
)
@click.option(
    "--instructions",
    "-i",
    default=None,
    type=int,
    help="Max number of instructions contained in a single node",
)
@click.option(
    "--arch",
    "-a",
    default="auto",
    help="Architecture of the binary (auto[default], riscv64, x86_64, arm64).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose results for a better detailed result.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write gadgets to the specified file path.",
)
@click.option(
    "--bad-chars",
    "-b",
    default="",
    help='Hex bytes to filter from gadget addresses (e.g. "000a0d").',
)
@click.option(
    "--json", "json_output", is_flag=True, help="Output gadgets as structured JSON."
)
@click.option(
    "--all-exec", is_flag=True, help="Analyze all executable sections, not just .text."
)
@click.option(
    "--algo",
    "-al",
    is_flag=True,
    help="Use strictly the algorithmic ranking (no ML).",
)
@click.option(
    "--max-clobber",
    "-mc",
    default=None,
    type=int,
    help="Hard max clobbered registers per gadget (overrides profile default).",
)
@click.version_option(version="2.0.7", prog_name="LCSAJdump")
def main(
    binary_path,
    depth,
    darkness,
    limit,
    min_score,
    instructions,
    verbose,
    output,
    arch,
    bad_chars,
    json_output,
    all_exec,
    algo,
    max_clobber,
):
    """
    LCSAJ ROP Finder.
    Analyze a binary to find ROP gadgets using Rainbow BFS algorithm.
    """
    print(
        "\33[33m"
        + r"""
██╗      ██████╗███████╗   █████╗       ██╗██████╗ ██╗   ██╗███╗   ███╗██████╗
██║     ██╔════╝██╔════╝  ██╔══██╗      ██║██╔══██╗██║   ██║████╗ ████║██╔══██╗
██║     ██║     ███████╗  ███████║      ██║██║  ██║██║   ██║██╔████╔██║██████╔╝
██║     ██║     ╚════██║  ██╔══██║ ██╗  ██║██║  ██║██║   ██║██║╚██╔╝██║██╔═══╝ 
███████╗╚██████╗███████║  ██║  ██║ ╚█████╔╝██████╔╝╚██████╔╝██║ ╚═╝ ██║██║     
╚══════╝ ╚═════╝╚══════╝  ╚═╝  ╚═╝  ╚════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝     
                                    v2.0.7

RISC-V    ARM64    x86-64


https://github.com/sponsors/Chris1sFlaggin - Support the project! | https://chris1sflaggin.it/LCSAJdump/ - Official Website
"""
        + "\33[0m"
    )

    print(f"[*] Analyzing Target: {binary_path}")

    if arch is None or arch == "auto":
        arch = auto_detect_env(binary_path)
        print(
            f"[\033[32m+\033[0m] Auto-detected architecture: \033[1m{arch.upper()}\033[0m"
        )

    # Load architecture defaults from config.py
    arch_config = ARCH_PROFILES.get(arch, {})
    arch_defaults = arch_config.get("search_params", {})

    # Apply defaults from config.py if user didn't override via CLI
    if depth is None:
        depth = arch_defaults.get("d", 20)
    if darkness is None:
        darkness = arch_defaults.get("darkness", 5)
    if limit is None:
        limit = arch_defaults.get("limit", 10)
    if instructions is None:
        instructions = arch_defaults.get("i", 15)
    if min_score is None:
        min_score = arch_defaults.get("m", 0)

    # Parse bad characters
    bad_bytes = set(bytes.fromhex(bad_chars)) if bad_chars else None

    # Resolve output destination
    out_path = output if output else None

    loader = BinaryLoader(binary_path, arch, all_exec=all_exec)
    insns = loader.disassemble()

    gb = LCSAJGraph(insns, arch)
    gb.binary_path = binary_path
    gb.build_lazy(max_depth=depth)

    finder = RainbowFinder(
        gb, max_depth=depth, max_darkness=darkness, max_insns=instructions
    )
    if max_clobber is not None:
        finder.max_clobber = max_clobber
    gadgets = finder.search()

    use_ml = not algo

    if algo and min_score == arch_defaults.get("m", 0):
        # Prevent the purely algorithmic mode from discarding all gadgets due to 
        # heavily negative scores generated by Optuna's aggressive penalties.
        # This occurs because the ML stage expects to Min-Max normalize these.
        min_score = -999999

    # ── Optional Legacy ML re-ranking (LightGBM static only) ──
    if use_ml:
        import glob as _glob
        import os as _os

        ml_model = None
        # Auto-detect model file if not specified. Search order:
        #   1. Current working directory (user's local override)
        #   2. Packaged models directory (lcsajdump_dbg/ml/models/)
        if ml_model is None:
            search_dirs = [
                ".",
                _os.path.join(_os.path.dirname(__file__), "ml", "models"),
            ]
            for d in search_dirs:
                candidates = sorted(
                    _glob.glob(_os.path.join(d, "*.pkl")),
                    reverse=True,
                )
                if candidates:
                    ml_model = candidates[0]
                    break
        if ml_model and _os.path.exists(ml_model):
            try:
                # patch_rainbowfinder monkey-patches finder.score_gadget so that
                # gadgets_to_json() and print_gadgets() automatically use ML scores.
                from .ml.model_scorer import patch_rainbowfinder

                patch_rainbowfinder(finder, model_path=ml_model, arch=arch)
                if getattr(finder, "_ml_patched", False):
                    print(
                        f"[\033[32m+\033[0m] ML re-ranking active ({_os.path.basename(ml_model)})"
                    )
            except (FileNotFoundError, KeyError, ValueError, ImportError) as e:
                print(
                    f"[\033[33m!\033[0m] ML re-ranking failed: {e} — using heuristic scores"
                )
        else:
            print(
                f"[\033[33m!\033[0m] --ml: no model file found (specify with --ml-model PATH)"
            )

    if json_output:
        json_str = finder.gadgets_to_json(
            limit=limit, min_score=min_score, bad_bytes=bad_bytes
        )
        if out_path:
            try:
                with open(out_path, "w") as fh:
                    fh.write(json_str)
                print(f"\n[\033[32m+\033[0m] JSON report saved in: {out_path}")
            except Exception as e:
                print(f"[!] Error while saving file: {e}")
        else:
            print(json_str)
    elif out_path:
        try:
            with open(out_path, "w") as fh:
                summary = finder.print_gadgets(
                    limit=len(gadgets),
                    min_score=min_score,
                    verbose=verbose,
                    out_file=fh,
                    bad_bytes=bad_bytes,
                )
            print(f"\n[\033[32m+\033[0m] Report saved in: {out_path}")
        except Exception as e:
            print(f"[!] Error while saving file: {e}")
            summary = None
    else:
        summary = finder.print_gadgets(
            limit=limit,
            min_score=min_score,
            verbose=verbose,
            out_file=sys.stdout,
            bad_bytes=bad_bytes,
        )

    if not json_output and summary:
        print(
            f"\nFound {summary['total']} gadgets. Showing {summary['shown_seq']} sequential + {summary['shown_jmp']} jump-based (use --limit N to show more)."
        )


if __name__ == "__main__":
    main()
