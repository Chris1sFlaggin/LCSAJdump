#!/usr/bin/env python3
"""
BOF CTF Analyzer v6.0 — static analysis, output compatto e leggibile
Dipendenze: pip install pwntools
Uso: python3 tool.py ./binary
"""

import sys, subprocess, re, argparse
from pathlib import Path

try:
    from pwn import *

    context.log_level = "error"
except ImportError:
    print("[!] pwntools non trovato: pip install pwntools")
    sys.exit(1)


class C:
    R = "\033[91m"
    G = "\033[92m"
    Y = "\033[93m"
    CY = "\033[96m"
    M = "\033[95m"
    W = "\033[0m"
    BD = "\033[1m"
    DIM = "\033[2m"
    UL = "\033[4m"


RISK_ORD = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
RISK_COL = {"CRITICAL": C.R + C.BD, "HIGH": C.R, "MEDIUM": C.Y, "LOW": C.DIM}
RISK_ICO = {"CRITICAL": "!!", "HIGH": "! ", "MEDIUM": "~ ", "LOW": "  "}

DANGEROUS = {
    "gets": ("CRITICAL", "no bounds check — overflow certo", False),
    "scanf": ("CRITICAL", "%s senza precisione → overflow", False),
    "strcpy": ("CRITICAL", "no bounds check su dst", False),
    "strcat": ("HIGH", "append illimitato", False),
    "sprintf": ("HIGH", "%s senza precisione", False),
    "vsprintf": ("HIGH", "vsprintf senza bounds", False),
    "printf": ("MEDIUM", "possibile format string", False),
    "fprintf": ("MEDIUM", "possibile format string", False),
    "read": ("MEDIUM", "size da rdx — da verificare", True),
    "fgets": ("LOW", "bounded — verifica size vs buf", True),
    "memcpy": ("MEDIUM", "size da rdx", True),
    "memmove": ("MEDIUM", "size da rdx", True),
    "recv": ("MEDIUM", "size da rdx", True),
    "recvfrom": ("MEDIUM", "size da rdx", True),
    "strncpy": ("LOW", "bounded — off-by-one?", True),
    "snprintf": ("LOW", "bounded — size corretto?", True),
}


# ─── helpers ──────────────────────────────────────────────────────────────────
def sep(title="", char="─", width=52):
    if title:
        pad = width - len(title) - 2
        print(f"\n{C.BD}{char*2} {title} {char*pad}{C.W}")
    else:
        print(f"{C.DIM}{char*width}{C.W}")


def tag(label, color=C.CY):
    return f"{color}[{label}]{C.W}"


def yn(val):
    return f"{C.G}yes{C.W}" if val else f"{C.R}no{C.W}"


# ─────────────────────────────────────────────────────────────────────────────
class BOFAnalyzer:

    def __init__(self, path):
        self.path = path
        self.name = Path(path).name
        try:
            self.elf = ELF(path, checksec=False)
            self.arch = self.elf.arch
            self.bits = self.elf.bits
        except Exception as e:
            print(f"{C.R}[!] ELF non valido: {e}{C.W}")
            sys.exit(1)

        sys.stdout.write(f"{C.DIM}disassemblando...{C.W}\r")
        sys.stdout.flush()
        self.disasm = subprocess.run(
            ["objdump", "-d", "-M", "intel", path], capture_output=True, text=True
        ).stdout
        self.symbols = (
            subprocess.run(["nm", path], capture_output=True, text=True).stdout
            + subprocess.run(["nm", "-D", path], capture_output=True, text=True).stdout
        )
        sys.stdout.write(" " * 30 + "\r")
        self.prot = {}

    # ── 1. CHECKSEC ──────────────────────────────────────────────────────────
    def check_security(self):
        sep("PROTEZIONI")
        self.prot = {
            "NX": self.elf.nx,
            "PIE": self.elf.pie,
            "Canary": self.elf.canary,
            "RELRO": self.elf.relro,
        }
        p = self.prot

        # riga unica con semafori
        def pill(k, v):
            col = C.G if v else C.R
            return f"{col}{'✓' if v else '✗'} {k}{C.W}"

        print(
            f"  {pill('NX',p['NX'])}   {pill('PIE',p['PIE'])}   {pill('Canary',p['Canary'])}",
            end="",
        )
        rc = C.G if p["RELRO"] == "Full" else (C.Y if p["RELRO"] == "Partial" else C.R)
        print(f"   {rc}RELRO:{p['RELRO'] or 'none'}{C.W}   {C.DIM}{self.arch}{C.W}")

        # strategia in una riga
        print()
        if not p["NX"]:
            print(f"  {tag('strategia',C.CY)} shellcode injection  (NX off)")
        elif not p["Canary"] and p["NX"]:
            print(
                f"  {tag('strategia',C.CY)} ret2libc / ROP  (no canary → overflow diretto al ret)"
            )
        elif p["Canary"] and not p["PIE"]:
            print(
                f"  {tag('strategia',C.CY)} canary leak  +  ROP su indirizzi statici  (PIE off)"
            )
        elif p["Canary"] and p["PIE"]:
            print(f"  {tag('strategia',C.CY)} canary leak  +  PIE leak  +  ROP")
        return self.prot

    # ── 2. FUNZIONI VULNERABILI ───────────────────────────────────────────────
    def find_vulnerable_calls(self):
        sep("FUNZIONI VULNERABILI")
        found = {}
        for func, (risk, reason, bounded) in DANGEROUS.items():
            addrs = []
            pat = re.compile(
                rf"([0-9a-f]+):\s+(?:[0-9a-f]{{2}} )+\s+call\s+[0-9a-f]+\s+"
                rf"<{re.escape(func)}(?:@plt)?>",
                re.IGNORECASE,
            )
            for m in pat.finditer(self.disasm):
                addrs.append("0x" + m.group(1))
            in_plt = (
                func in self.symbols
                or f"<{func}@" in self.disasm
                or f"<{func}>" in self.disasm
            )
            if addrs or in_plt:
                found[func] = {
                    "risk": risk,
                    "reason": reason,
                    "bounded": bounded,
                    "addrs": addrs or ["PLT"],
                }

        if not found:
            print(f"  {C.G}nessuna funzione pericolosa nota{C.W}")
            print()
            return []

        # allineamento colonne
        max_fname = max(len(f) for f in found) + 2
        for func, info in sorted(found.items(), key=lambda x: RISK_ORD[x[1]["risk"]]):
            col = RISK_COL[info["risk"]]
            ico = RISK_ICO[info["risk"]]
            btag = f"{C.DIM}[b]{C.W}" if info["bounded"] else "   "
            addrs_s = "  ".join(info["addrs"][:3])
            fname_s = f"{func}()".ljust(max_fname)
            print(
                f"  {col}{ico}{C.W} {C.BD}{fname_s}{C.W} {btag}  {C.DIM}{info['reason']}{C.W}"
            )
            print(f"       {C.CY}{addrs_s}{C.W}")
        print()
        return list(found.items())

    # ── 3. ANALISI SIZE ───────────────────────────────────────────────────────
    def analyze_read_size(self):
        sep("ANALISI SIZE read/recv/fgets")
        findings = []
        call_pat = re.compile(
            r"([0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s+call\s+[0-9a-f]+\s+"
            r"<(read|fgets|recv|memcpy|memmove|recvfrom)(?:@plt)?>",
            re.IGNORECASE,
        )

        for m in call_pat.finditer(self.disasm):
            call_addr = int(m.group(1), 16)
            func_name = m.group(2)
            start = max(0, m.start() - 1200)
            lines = self.disasm[start : m.end()].strip().split("\n")[-28:-1]

            size_val = None
            lea_found = None
            frame_sz = None
            for line in lines:
                l = line.strip()
                mx = re.search(r"mov\s+(?:edx|rdx),\s*(?:0x)?([0-9a-f]+)", l, re.I)
                if mx:
                    v = int(mx.group(1), 16)
                    if 0 < v < 0x100000:
                        size_val = v
                mx = re.search(
                    r"lea\s+\w+,\s*\[(?:rbp|rsp)[+-](?:0x)?([0-9a-f]+)\]", l, re.I
                )
                if mx:
                    lea_found = int(mx.group(1), 16)
                mx = re.search(r"sub\s+(?:rsp|esp),\s*(?:0x)?([0-9a-f]+)", l, re.I)
                if mx:
                    frame_sz = int(mx.group(1), 16)

            # riga compatta
            sz_s = f"{C.BD}{size_val}B{C.W}" if size_val else f"{C.DIM}?B{C.W}"
            buf_s = f"{lea_found}B (rbp-0x{lea_found:x})" if lea_found else "?"

            if size_val and lea_found:
                if size_val > lea_found:
                    d = size_val - lea_found
                    verdict = f"{C.R}{C.BD}OVERFLOW +{d}B{C.W}"
                elif size_val == lea_found:
                    verdict = f"{C.Y}OFF-BY-ONE{C.W}"
                else:
                    verdict = f"{C.G}ok{C.W}"
            elif size_val:
                verdict = f"{C.Y}size={size_val}B — controlla manualmente{C.W}"
            else:
                verdict = f"{C.DIM}size non determinabile{C.W}"

            print(f"  {C.CY}{func_name}(){C.W} @ 0x{call_addr:x}")
            print(f"    size={sz_s}  buf={C.CY}{buf_s}{C.W}  →  {verdict}")
            if frame_sz:
                print(f"    {C.DIM}frame: 0x{frame_sz:x} ({frame_sz}B){C.W}")
            print()
            findings.append(
                {
                    "func": func_name,
                    "addr": call_addr,
                    "size": size_val,
                    "buf": lea_found,
                }
            )

        if not findings:
            print(f"  {C.DIM}nessuna call trovata{C.W}\n")
        return findings

    # ── 4. LAYOUT STACK ───────────────────────────────────────────────────────
    def analyze_stack_layout(self):
        sep("LAYOUT STACK")
        offsets = []
        func_pat = re.compile(
            r"([0-9a-f]+) <([^>]+)>:\n((?:.*\n)*?)(?=\n[0-9a-f]+ <|\Z)"
        )

        for m in func_pat.finditer(self.disasm):
            faddr = m.group(1)
            fname = m.group(2)
            body = m.group(3)
            if "@" in fname:
                continue
            if not re.search(r"\bret\b|\bleave\b", body):
                continue

            frame_m = re.search(r"sub\s+(?:rsp|esp),\s*(?:0x)?([0-9a-f]+)", body, re.I)
            frame = int(frame_m.group(1), 16) if frame_m else None

            bufs = sorted(
                set(
                    int(lm.group(1), 16)
                    for lm in re.finditer(
                        r"lea\s+\w+,\s*\[(?:rbp|rsp)[+-](?:0x)?([0-9a-f]+)\]",
                        body,
                        re.I,
                    )
                ),
                reverse=True,
            )

            has_can = bool(re.search(r"fs:0x28|__stack_chk|xor.*canary", body, re.I))
            if frame is None and not bufs:
                continue

            # header funzione
            print(f"  {C.BD}{fname}{C.W}  {C.DIM}@ 0x{faddr}{C.W}")

            if bufs:
                biggest = bufs[0]
                if has_can or self.prot.get("Canary"):
                    rip = biggest + 8 + 8
                    can_lbl = f"  {C.Y}canary{C.W} @ rbp-0x8"
                else:
                    rip = biggest + 8
                    can_lbl = ""

                # stack diagram su una riga
                print(f"    buf={C.CY}{biggest}B{C.W}{can_lbl}  frame={frame or '?'}B")

                # struttura payload compatta
                pad_s = f"{C.DIM}A×{biggest}{C.W}"
                can_s = (
                    f" {C.Y}canary{C.W} "
                    if (has_can or self.prot.get("Canary"))
                    else ""
                )
                print(f"    payload: [{pad_s}][{can_s}][rbp 8B][{C.R}RET{C.W}]")
                print(f"    {C.R}{C.BD}offset→RIP: {rip}B{C.W}")

                offsets.append(
                    {
                        "func": fname,
                        "buf": biggest,
                        "canary": has_can or self.prot.get("Canary", False),
                        "rip_offset": rip,
                    }
                )
            print()

        if not offsets:
            print(f"  {C.DIM}non conclusivo — GDB: info frame / x/80gx $rsp{C.W}\n")
        return offsets

    # ── 5. CANARY LEAK VECTORS ────────────────────────────────────────────────
    def detect_canary_leak_vectors(self):
        if not self.prot.get("Canary"):
            return
        sep("VETTORI LEAK CANARY")
        vectors = []
        for ff in ["printf", "fprintf", "sprintf", "vprintf"]:
            if ff in self.symbols or f"<{ff}@" in self.disasm:
                vectors.append(("FMT-STR", ff, "%p/%x → leaka stack, canary a rbp-0x8"))
        if "write" in self.symbols or "<write@" in self.disasm:
            vectors.append(
                ("INFO-LEAK", "write", "size>strlen → leaka stack incluso canary")
            )
        if "puts" in self.symbols or "<puts@" in self.disasm:
            vectors.append(("INFO-LEAK", "puts", "buf non null-term → leaka adiacenti"))
        if "fork" in self.symbols or "<fork@" in self.disasm:
            vectors.append(
                ("BRUTEFORCE", "fork", "figlio mantiene canary → 256 try/byte")
            )
        for f in ["read", "fgets", "recv"]:
            if f in self.symbols or f"<{f}@" in self.disasm:
                vectors.append(
                    (
                        "OFF-BY-ONE",
                        f,
                        'overflow 1B → sovrascrive \\x00 → puts leaka 7B → b"\\x00"+recv(7)',
                    )
                )
                break

        max_k = max(len(k) for k, _, _ in vectors) if vectors else 10
        for kind, func, desc in vectors:
            col = C.R if kind in ("FMT-STR", "BRUTEFORCE") else C.Y
            print(f"  {col}{kind.ljust(max_k)}{C.W}  {C.BD}{func}(){C.W}")
            print(f"    {C.DIM}{desc}{C.W}")
        print()

    # ── 6. ROP GADGETS & SIMBOLI ─────────────────────────────────────────────
    def find_rop_gadgets(self):
        sep("ROP GADGETS & PLT")
        try:
            rop = ROP(self.elf)
        except Exception:
            print(f"  {C.Y}ROP non disponibile{C.W}\n")
            return {}

        gadgets = {}
        wanted = [
            (["pop rdi", "ret"], "pop rdi ; ret      ← 1° arg"),
            (["pop rsi", "ret"], "pop rsi ; ret      ← 2° arg"),
            (["pop rdx", "ret"], "pop rdx ; ret      ← 3° arg"),
            (["pop rsp", "ret"], "pop rsp ; ret      ← pivot"),
            (["ret"], "ret                ← alignment"),
            (["pop rbp", "ret"], "pop rbp ; ret"),
            (["pop rdi", "pop rsi", "ret"], "pop rdi ; pop rsi ; ret"),
        ]
        for seq, label in wanted:
            try:
                g = rop.find_gadget(seq)
                if g:
                    gadgets[label] = g[0]
                    print(f"  {C.G}+{C.W} {C.CY}0x{g[0]:x}{C.W}  {label}")
            except Exception:
                pass
        if not gadgets:
            print(f"  {C.DIM}nessun gadget — ROPgadget --binary {self.path} --rop{C.W}")

        # PLT su una colonna compatta
        print(f"\n  {C.BD}plt:{C.W}")
        for sym in [
            "puts",
            "printf",
            "system",
            "execve",
            "read",
            "write",
            "exit",
            "mprotect",
        ]:
            if sym in self.elf.plt:
                col = C.G if sym in ("system", "execve") else C.CY
                got = self.elf.got.get(sym, 0)
                print(
                    f"    {col}{sym}{C.W} = 0x{self.elf.plt[sym]:x}  {C.DIM}got=0x{got:x}{C.W}"
                )

        # simboli win/flag/shell
        print(f"\n  {C.BD}simboli interessanti:{C.W}")
        sus = re.compile(
            r"([0-9a-f]+)\s+\w\s+(win|flag|shell|backdoor|secret|get_flag|give_flag|"
            r"vuln|overflow|bof|chall|pwn|door|spawn|cat_flag)[^\s]*",
            re.I,
        )
        found_s = False
        for line in self.symbols.splitlines():
            mx = sus.search(line)
            if mx:
                addr, name = mx.group(1), mx.group(2)
                sym_addr = self.elf.sym.get(name, 0) or self.elf.plt.get(name, 0)
                addr_s = f"0x{sym_addr:x}" if sym_addr else f"0x{addr}"
                print(f"    {C.G}{C.BD}★ {name}{C.W}  @ {C.CY}{addr_s}{C.W}")
                found_s = True
        if not found_s:
            print(f"    {C.DIM}nessuno{C.W}")
        print()
        return gadgets

    # ── 7. RIEPILOGO ─────────────────────────────────────────────────────────
    def print_summary(self, offsets, read_findings):
        sep("RIEPILOGO", "═")

        any_bof = False
        for f in read_findings:
            if f["size"] and f["buf"]:
                if f["size"] > f["buf"]:
                    d = f["size"] - f["buf"]
                    print(
                        f"  {C.R}{C.BD}[BOF]{C.W} {f['func']}()  "
                        f"legge {f['size']}B  in {f['buf']}B  (+{d}B)"
                    )
                    any_bof = True
                elif f["size"] == f["buf"]:
                    print(
                        f"  {C.Y}[OFF-BY-ONE]{C.W} {f['func']}()  size==buf ({f['buf']}B)"
                    )
                    any_bof = True
        if not any_bof:
            print(f"  {C.DIM}nessun overflow confermato staticamente{C.W}")

        if offsets:
            b = offsets[0]
            sep()
            can_s = (
                f"{C.Y}sì{C.W}  → leak prima del BOF"
                if b["canary"]
                else f"{C.G}no{C.W}  → overflow diretto"
            )
            print(f"  target     {C.CY}{b['func']}{C.W}")
            print(f"  buf size   {C.BD}{b['buf']}B{C.W}")
            print(f"  canary     {can_s}")
            print(f"  {C.R}{C.BD}RIP offset {b['rip_offset']}B{C.W}")

            if b["canary"]:
                buf = b["buf"]
                print(f"\n  {C.DIM}[A×{buf}][canary 8B][rbp 8B][RET →]{C.W}")

        sep()
        print(f"  {C.DIM}ROPgadget --binary {self.path} --rop | grep 'pop rdi'")
        print(f"  one_gadget libc.so.6")
        print(f"  readelf -s {self.path}{C.W}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="BOF CTF Analyzer v6")
    pa.add_argument("binary")
    pa.add_argument("--no-rop", action="store_true")
    args = pa.parse_args()

    path = args.binary
    if not Path(path).exists():
        print(f"{C.R}[!] non trovato: {path}{C.W}")
        sys.exit(1)

    print(f"\n{C.CY}{C.BD}BOF CTF Analyzer v6.0{C.W}  {C.DIM}static · no fuzzing{C.W}")
    print(f"{C.BD}target:{C.W} {path}  {C.DIM}({Path(path).stat().st_size}B){C.W}")

    a = BOFAnalyzer(path)
    a.check_security()
    a.find_vulnerable_calls()
    rf = a.analyze_read_size()
    of = a.analyze_stack_layout()
    a.detect_canary_leak_vectors()
    if not args.no_rop:
        a.find_rop_gadgets()
    a.print_summary(of, rf)


if __name__ == "__main__":
    main()
