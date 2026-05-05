import collections
import json
import sys
import re

try:
    from lcsajdump.ml.features import clobbered_registers as _clobber_fn
    _HAVE_CLOBBER = True
except ImportError:
    _HAVE_CLOBBER = False


def _count_clobbered(insns):
    """Count distinct registers clobbered by an instruction sequence."""
    if _HAVE_CLOBBER:
        return len(_clobber_fn(insns))
    return 0


def reg_in_op(reg_config, op_str):
    if not reg_config:
        return False
    if isinstance(reg_config, str):
        return reg_config in op_str
    return any(r in op_str for r in reg_config)


class RainbowFinder:
    def __init__(self, graph_manager, max_depth, max_darkness, max_insns):
        self.gm = graph_manager
        self.gadgets = []
        self.MAX_DEPTH = max_depth
        self.MAX_DARKNESS = max_darkness
        self.profile = self.gm.profile
        self.MAX_INSNS = max_insns

        self.weights = self.profile.get("scoring_weights", {})
        self.base_score = self.weights.get("base_score", 100)
        self.insn_penalty = self.weights.get("insn_penalty", 2)

        self.l_reg = self.profile.get("link_reg")
        self.a_reg = self.profile.get("primary_arg_reg")
        self.f_reg = self.profile.get("frame_reg")

        self.t_mnems = self.profile.get("trampoline_mnems", set())
        self.r_mnems = self.profile.get("ret_mnems", set())
        self.c_mnems = self.profile.get("call_mnems", set())

        self.bonus_link = self.weights.get("bonus_link_reg", 50)
        self.bonus_arg = self.weights.get("bonus_arg_reg", 40)
        self.bonus_frame = self.weights.get("bonus_frame_reg", 30)
        self.bonus_tramp = self.weights.get("bonus_trampoline", 30)
        self.penalty_ret = self.weights.get("penalty_bad_ret", 20)
        self.penalty_call = self.weights.get("penalty_internal_call", 150)
        self.bonus_direct_call = self.weights.get("bonus_direct_call", 0)
        self.pivot_always = self.profile.get("pivot_always_mnems", frozenset())
        self.pivot_sp_mnems = self.profile.get("pivot_sp_mnems", frozenset())
        self.sp_reg = self.profile.get("stack_pointer_reg", "")
        self.bonus_pivot = self.weights.get("bonus_pivot", 0)
        self.syscall_mnems = self.profile.get("syscall_mnems", frozenset())
        self.bonus_syscall = self.weights.get("bonus_syscall", 80)
        self.clobber_penalty = self.weights.get("clobber_penalty", 60)
        self.long_chain_penalty = self.weights.get("long_chain_penalty", 25)
        self.max_clobber = self.weights.get("max_clobber", 3)

    @staticmethod
    def _addr_contains_bad_bytes(addr, bad_bytes):
        """Check if any byte in the significant portion of the address matches bad_bytes.
        Only checks bytes up to the highest non-zero byte (e.g., 0x40100a -> \\x0a\\x10\\x40)."""
        if addr == 0:
            return 0 in bad_bytes
        byte_len = (addr.bit_length() + 7) // 8
        addr_bytes = addr.to_bytes(byte_len, byteorder="little")
        return any(b in bad_bytes for b in addr_bytes)

    def score_gadget(self, path):
        score = self.base_score
        full_insns = []
        for addr in path:
            if addr in self.gm.addr_to_node:
                full_insns.extend(self.gm.addr_to_node[addr]["insns"])

        score -= len(full_insns) * self.insn_penalty

        n_clobber = _count_clobbered(full_insns)
        score -= n_clobber * self.clobber_penalty
        score -= max(0, len(full_insns) - 5) * self.long_chain_penalty

        has_link_reg = any(reg_in_op(self.l_reg, i.op_str) for i in full_insns)
        has_arg_reg = any(reg_in_op(self.a_reg, i.op_str) for i in full_insns)
        has_frame_reg = any(reg_in_op(self.f_reg, i.op_str) for i in full_insns)
        has_J = any(i.mnemonic.lower() in self.t_mnems for i in full_insns)

        has_internal_call = any(
            i.mnemonic.lower() in self.c_mnems for i in full_insns[:-1]
        )

        if has_link_reg:
            score += self.bonus_link
        if has_arg_reg:
            score += self.bonus_arg
        if has_frame_reg:
            score += self.bonus_frame
        has_pivot = any(
            i.mnemonic.lower() in self.pivot_always
            or (
                i.mnemonic.lower() in self.pivot_sp_mnems
                and self.sp_reg
                and self.sp_reg in i.op_str.lower()
            )
            for i in full_insns
        )
        if has_pivot:
            score += self.bonus_pivot
        if has_J:
            score += self.bonus_tramp
        if has_internal_call:
            score -= self.penalty_call

        if full_insns:
            last = full_insns[-1]
            if last.mnemonic.lower() in self.syscall_mnems:
                score += self.bonus_syscall
            elif (
                last.mnemonic.lower() in self.r_mnems
                and last.op_str  # Solo se ci sono operandi, verificare se è un "bad ret"
                and not reg_in_op(self.l_reg, last.op_str)
            ):
                score -= self.penalty_ret

        last_node = self.gm.addr_to_node.get(path[-1])
        if last_node and "direct_call_target" in last_node:
            direct_call_bonus = self.bonus_direct_call
            score += direct_call_bonus

        return score

    def search(self):
        print(
            f"\n[*] RainbowBFS config: Depth={self.MAX_DEPTH}, Darkness={self.MAX_DARKNESS}, Max Insns={self.MAX_INSNS}"
        )
        tails = self.gm.get_gadget_tails()

        queue = collections.deque()

        self.grouped_gadgets = {}

        for t in tails:
            insn_count = len(t["insns"])
            if insn_count <= self.MAX_INSNS:
                queue.append((t["start"], (t["start"],), {t["start"]}, insn_count))

        node_darkness = collections.defaultdict(int)
        pruned_darkness = 0
        pruned_insns = 0
        pruned_depth = 0
        duplicate_merges = 0

        while queue:
            head, path_tuple, visited, total_insns = queue.popleft()

            if len(path_tuple) >= 1:
                gadget_insns = []
                for addr in path_tuple:
                    gadget_insns.extend(self.gm.addr_to_node[addr]["insns"])

                sig = "; ".join([f"{i.mnemonic} {i.op_str}" for i in gadget_insns])
                start_addr = path_tuple[0]

                if sig not in self.grouped_gadgets:
                    self.grouped_gadgets[sig] = {
                        "path": list(path_tuple),
                        "addresses": {start_addr},
                    }
                else:
                    self.grouped_gadgets[sig]["addresses"].add(start_addr)
                    duplicate_merges += 1

            if len(path_tuple) >= self.MAX_DEPTH:
                pruned_depth += len(self.gm.reverse_graph.get(head, []))
                continue

            for parent in self.gm.reverse_graph.get(head, []):
                if parent in visited:
                    continue

                parent_node = self.gm.addr_to_node[parent]
                new_total_insns = total_insns + len(parent_node["insns"])

                if new_total_insns > self.MAX_INSNS:
                    pruned_insns += 1
                    continue

                if node_darkness[parent] >= self.MAX_DARKNESS:
                    pruned_darkness += 1
                    continue

                node_darkness[parent] += 1

                new_path = (parent,) + path_tuple
                new_visited = visited.copy()
                new_visited.add(parent)

                queue.append((parent, new_path, new_visited, new_total_insns))

        total_pruned = pruned_darkness + pruned_insns + pruned_depth
        print(
            f"[*] Pruning: {total_pruned} branches total | {pruned_darkness} darkness | {pruned_insns} insn limit | {pruned_depth} depth limit | {duplicate_merges} duplicates merged."
        )
        if pruned_darkness > pruned_insns + pruned_depth and pruned_darkness > 0:
            print(
                f"[\033[33m!\033[0m] High darkness pruning. Consider increasing -k (current: {self.MAX_DARKNESS})."
            )
        self.gadgets = [g["path"] for g in self.grouped_gadgets.values()]
        return self.gadgets

    def _classify_gadget(self, path):
        last_node = self.gm.addr_to_node[path[-1]]
        last_insn = last_node["last_insn"]
        mnem = last_insn.mnemonic.lower()

        if (
            mnem in self.profile["unconditional_jumps"]
            and mnem not in self.profile["ret_mnems"]
        ):
            return "JOP", "Jump-Based"

        elif mnem.startswith(self.profile["branch_prefixes"]):
            return "CONDITIONAL", "Jump-Based"

        if len(path) == 1:
            return "LINEAR", "Sequential"
        else:
            return "FALLTHROUGH", "Sequential"

    def _safe_print(self, text, file=sys.stdout):
        if file != sys.stdout:
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            text = ansi_escape.sub("", text)
        print(text, file=file)

    def print_gadgets(
        self, limit, min_score, verbose, out_file=sys.stdout, bad_bytes=None
    ):
        if verbose:
            self._safe_print(
                f"\n[*] Verbose Mode: Showing all {len(self.gadgets)} gadgets individually...",
                file=out_file,
            )
            categories = {"Sequential": [], "Jump-Based": []}

            for g in self.gadgets:
                if bad_bytes:
                    start_addr = (
                        self.gm.addr_to_node[g[0]]["insns"][0].address
                        if g[0] in self.gm.addr_to_node
                        else g[0]
                    )
                    if self._addr_contains_bad_bytes(start_addr, bad_bytes):
                        continue

                full_g_insns = []
                for addr in g:
                    if addr in self.gm.addr_to_node:
                        full_g_insns.extend(self.gm.addr_to_node[addr]["insns"])
                if _count_clobbered(full_g_insns) > self.max_clobber:
                    continue

                s = self.score_gadget(g)
                if s < min_score:
                    continue

                tag, cat = self._classify_gadget(g)
                if cat not in categories:
                    cat = "Sequential"
                categories[cat].append((s, g, tag))

            for cat_name in ["Sequential", "Jump-Based"]:
                sorted_gadgets = sorted(
                    categories[cat_name], key=lambda x: x[0], reverse=True
                )

                actual_count = min(limit, len(sorted_gadgets))

                if actual_count == 0:
                    continue

                self._safe_print(f"\033[33m\n{'=' * 80}\033[0m", file=out_file)
                self._safe_print(
                    f"\033[33m--- TOP {actual_count} UNIQUE {cat_name.upper()} GADGETS ---\033[0m",
                    file=out_file,
                )
                self._safe_print(f"\033[33m{'=' * 80}\033[0m", file=out_file)

                for i, (s, p, tag) in enumerate(sorted_gadgets[:limit]):
                    self._safe_print(
                        f"\nRANK #{i + 1} | SCORE: {s} | TYPE: {tag}", file=out_file
                    )
                    for addr in p:
                        if addr in self.gm.addr_to_node:
                            node = self.gm.addr_to_node[addr]
                            for insn in node["insns"]:
                                self._safe_print(
                                    f"  \033[33m{hex(insn.address)}\033[0m: {insn.mnemonic} {insn.op_str}",
                                    file=out_file,
                                )

            shown_seq = min(limit, len(categories["Sequential"]))
            shown_jmp = min(limit, len(categories["Jump-Based"]))
            return {
                "total": len(categories["Sequential"]) + len(categories["Jump-Based"]),
                "sequential": len(categories["Sequential"]),
                "jump_based": len(categories["Jump-Based"]),
                "shown_seq": shown_seq,
                "shown_jmp": shown_jmp,
            }

        else:
            categories = {"Sequential": [], "Jump-Based": []}

            for sig, data in self.grouped_gadgets.items():
                path = data["path"]
                addrs = sorted(list(data["addresses"]))

                if bad_bytes:
                    addrs = [
                        a
                        for a in addrs
                        if not self._addr_contains_bad_bytes(a, bad_bytes)
                    ]
                    if not addrs:
                        continue

                path_insns = []
                for addr in path:
                    if addr in self.gm.addr_to_node:
                        path_insns.extend(self.gm.addr_to_node[addr]["insns"])
                if _count_clobbered(path_insns) > self.max_clobber:
                    continue

                s = self.score_gadget(path)
                if s < min_score:
                    continue

                tag, cat = self._classify_gadget(path)
                if cat not in categories:
                    cat = "Sequential"

                categories[cat].append(
                    {"score": s, "signature": sig, "addresses": addrs}
                )

            for cat_name in ["Sequential", "Jump-Based"]:
                sorted_gadgets = sorted(
                    categories[cat_name], key=lambda x: x["score"], reverse=True
                )

                actual_count = min(limit, len(sorted_gadgets))

                if actual_count == 0:
                    continue

                self._safe_print(f"\033[33m\n{'=' * 80}\033[0m", file=out_file)
                self._safe_print(
                    f"\033[33m--- TOP {actual_count} UNIQUE {cat_name.upper()} GADGETS ---\033[0m",
                    file=out_file,
                )
                self._safe_print(f"\033[33m{'=' * 80}\033[0m", file=out_file)

                for i, item in enumerate(sorted_gadgets[:limit]):
                    addrs = item["addresses"]
                    count = len(addrs)
                    sig = item["signature"]

                    primary_addr = hex(addrs[0])

                    self._safe_print(
                        f"\033[33m{primary_addr}\033[0m: {sig}", file=out_file
                    )

                    if count > 1:
                        others_to_show = addrs[1:5]
                        others_str = ", ".join([hex(a) for a in others_to_show])
                        more_str = f" ... (+{count - 5} others)" if count > 5 else ""
                        self._safe_print(
                            f"  \033[90mFound {count} times (e.g. at {others_str}{more_str})\033[0m",
                            file=out_file,
                        )

            shown_seq = min(limit, len(categories["Sequential"]))
            shown_jmp = min(limit, len(categories["Jump-Based"]))
            return {
                "total": len(categories["Sequential"]) + len(categories["Jump-Based"]),
                "sequential": len(categories["Sequential"]),
                "jump_based": len(categories["Jump-Based"]),
                "shown_seq": shown_seq,
                "shown_jmp": shown_jmp,
            }

    def gadgets_to_json(self, limit, min_score, bad_bytes=None):
        """Return gadgets as a JSON string."""
        categories = {"Sequential": [], "Jump-Based": []}

        for sig, data in self.grouped_gadgets.items():
            path = data["path"]
            addrs = sorted(list(data["addresses"]))

            if bad_bytes:
                addrs = [
                    a for a in addrs if not self._addr_contains_bad_bytes(a, bad_bytes)
                ]
                if not addrs:
                    continue

            json_path_insns = []
            for addr in path:
                if addr in self.gm.addr_to_node:
                    json_path_insns.extend(self.gm.addr_to_node[addr]["insns"])
            if _count_clobbered(json_path_insns) > self.max_clobber:
                continue

            s = self.score_gadget(path)
            if s < min_score:
                continue

            tag, cat = self._classify_gadget(path)
            if cat not in categories:
                cat = "Sequential"

            insns_list = []
            for addr in path:
                if addr in self.gm.addr_to_node:
                    for insn in self.gm.addr_to_node[addr]["insns"]:
                        insns_list.append(
                            {
                                "address": hex(insn.address),
                                "mnemonic": insn.mnemonic,
                                "op_str": insn.op_str,
                            }
                        )

            entry = {
                "primary_address": hex(addrs[0]),
                "type": cat,
                "tag": tag,
                "score": s,
                "instructions": insns_list,
                "duplicate_addresses": [hex(a) for a in addrs[1:]],
                "duplicate_count": len(addrs),
            }
            categories[cat].append(entry)

        for cat in categories:
            categories[cat] = sorted(
                categories[cat], key=lambda x: x["score"], reverse=True
            )[:limit]

        result = {
            "sequential": categories["Sequential"],
            "jump_based": categories["Jump-Based"],
            "summary": {
                "sequential_count": len(categories["Sequential"]),
                "jump_based_count": len(categories["Jump-Based"]),
                "total": len(categories["Sequential"]) + len(categories["Jump-Based"]),
            },
        }
        return json.dumps(result, indent=2)
