import re
import capstone
from collections import defaultdict
from .loader import draw_progress
from .config import ARCH_PROFILES

HEX_PATTERN = re.compile(r'(0x[0-9a-fA-F]+)')

class LCSAJGraph:
    def __init__(self, instructions, arch="riscv64"):
        self.instructions = instructions
        self.arch = arch

        profile_data = ARCH_PROFILES[arch]
        self.profile = profile_data

        self.jump_mnems = set(profile_data["jump_mnems"])
        self.unconditional_jumps = set(profile_data["unconditional_jumps"])
        self.ret_mnems = set(profile_data["ret_mnems"])
        self.branch_prefixes = profile_data["branch_prefixes"]
        self.trampoline_mnems = profile_data["trampoline_mnems"]

        self.nodes = []
        self.addr_to_node = {}
        self.insn_to_block_start = {}
        self.reverse_graph = defaultdict(list)

    def build(self):
        self._create_nodes()
        self._discover_unaligned_x86()
        self._build_edges()

    def _create_nodes(self):
        if not self.instructions: return
        total_insns = len(self.instructions)
        current_block_insns = []
        block_start = self.instructions[0].address

        jump_mnems = self.jump_mnems
        branch_prefixes = self.branch_prefixes

        print(f"[*] Building LCSAJ Nodes for {self.arch}...")
        for idx, insn in enumerate(self.instructions):
            if idx % 5000 == 0:
                draw_progress(idx, total_insns, "Building Graph")

            current_block_insns.append(insn)
            mnem = insn.mnemonic.lower()

            is_jump = mnem in jump_mnems or mnem.startswith(branch_prefixes)

            if is_jump:
                self._add_node(block_start, current_block_insns)
                current_block_insns = []
                if idx + 1 < total_insns:
                    block_start = self.instructions[idx+1].address

        if current_block_insns:
            self._add_node(block_start, current_block_insns)
        draw_progress(total_insns, total_insns, "Building Graph")

    def _discover_unaligned_x86(self):
        if self.arch not in ["x86_64", "x86_32"]:
            return
            
        print("[*] Discovering unaligned shadow instructions...")
        md = capstone.Cs(self.profile["cs_arch"], self.profile["cs_mode"])
        md.detail = True
        
        shadow_nodes = []
        for insn in self.instructions:
            if insn.size > 1:
                b = bytes(insn.bytes)
                for i in range(1, insn.size):
                    slice_bytes = b[i:]
                    for shadow_insn in md.disasm(slice_bytes, insn.address + i):
                        if shadow_insn.size == len(slice_bytes):
                            start_addr = shadow_insn.address
                            if start_addr not in self.insn_to_block_start:
                                node = {
                                    'start': start_addr,
                                    'end': start_addr,
                                    'insns': [shadow_insn],
                                    'last_insn': shadow_insn
                                }
                                shadow_nodes.append(node)
                        break
                        
        for node in shadow_nodes:
            self.nodes.append(node)
            self.addr_to_node[node['start']] = node
            self.insn_to_block_start[node['start']] = node['start']

    def _add_node(self, start, insns):
        node = {'start': start, 'end': insns[-1].address, 'insns': insns, 'last_insn': insns[-1]}
        self.nodes.append(node)
        self.addr_to_node[start] = node
        for i in insns:
            self.insn_to_block_start[i.address] = start

    def _build_edges(self):
        print("[*] Building edges...")
        insn_map = self.insn_to_block_start
        uncond_jumps = self.unconditional_jumps
        jump_mnems = self.jump_mnems
        branch_prefixes = self.branch_prefixes
        rev_graph = self.reverse_graph

        for node in self.nodes:
            last = node['last_insn']
            mnem = last.mnemonic.lower()
            start_addr = node['start']

            # (Fallthrough)
            if mnem not in uncond_jumps:
                next_addr = last.address + last.size
                if next_addr in insn_map:
                    target = insn_map[next_addr]
                    rev_graph[target].append(start_addr)

            # (Branch/Jump)
            if mnem in jump_mnems or mnem.startswith(branch_prefixes):
                op_str = last.op_str

                match = HEX_PATTERN.search(op_str)

                if match:
                    try:
                        addr = int(match.group(1), 16)
                        if addr in insn_map:
                            target = insn_map[addr]
                            rev_graph[target].append(start_addr)
                    except ValueError:
                        pass

    def build_lazy(self, max_depth: int = 20):
        """Lazy variant of build(): only keep nodes reachable from gadget tails within max_depth.

        Saves memory and edge-building time on large binaries (libc-scale).
        _create_nodes() still runs in full (block boundaries are needed), but the reverse
        graph and final node list are trimmed to the reachable subgraph.
        """
        self._create_nodes()
        self._discover_unaligned_x86()
        tail_starts = self._find_tail_starts()
        reachable = self._bfs_reachable_from_tails(tail_starts, max_depth)
        self._build_edges_filtered(reachable)
        self.nodes = [n for n in self.nodes if n['start'] in reachable]

    def _find_tail_starts(self) -> set:
        """Return block-start addresses that contain a gadget-tail instruction.

        Tails: ret_mnems, indirect unconditional trampolines, and direct-call trampolines
        whose target is inside insn_map. Does not require reverse_graph.
        """
        ret_mnems = self.ret_mnems
        uncond_jumps = self.unconditional_jumps
        call_mnems = self.profile.get("call_mnems", set())
        insn_map = self.insn_to_block_start
        tails = set()

        for n in self.nodes:
            last_insn = n['last_insn']
            mnem = last_insn.mnemonic.lower()
            op_str = last_insn.op_str

            if mnem in ret_mnems:
                tails.add(n['start'])
            elif mnem in uncond_jumps and mnem in self.trampoline_mnems:
                hex_match = HEX_PATTERN.search(op_str)
                if not hex_match:
                    tails.add(n['start'])
                elif mnem in call_mnems:
                    try:
                        target_addr = int(hex_match.group(1), 16)
                        if target_addr in insn_map:
                            tails.add(n['start'])
                    except (ValueError, KeyError):
                        pass
        return tails

    def _bfs_reachable_from_tails(self, tail_starts: set, max_depth: int) -> set:
        """BFS backwards from tail block-starts through the raw block graph.

        Builds a lightweight predecessor map on-the-fly (not stored) and walks backwards
        up to max_depth hops. Returns the set of reachable block-start addresses.
        """
        insn_map = self.insn_to_block_start
        uncond_jumps = self.unconditional_jumps
        jump_mnems = self.jump_mnems
        branch_prefixes = self.branch_prefixes

        pred: dict = defaultdict(set)
        for node in self.nodes:
            last = node['last_insn']
            mnem = last.mnemonic.lower()
            start_addr = node['start']

            if mnem not in uncond_jumps:
                next_addr = last.address + last.size
                if next_addr in insn_map:
                    pred[insn_map[next_addr]].add(start_addr)

            if mnem in jump_mnems or mnem.startswith(branch_prefixes):
                match = HEX_PATTERN.search(last.op_str)
                if match:
                    try:
                        addr = int(match.group(1), 16)
                        if addr in insn_map:
                            pred[insn_map[addr]].add(start_addr)
                    except ValueError:
                        pass

        visited = set(tail_starts)
        frontier = set(tail_starts)
        for _ in range(max_depth):
            next_frontier = set()
            for node_start in frontier:
                for p in pred.get(node_start, ()):
                    if p not in visited:
                        visited.add(p)
                        next_frontier.add(p)
            if not next_frontier:
                break
            frontier = next_frontier
        return visited

    def _build_edges_filtered(self, reachable: set):
        """Like _build_edges() but only adds reverse_graph entries for nodes in reachable."""
        print("[*] Building edges (lazy)...")
        insn_map = self.insn_to_block_start
        uncond_jumps = self.unconditional_jumps
        jump_mnems = self.jump_mnems
        branch_prefixes = self.branch_prefixes
        rev_graph = self.reverse_graph

        for node in self.nodes:
            start_addr = node['start']
            if start_addr not in reachable:
                continue
            last = node['last_insn']
            mnem = last.mnemonic.lower()

            if mnem not in uncond_jumps:
                next_addr = last.address + last.size
                if next_addr in insn_map:
                    target = insn_map[next_addr]
                    if target in reachable:
                        rev_graph[target].append(start_addr)

            if mnem in jump_mnems or mnem.startswith(branch_prefixes):
                match = HEX_PATTERN.search(last.op_str)
                if match:
                    try:
                        addr = int(match.group(1), 16)
                        if addr in insn_map:
                            target = insn_map[addr]
                            if target in reachable:
                                rev_graph[target].append(start_addr)
                    except ValueError:
                        pass

    def get_gadget_tails(self):
        ret_mnems = self.ret_mnems
        uncond_jumps = self.unconditional_jumps
        call_mnems = self.profile.get("call_mnems", set())
        insn_map = self.insn_to_block_start
        tails = []

        for n in self.nodes:
            last_insn = n['last_insn']
            mnem = last_insn.mnemonic.lower()
            op_str = last_insn.op_str

            if mnem in ret_mnems:
                tails.append(n)

            elif mnem in uncond_jumps and mnem in self.trampoline_mnems:
                hex_match = HEX_PATTERN.search(op_str)

                if not hex_match:
                    tails.append(n)

                elif mnem in call_mnems:
                    try:
                        target_addr = int(hex_match.group(1), 16)
                        if target_addr in insn_map:
                            n['direct_call_target'] = target_addr
                            tails.append(n)
                    except (ValueError, KeyError):
                        pass

        return tails

    def _bfs_reachable_from_tails(self, tail_starts: set, max_depth: int) -> set:
        """BFS backwards from tail block-starts through the raw block graph.

        Builds a lightweight predecessor map on-the-fly (not stored) and walks backwards
        up to max_depth hops. Returns the set of reachable block-start addresses.
        """
        insn_map = self.insn_to_block_start
        uncond_jumps = self.unconditional_jumps
        jump_mnems = self.jump_mnems
        branch_prefixes = self.branch_prefixes

        pred: dict = defaultdict(set)
        for node in self.nodes:
            last = node['last_insn']
            mnem = last.mnemonic.lower()
            start_addr = node['start']

            if mnem not in uncond_jumps:
                next_addr = last.address + last.size
                if next_addr in insn_map:
                    pred[insn_map[next_addr]].add(start_addr)

            if mnem in jump_mnems or mnem.startswith(branch_prefixes):
                match = HEX_PATTERN.search(last.op_str)
                if match:
                    try:
                        addr = int(match.group(1), 16)
                        if addr in insn_map:
                            pred[insn_map[addr]].add(start_addr)
                    except ValueError:
                        pass

        visited = set(tail_starts)
        frontier = set(tail_starts)
        for _ in range(max_depth):
            next_frontier = set()
            for node_start in frontier:
                for p in pred.get(node_start, ()):
                    if p not in visited:
                        visited.add(p)
                        next_frontier.add(p)
            if not next_frontier:
                break
            frontier = next_frontier
        return visited

    def _build_edges_filtered(self, reachable: set):
        """Like _build_edges() but only adds reverse_graph entries for nodes in reachable."""
        print("[*] Building edges (lazy)...")
        insn_map = self.insn_to_block_start
        uncond_jumps = self.unconditional_jumps
        jump_mnems = self.jump_mnems
        branch_prefixes = self.branch_prefixes
        rev_graph = self.reverse_graph

        for node in self.nodes:
            start_addr = node['start']
            if start_addr not in reachable:
                continue
            last = node['last_insn']
            mnem = last.mnemonic.lower()

            if mnem not in uncond_jumps:
                next_addr = last.address + last.size
                if next_addr in insn_map:
                    target = insn_map[next_addr]
                    if target in reachable:
                        rev_graph[target].append(start_addr)

            if mnem in jump_mnems or mnem.startswith(branch_prefixes):
                match = HEX_PATTERN.search(last.op_str)
                if match:
                    try:
                        addr = int(match.group(1), 16)
                        if addr in insn_map:
                            target = insn_map[addr]
                            if target in reachable:
                                rev_graph[target].append(start_addr)
                    except ValueError:
                        pass

    def get_gadget_tails(self):
        ret_mnems = self.ret_mnems
        uncond_jumps = self.unconditional_jumps
        call_mnems = self.profile.get("call_mnems", set())
        insn_map = self.insn_to_block_start
        tails = []

        for n in self.nodes:
            last_insn = n['last_insn']
            mnem = last_insn.mnemonic.lower()
            op_str = last_insn.op_str

            if mnem in ret_mnems:
                tails.append(n)

            elif mnem in uncond_jumps and mnem in self.trampoline_mnems:
                hex_match = HEX_PATTERN.search(op_str)

                if not hex_match:
                    tails.append(n)

                elif mnem in call_mnems:
                    try:
                        target_addr = int(hex_match.group(1), 16)
                        if target_addr in insn_map:
                            n['direct_call_target'] = target_addr
                            tails.append(n)
                    except (ValueError, KeyError):
                        pass

        return tails
