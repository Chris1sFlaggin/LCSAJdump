"""
semantic_features.py — Symbolic execution for deep semantic feature extraction.
(Deep Clean Version: Aggressive Memory Management)
"""

from __future__ import annotations
import logging
import gc
import os

# Zittiamo angr
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)

_project_cache = {}

def extract_semantic_features(binary_path: str, gadget_addr: int, gadget_size: int, arch: str) -> dict:
    feats = {"sm_controls_arg_reg": 0, "sm_stack_pivot_size": -999, "sm_writes_memory": 0}
    
    try:
        import angr
        import claripy
    except ImportError:
        return feats

    # --- AGGRESSIVE CLEANUP ---
    # Se cambiamo binario, svuotiamo tutto il vecchio progetto dalla RAM
    global _project_cache
    if _project_cache and binary_path not in _project_cache:
        _project_cache.clear()
        gc.collect() 

    try:
        if binary_path not in _project_cache:
            _project_cache[binary_path] = angr.Project(
                binary_path, 
                auto_load_libs=False,
                use_sim_procedures=False
            )
        proj = _project_cache[binary_path]
        
        arg_reg = "rdi" if arch == "x86_64" else "eax" if arch == "x86_32" else "x0" if arch == "arm64" else "a0"
        sp_reg = "rsp" if arch == "x86_64" else "esp" if arch == "x86_32" else "sp"

        state = proj.factory.blank_state(
            addr=gadget_addr,
            add_options={
                angr.options.LAZY_SOLVES, 
                angr.options.ZERO_FILL_UNCONSTRAINED_MEMORY, 
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS
            }
        )
        
        # Timeout nativo Z3 a 5 secondi
        state.solver._solver.set("timeout", 5000)
        
        # Inizializzazione stack
        sp = state.solver.eval(state.regs.sp)
        for i in range(16):
            sym_var = claripy.BVS(f"stack_var_{i}", 64)
            state.memory.store(sp + (i * (proj.arch.bits // 8)), sym_var)

        initial_sp = state.solver.eval(getattr(state.regs, sp_reg))
        simgr = proj.factory.simgr(state, save_unconstrained=True)
        
        # Step limitati per non divergere
        steps = 0
        while (simgr.active or simgr.unconstrained) and steps < 4:
            if len(simgr.active) > 1:
                simgr.active = [simgr.active[0]]
            simgr.step()
            steps += 1
            if simgr.unconstrained:
                break
            
        final_state = None
        if simgr.unconstrained: final_state = simgr.unconstrained[0]
        elif simgr.deadended: final_state = simgr.deadended[0]
        elif simgr.active: final_state = simgr.active[0]
            
        if final_state:
            try:
                sp_val = getattr(final_state.regs, sp_reg)
                if not sp_val.symbolic:
                    final_sp = final_state.solver.eval(sp_val)
                    feats["sm_stack_pivot_size"] = final_sp - initial_sp 
                else:
                    feats["sm_stack_pivot_size"] = -1 
                
                arg_val = getattr(final_state.regs, arg_reg)
                if arg_val.symbolic:
                    feats["sm_controls_arg_reg"] = 1
                    
                for action in final_state.history.actions:
                    if action.type == "mem" and action.action == "write":
                        feats["sm_writes_memory"] = 1
                        break
            except:
                pass
        
        # Pulizia post-gadget
        del state
        del simgr
                
    except Exception:
        pass
        
    return feats