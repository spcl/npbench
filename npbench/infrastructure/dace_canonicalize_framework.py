# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import importlib
import traceback

import numpy as np

from npbench.infrastructure import Benchmark
from npbench.infrastructure.dace_framework import DaceFramework
from typing import Callable, Sequence, Tuple


class DaceCanonicalizeFramework(DaceFramework):
    """Runs DaCe's ``canonicalize`` pipeline instead of ``auto_optimize``.

    Reuses the ``*_dace.py`` benchmark implementations (``postfix`` ``dace``) and every
    calling convention of :class:`DaceFramework`; only the SDFG-optimization step differs.
    The CPU and GPU variants share this class and differ only by the ``arch`` field in
    their framework JSON, exactly as ``dace_cpu`` / ``dace_gpu`` share :class:`DaceFramework`.

    Contrast with ``dace_cpu`` / ``dace_gpu`` (WCR-config OFF): the canonicalize pipeline
    leaves ``sdfg.openmp_array_reductions = True`` (WCR-config ON), so a whole-buffer WCR
    accumulator of a parallel map lowers to an OpenMP ``reduction(op:A[0:n])`` array-section
    clause instead of per-element atomics.
    """

    def copy_func(self) -> Callable:
        """Copy inputs to the device: cupy for the GPU variant, ``np.copy`` otherwise."""
        if self.info["arch"] == "gpu":
            import cupy

            def cp_copy_func(arr):
                darr = cupy.asarray(arr)
                cupy.cuda.stream.get_current_stream().synchronize()
                return darr

            return cp_copy_func
        return np.copy

    def implementations(self, bench: Benchmark) -> Sequence[Tuple[Callable, str]]:
        """Build the SDFG for ``bench``, run the canonicalize pipeline + target
        finalization, turn the WCR-config ON, then compile."""
        import dace  # noqa: F401
        from dace.transformation.passes.canonicalize import canonicalize
        from dace.transformation.passes.canonicalize.finalize import finalize_for_target

        module_pypath = "npbench.benchmarks.{r}.{m}".format(r=bench.info["relative_path"].replace('/', '.'),
                                                            m=bench.info["module_name"])
        postfix = self.info["postfix"] if "postfix" in self.info.keys() else self.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        try:
            module = importlib.import_module(module_str)
            ct_impl = vars(module)[func_str]
        except Exception as e:
            print("Failed to load the DaCe implementation.")
            raise e

        target = "gpu" if self.info["arch"] == "gpu" else "cpu"

        sdfg = ct_impl.to_sdfg(simplify=True)
        sdfg._name = "canonicalize"

        canonicalize(sdfg,
                     validate=True,
                     target=target,
                     peel_limit=4,
                     break_anti_dependence=True,
                     interchange_carry_with_map=True,
                     scatter_to_guarded_maps=True)
        finalize_for_target(sdfg, target)

        # WCR-config ON. ``canonicalize`` already sets this on every nested SDFG at the end of
        # the pipeline; set it again explicitly right before compile for clarity.
        for nested in sdfg.all_sdfgs_recursive():
            nested.openmp_array_reductions = True

        implementations = []
        try:
            dc_exec = sdfg.compile()
            implementations.append((dc_exec, sdfg._name))
        except Exception as e:
            print("Failed to compile DaCe {a} canonicalize implementation.".format(a=self.info["arch"]))
            print(e)
            traceback.print_exc()

        return implementations
