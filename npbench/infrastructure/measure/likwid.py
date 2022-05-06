import timeit

from npbench.infrastructure.measure.metric import Metric
from npbench.infrastructure import Framework

from typing import Any, Callable, Dict, Sequence, Tuple

class Likwid(Metric):

    template = """
def inner(_it, _timer{init}):
    {setup}

    import pylikwid

    pylikwid.markerinit()
    pylikwid.markerthreadinit()

    pylikwid.markerstartregion("Compute")

    {stmt}

    pylikwid.markerstopregion("Compute")

    nr_events, report, time, count = pylikwid.markergetregion("Compute") 
    pylikwid.markerclose()

    return report, {output}
"""

    def __init__(self) -> None:
        super().__init__()

    def execute(self, bench, frmwrk: Framework, impl: Callable, impl_name: str, bdata: Dict[str, Any], **kwargs) -> Tuple[Any, Sequence[float]]:
        report_str = frmwrk.info["full_name"] + " - " + impl_name

        try:
            copy = frmwrk.copy_func()
            setup_str = frmwrk.setup_str(bench, impl)
            exec_str = frmwrk.exec_str(bench, impl)
        except Exception as e:
            print("Failed to load the {} implementation.".format(report_str))
            print(e)
            return None, None

        ldict = {'__npb_impl': impl, '__npb_copy': copy, **bdata}

        try:
            out, counters = self.benchmark(
                stmt=exec_str,
                setup=setup_str,
                context=ldict,
                output='__npb_result'
            )
        except Exception as e:
            print("Failed to execute the {} implementation.".format(report_str))
            print(e)
            return None, None
        
        if out is not None:
            if isinstance(out, (tuple, list)):
                out = list(out)
            else:
                out = [out]
        else:
            out = []
        if "out_args" in bench.info.keys():
            out += [ldict[a] for a in self.frmwrk.args(bench)]
        
        return out, counters

    def benchmark(
        self,
        stmt,
        setup="pass",
        output=None,
        context=None,
        **kwargs
    ):
        timeit.template = Likwid.template.format(
            init='{init}',
            setup='{setup}',
            stmt='{stmt}',
            output=output
        )

        ldict = {**context}
        output = timeit.repeat(
            stmt,
            setup=setup,
            repeat=1,
            number=1,
            globals=ldict
        )

        report, out = output[0]
        return out, report
