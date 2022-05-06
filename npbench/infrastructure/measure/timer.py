import timeit
import numpy as np

from npbench.infrastructure.measure.metric import Metric

class Timer(Metric):

    timeit_tmpl = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        {stmt}
    _t1 = _timer()
    return _t1 - _t0, {output}
"""

    def __init__(self) -> None:
        super().__init__()

    def benchmark(
        self,
        stmt,
        setup="pass",
        out_text="",
        repeat=1,
        context={},
        output=None,
        verbose=True
    ):
        timeit.template = Timer.timeit_tmpl.format(
            init='{init}',
            setup='{setup}',
            stmt='{stmt}',
            output=output
        )

        ldict = {**context}
        output = timeit.repeat(
            stmt,
            setup=setup,
            repeat=repeat,
            number=1,
            globals=ldict
        )
        
        res = output[0][1]
        raw_time_list = [a for a, _ in output]
        raw_time = np.median(raw_time_list)
        ms_time = _time_to_ms(raw_time)
        
        if verbose:
            print("{}: {}ms".format(out_text, ms_time))
        
        return res, raw_time_list

def _time_to_ms(raw: float) -> int:
    return int(round(raw * 1000))
