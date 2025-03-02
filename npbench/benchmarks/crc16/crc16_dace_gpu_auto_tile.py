import numpy as np
import dace

N = dace.symbol('N')
poly: dace.uint16 = 0x8408


# Adapted from https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6
@dace.program
def _crc16(data: dace.uint8[N]):
    '''
    CRC-16-CCITT Algorithm
    '''
    crc: dace.uint16 = 0xFFFF
    for i in range(N):
        b = data[i]
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = (~crc & 0xFFFF)
    crc = (crc << 8) | ((crc >> 8) & 0xFF)
    r = crc & 0xFFFF
    return r

_best_config = None

def autotuner(data):
    global _best_config
    if _best_config is not None:
        return

    def get_max_ndim(inputs):
        return max((arg.ndim for arg in inputs if hasattr(arg, "ndim")), default=0)

    from npbench.infrastructure.dace_gpu_auto_tile_framework import DaceGPUAutoTileFramework
    _best_config = DaceGPUAutoTileFramework.autotune(
        _crc16.to_sdfg(),
        {"data": data},
        dims=get_max_ndim([data])
    )

def crc16(data):
    global _best_config
    r = _best_config(data)
    return r
