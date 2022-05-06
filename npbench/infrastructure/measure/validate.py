import numpy as np

from numbers import Number
from typing import Union

def validate(ref, val, framework="Unknown"):
    if not isinstance(ref, (tuple, list)):
        ref = [ref]
    if not isinstance(val, (tuple, list)):
        val = [val]

    valid = True
    for r, v in zip(ref, val):
        if np.allclose(r, v):
           continue

        try:
            import cupy
            if isinstance(v, cupy.ndarray):
                relerror = _relative_error(r, cupy.asnumpy(v))
            else:
                relerror = _relative_error(r, v)
        except Exception:
            relerror = _relative_error(r, v)
        
        if relerror < 1e-05:
            continue
        
        valid = False
        print("Relative error: {}".format(relerror))
        # return False
    
    if not valid:
        print("{} did not validate!".format(framework))
    
    return valid

def _relative_error(ref: Union[Number, np.ndarray], val: Union[Number, np.ndarray]) -> float:
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)
