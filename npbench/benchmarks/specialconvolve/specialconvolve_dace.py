# Original code: https://stackoverflow.com/a/2205095
# License: CC BY-SA 2.5 (https://creativecommons.org/licenses/by-sa/2.5/legalcode)

import dace
import numpy as np

M, N = (dace.symbol(s) for s in ('M', 'N'))

@dace.program
def specialconvolve(a):
    rowconvol = a[1:-1,:] + a[:-2,:] + a[2:,:]
    colconvol = rowconvol[:,1:-1] + rowconvol[:,:-2] + rowconvol[:,2:] - 9*a[1:-1,1:-1]
    return colconvol
