# Original code: https://stackoverflow.com/a/2205095
# License: CC BY-SA 2.5 (https://creativecommons.org/licenses/by-sa/2.5/legalcode)

import numpy as np

def specialconvolve(a):
    rowconvol = a[1:-1,:] + a[:-2,:] + a[2:,:]
    colconvol = rowconvol[:,1:-1] + rowconvol[:,:-2] + rowconvol[:,2:] - 9*a[1:-1,1:-1]
    return colconvol
