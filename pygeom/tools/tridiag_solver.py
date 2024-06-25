from typing import TYPE_CHECKING

from numpy import float64, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray

def tridiag_solver(a: 'NDArray[float64]', b: 'NDArray[float64]', c: 'NDArray[float64]',
                   d: 'NDArray[float64]') -> 'NDArray[float64]':

    num = b.size
    gm = zeros(num, dtype=float64)
    r = zeros(d.shape, dtype=float64)

    bt = b[0]
    r[0, :] = (d[0, :]-a[0]*r[0, :])/bt

    for i in range(1, num):
        gm[i] = c[i-1]/bt
        bt = b[i]-a[i-1]*gm[i]
        r[i, :] = (d[i, :]-a[i-1]*r[i-1, :])/bt

    for i in range(num-2, -1, -1):
        r[i, :] -= gm[i+1]*r[i+1, :]

    return r
