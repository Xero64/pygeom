from math import comb
from typing import TYPE_CHECKING, List, Union

from numpy import asarray, float64, zeros

if TYPE_CHECKING:
    from sympy import Symbol, Add
    from numpy.typing import NDArray
    Numeric = Union[float64, NDArray[float64]]

def bernstein_poly(n: int, i: int, t: 'Numeric') -> 'Numeric':
    omt = 1 - t
    nmi = n - i
    return comb(n, i)*t**i*omt**nmi

def bernstein_polys(n: int, t: 'Numeric') -> 'NDArray[float64]':
    t = asarray(t, dtype=float64)
    polys = zeros((n + 1, t.size), dtype=float64)
    for i in range(n + 1):
        polys[i, :] = bernstein_poly(n, i, t)
    if t.size == 1:
        polys = polys.flatten()
    return polys

def bernstein_poly_derivative(n: int, i: int, t: 'Numeric') -> 'Numeric':
    omt = 1 - t
    nmi = n - i
    im1 = i - 1
    nmim1 = nmi - 1
    return comb(n, i)*(i*t**im1*omt**nmi - t**i*nmi*omt**nmim1)

def bernstein_poly_derivatives(n: int, t: 'Numeric') -> 'NDArray[float64]':
    t = asarray(t, dtype=float64)
    polyders = zeros((n + 1, t.size), dtype=float64)
    for i in range(n + 1):
        polyders[i, :] = bernstein_poly_derivative(n, i, t)
    if t.size == 1:
        polyders = polyders.flatten()
    return polyders

def symbolic_bernstein_polys(n: int, t: 'Symbol') -> List['Add']:
    polys = []
    for i in range(n + 1):
        polys.append(bernstein_poly(n, i, t))
    return polys

def symbolic_bernstein_poly_derivatives(n: int, t: 'Symbol') -> List['Add']:
    polyders = []
    for i in range(n + 1):
        polyders.append(bernstein_poly_derivative(n, i, t))
    return polyders
