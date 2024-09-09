from math import comb
from typing import TYPE_CHECKING, List, Union

from numpy import asarray, zeros

if TYPE_CHECKING:
    from sympy import Symbol, Add
    from numpy.typing import NDArray

def bernstein_polynomial(n: int, i: int, t: 'NDArray') -> 'NDArray':
    cmb = comb(n, i)
    omt = 1 - t
    nmi = n - i
    return cmb*t**i*omt**nmi

def bernstein_polynomials(n: int, t: 'NDArray') -> 'NDArray':
    t = asarray(t)
    polys = zeros((n + 1, t.size))
    for i in range(n + 1):
        polys[i, :] = bernstein_polynomial(n, i, t)
    if t.size == 1:
        polys = polys.ravel()
    return polys

def bernstein_derivative(n: int, i: int, t: 'NDArray') -> 'NDArray':
    cmb = comb(n, i)
    omt = 1 - t
    nmi = n - i
    im1 = i - 1
    nmim1 = nmi - 1
    term1 = 0.0
    if im1 >= 0:
        term1 = i*t**im1*omt**nmi
    term2 = 0.0
    if nmim1 >= 0:
        term2 = t**i*nmi*omt**nmim1
    return cmb*(term1 - term2)

def bernstein_derivatives(n: int, t: 'NDArray') -> 'NDArray':
    t = asarray(t)
    polyders = zeros((n + 1, t.size))
    for i in range(n + 1):
        polyders[i, :] = bernstein_derivative(n, i, t)
    if t.size == 1:
        polyders = polyders.ravel()
    return polyders

def symbolic_bernstein_polynomials(n: int, t: 'Symbol') -> List['Add']:
    polys = []
    for i in range(n + 1):
        polys.append(bernstein_polynomial(n, i, t))
    return polys

def symbolic_bernstein_derivatives(n: int, t: 'Symbol') -> List['Add']:
    polyders = []
    for i in range(n + 1):
        polyders.append(bernstein_derivative(n, i, t))
    return polyders
