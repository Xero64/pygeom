from math import comb
from typing import TYPE_CHECKING

from numpy import empty, size, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sympy import Symbol

def bernstein_polynomial(n: int, i: int, t: 'NDArray') -> 'NDArray':
    cmb = comb(n, i)
    omt = 1 - t
    nmi = n - i
    return cmb*t**i*omt**nmi

def bernstein_first_derivative(n: int, i: int, t: 'NDArray') -> 'NDArray':
    cmb = comb(n, i)
    omt = 1 - t
    nmi = n - i
    im1 = i - 1
    nmim1 = nmi - 1
    terms = 0
    if im1 >= 0:
        dtidt = i*t**im1
        omtnmi = omt**nmi
        terms += dtidt*omtnmi
    if nmim1 >= 0:
        ti = t**i
        domtnmidt = -nmi*omt**nmim1
        terms += ti*domtnmidt
    return cmb*terms

def bernstein_second_derivative(n: int, i: int, t: 'NDArray') -> 'NDArray':
    cmb = comb(n, i)
    omt = 1 - t
    nmi = n - i
    im1 = i - 1
    nmim1 = nmi - 1
    im2 = i - 2
    nmim2 = nmi - 2
    terms = 0
    if im2 >= 0:
        d2tidt2 = i*im1*t**im2
        omtnmi = omt**nmi
        terms += d2tidt2*omtnmi
    if im1 >= 0 and nmim1 >= 0:
        dtidt = i*t**im1
        domtnmidt = -nmi*omt**nmim1
        terms += 2*dtidt*domtnmidt
    if nmim2 >= 0:
        ti = t**i
        d2omtnmi2dt2 = nmi*nmim1*omt**nmim2
        terms += ti*d2omtnmi2dt2
    return cmb*terms

def bernstein_polynomials(n: int, t: 'NDArray') -> 'NDArray':
    size_t = size(t)
    polys = zeros((n + 1, size_t), dtype=type(t))
    for i in range(n + 1):
        polys[i, :] = bernstein_polynomial(n, i, t)
    if size_t == 1:
        polys = polys.ravel()
    return polys

def bernstein_first_derivatives(n: int, t: 'NDArray') -> 'NDArray':
    size_t = size(t)
    polyder1s = zeros((n + 1, size_t))
    for i in range(n + 1):
        polyder1s[i, :] = bernstein_first_derivative(n, i, t)
    if size_t == 1:
        polyder1s = polyder1s.ravel()
    return polyder1s

def bernstein_second_derivatives(n: int, t: 'NDArray') -> 'NDArray':
    size_t = size(t)
    polyder2s = zeros((n + 1, size_t))
    for i in range(n + 1):
        polyder2s[i, :] = bernstein_second_derivative(n, i, t)
    if size_t == 1:
        polyder2s = polyder2s.ravel()
    return polyder2s

def symbolic_bernstein_polynomials(n: int, t: 'Symbol') ->  'NDArray':
    polys = empty(n + 1, dtype=object)
    for i in range(n + 1):
        polys[i] = bernstein_polynomial(n, i, t)
    return polys

def symbolic_bernstein_first_derivatives(n: int, t: 'Symbol') ->  'NDArray':
    polyder1s = empty(n + 1, dtype=object)
    for i in range(n + 1):
        polyder1s[i] = bernstein_first_derivative(n, i, t)
    return polyder1s

def symbolic_bernstein_second_derivatives(n: int, t: 'Symbol') ->  'NDArray':
    polyder2s = empty(n + 1, dtype=object)
    for i in range(n + 1):
        polyder2s[i] = bernstein_second_derivative(n, i, t)
    return polyder2s
