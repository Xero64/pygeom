from typing import TYPE_CHECKING, Union

from numpy import float64, linspace, logical_and, where, zeros, unique, isscalar

if TYPE_CHECKING:
    from numpy.typing import NDArray
    Numeric = Union[float64, NDArray[float64]]

def basis_functions(p: int, k: 'NDArray[float64]',
                    u: 'NDArray[float64]') -> 'NDArray[float64]':
    m = k.size - 1
    if isscalar(u):
        Nu = zeros(m, dtype=float64)
    else:
        n = u.size
        Nu = zeros((m, n), dtype=float64)
    for i in range(m):
        check1 = u >= k[i]
        check2 = u < k[i + 1]
        check = logical_and(check1, check2)
        Nu[i] = where(check, 1.0, 0.0)
    u_check = u == k[-1]
    Nu[-1] = where(u_check, 1.0, Nu[-1])
    if p > 0:
        for j in range(1, p + 1):
            prevNu = Nu.copy()
            Nu = zeros((m - j, n), dtype=float64)
            for i in range(m - j):
                Dk1 = k[i + j] - k[i]
                Dk2 = k[i + j + 1] - k[i + 1]
                slope1 = 1.0
                if Dk1 != 0.0:
                    slope1 = (u - k[i])/Dk1
                slope2 = 1.0
                if Dk2 != 0.0:
                    slope2 = (k[i + j + 1] - u)/Dk2
                Nu[i] = slope1*prevNu[i] + slope2*prevNu[i + 1]
    return Nu

def basis_derivatives(p: int, k: 'NDArray[float64]',
                      u: 'NDArray[float64]') -> 'NDArray[float64]':
    m = k.size - 1
    if isscalar(u):
        dNu = zeros(m - p, dtype=float64)
    else:
        n = u.size
        dNu = zeros((m - p, n), dtype=float64)
    kp = k.copy()
    if kp[-1] == kp[-2]:
        kp = kp[:-1]
    pNu = p*basis_functions(p - 1, kp, u)
    for i in range(m - p):
        Dk1 = k[i + p] - k[i]
        Dk2 = k[i + p + 1] - k[i + 1]
        if Dk1 != 0.0:
            dNu[i] += pNu[i]/Dk1
        if Dk2 != 0.0:
            dNu[i] -= pNu[i + 1]/Dk2
    return dNu

def basis_second_derivatives(p: int, k: 'NDArray[float64]',
                             u: 'NDArray[float64]') -> 'NDArray[float64]':
    m = k.size - 1
    if isscalar(u):
        d2Nu = zeros(m - p, dtype=float64)
    else:
        n = u.size
        d2Nu = zeros((m - p, n), dtype=float64)
    kp = k.copy()
    if kp[-1] == kp[-2]:
        kp = kp[:-1]
    pdNu = p*basis_derivatives(p - 1, kp, u)
    for i in range(m - p):
        Dk1 = k[i + p] - k[i]
        Dk2 = k[i + p + 1] - k[i + 1]
        if Dk1 != 0.0:
            d2Nu[i] += pdNu[i]/Dk1
        if Dk2 != 0.0:
            d2Nu[i] -= pdNu[i + 1]/Dk2
    return d2Nu

def default_knots(numpnt: int, degree: int) -> 'NDArray[float64]':
    numk = (numpnt - 1) // degree
    link = linspace(0.0, 1.0, numk + 1, dtype=float64)
    knots = zeros(numk*2, dtype=float64)
    knots[::2] = link[:-1]
    knots[1::2] = link[1:]
    return knots

def knot_linspace(num: int, knots: 'NDArray[float64]') -> 'NDArray[float64]':
    unknots = unique(knots)
    m = unknots.size - 1
    u = zeros(num*m + 1, dtype=float64)
    for i in range(m):
        ip1 = i + 1
        a = i*num
        b = ip1*num
        ka = unknots[i]
        kb = unknots[ip1]
        u[a:b] = linspace(ka, kb, num, dtype=float64, endpoint=False)
    u[-1] = unknots[-1]
    return u
