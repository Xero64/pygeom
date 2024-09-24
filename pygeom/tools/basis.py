from typing import TYPE_CHECKING

from numpy import linspace, logical_and, shape, unique, where, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray

def basis_functions(p: int, k: 'NDArray', u: 'NDArray') -> 'NDArray':
    m = k.size - 1
    ushp = shape(u)
    Nushp = (m, *ushp)
    Nu = zeros(Nushp)
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
            Nushp = (m - j, *ushp)
            Nu = zeros(Nushp)
            for i in range(m - j):
                Dk1 = k[i + j] - k[i]
                Dk2 = k[i + 1 + j] - k[i + 1]
                slope1 = 1.0
                if Dk1 != 0.0:
                    slope1 = (u - k[i])/Dk1
                slope2 = 1.0
                if Dk2 != 0.0:
                    slope2 = (k[i + j + 1] - u)/Dk2
                Nu[i] = slope1*prevNu[i] + slope2*prevNu[i + 1]
    return Nu

def basis_first_derivatives(p: int, k: 'NDArray', u: 'NDArray') -> 'NDArray':
    m = k.size - 1
    ushp = shape(u)
    dNushp = (m - p, *ushp)
    dNu = zeros(dNushp)
    kp = k.copy()
    if kp[-1] == kp[-2]:
        kp = kp[:-1]
    pNu = p*basis_functions(p - 1, kp, u)
    for i in range(m - p):
        Dk1 = k[i + p] - k[i]
        Dk2 = k[i + 1 + p] - k[i + 1]
        if Dk1 != 0.0:
            dNu[i] += pNu[i]/Dk1
        if Dk2 != 0.0:
            dNu[i] -= pNu[i + 1]/Dk2
    return dNu

def basis_second_derivatives(p: int, k: 'NDArray', u: 'NDArray') -> 'NDArray':
    m = k.size - 1
    ushp = shape(u)
    d2Nushp = (m - p, *ushp)
    d2Nu = zeros(d2Nushp)
    kp = k.copy()
    if kp[-1] == kp[-2]:
        kp = kp[:-1]
    pdNu = p*basis_first_derivatives(p - 1, kp, u)
    for i in range(m - p):
        Dk1 = k[i + p] - k[i]
        Dk2 = k[i + 1 + p] - k[i + 1]
        if Dk1 != 0.0:
            d2Nu[i] += pdNu[i]/Dk1
        if Dk2 != 0.0:
            d2Nu[i] -= pdNu[i + 1]/Dk2
    return d2Nu

def knots_from_spacing(spacing: 'NDArray', degree: int) -> 'NDArray':
    fullknots = spacing.repeat(degree)
    if degree > 1:
        fullknots = fullknots[degree-1:-degree+1]
    return fullknots

def default_knots(numpnt: int, degree: int) -> 'NDArray':
    if degree < 1:
        raise ValueError('default_knots degree may not be less than 1.')
    numknot = (numpnt - 1) // degree
    spacing = linspace(0.0, 1.0, numknot + 1)
    return knots_from_spacing(spacing, degree)

def knot_linspace(num: int, knots: 'NDArray') -> 'NDArray':
    unknots = unique(knots)
    m = unknots.size - 1
    u = zeros(num*m + 1, knots.dtype)
    for i in range(m):
        ip1 = i + 1
        a = i*num
        b = ip1*num
        ka = unknots[i]
        kb = unknots[ip1]
        u[a:b] = linspace(ka, kb, num, endpoint=False)
    u[-1] = unknots[-1]
    return u
