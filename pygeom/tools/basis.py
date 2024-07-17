from typing import TYPE_CHECKING, Union

from numpy import float64, logical_and, where, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray
    Numeric = Union[float64, NDArray[float64]]

def basis_functions(p: int, k: 'NDArray[float64]',
                    u: 'NDArray[float64]') -> 'NDArray[float64]':
    m = k.size - 1
    n = u.size
    Nu = zeros((m, n), dtype=float64)
    for i in range(m):
        check1 = u >= k[i]
        check2 = u < k[i + 1]
        if i == 0:
            check = check2
        elif i == m - 1:
            check = check1
        else:
            check = logical_and(check1, check2)
        Nu[i, :] = where(check, 1.0, 0.0)
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
                Nu[i, :] = slope1*prevNu[i, :] + slope2*prevNu[i + 1, :]
    return Nu

def basis_derivatives(p: int, k: 'NDArray[float64]',
                      u: 'NDArray[float64]') -> 'NDArray[float64]':
    m = k.size - 1
    n = u.size
    Nu = basis_functions(p - 1, k, u)
    dNu = zeros((m-p, n), dtype=float64)
    for i in range(m-p):
        Dk1 = k[i + p] - k[i]
        Dk2 = k[i + p + 1] - k[i + 1]
        slope1 = 0.0
        if Dk1 != 0.0:
            slope1 = p/Dk1
        slope2 = 0.0
        if Dk2 != 0.0:
            slope2 = p/Dk2
        dNu[i, :] = slope1*Nu[i, :] - slope2*Nu[i + 1, :]
    return dNu
