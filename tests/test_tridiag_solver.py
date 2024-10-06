from numpy import asarray, diag, isclose
from numpy.linalg import solve

from pygeom.tools.solvers import tridiag_solver

a = asarray([3.5, 2.2, 1.4])
b = asarray([4.3, 5.7, -6.1, 4.1])
c = asarray([1.8, -2.1, 3.3])

d = asarray([[1.2, 2.3, 3.4, -1.2],
             [-3.1, 0.0, 2.5, 4.0],
             [3.8, -2.2, -1.1, 0.0],
             [5.6, 12.1, -6.5, -2.1]])

e = diag(b) + diag(a, -1) + diag(c, 1)

res1 = tridiag_solver(a, b, c, d)
res2 = solve(e, d)

def test_tridiag_solver():
    assert isclose(res1, res2, atol=1e-12).all()
