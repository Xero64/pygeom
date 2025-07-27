from typing import TYPE_CHECKING

from numpy import full, logical_and, ndarray

from ..tools.basis import knot_linspace

if TYPE_CHECKING:
    from numpy.typing import NDArray
    BCLike = tuple[tuple[int, float], tuple[int, float]] | None

BCSTR1 = ('quadratic', 'not-a-knot', 'natural', 'clamped', 'periodic')
BCSTR2 = ('quadratic', 'not-a-knot', 'natural', 'clamped')

class LinearSpline1D():
    u"""This class stores a 1D parametric linear spline."""
    s: 'NDArray' = None
    r: 'NDArray' = None
    _Dr: 'NDArray' = None
    _Ds: 'NDArray' = None

    def __init__(self, s: 'NDArray', r: 'NDArray',
                 validate: bool = True) -> None:
        u"""This function initialises the object."""
        self.s = s
        self.r = r
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        if not isinstance(self.s, ndarray):
            raise ValueError('Input s must be a ndarray.')
        if self.s.ndim != 1:
            raise ValueError('Input s must be a 1D ndarray.')
        if not isinstance(self.r, ndarray):
            raise ValueError('Input r must be a ndarray.')
        if self.r.ndim != 1:
            raise ValueError('Input r must be a 1D ndarray.')

    @property
    def Dr(self) -> 'NDArray':
        if self._Dr is None:
            self._Dr = self.r[1:] - self.r[:-1]
        return self._Dr

    @property
    def Ds(self) -> 'NDArray':
        if self._Ds is None:
            self._Ds = self.s[1:] - self.s[:-1]
        return self._Ds

    def evaluate_points_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the spline at a given s."""
        r = full(s.shape, float('nan'))
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            ra = self.r[a]
            rb = self.r[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            Av = (sb - sv)/Dsi
            Bv = (sv - sa)/Dsi
            r[s_check] = ra*Av + rb*Bv
        return r

    def evaluate_t(self, num: int) -> 'NDArray':
        return knot_linspace(num, self.s)

    def evaluate_points(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        points = self.evaluate_points_at_t(s)
        return points

    def __repr__(self):
        return '<LinearSpline1D>'
