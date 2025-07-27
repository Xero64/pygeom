from typing import TYPE_CHECKING, Any

from numpy import (arange, asarray, concatenate, diag, hstack, linspace, ones,
                   setdiff1d, vstack, zeros)
from numpy.linalg import lstsq, norm

from ..geom2d import Vector2D
from .solvers import solve_clsq

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..geom2d import BSplineCurve2D, Vector2D


def bspline2d_lstsq_fit(bspline: 'BSplineCurve2D', pnts_target: Vector2D,
                        **kwargs: dict[str, Any]) -> 'BSplineCurve2D':

    # Sizes
    numtgt = pnts_target.size
    numctl = bspline.ctlpnts.size

    # Free t values
    ind_t_f: 'NDArray' = kwargs.get('ind_t_f', arange(1, numtgt - 1))

    # Constrained point indices
    ind_p_c: 'NDArray' = kwargs.get('ind_p_c', asarray([0, numtgt - 1]))

    # Constrained tangents at point indices
    tgts_dict: dict[int, Vector2D] = kwargs.get('tgts_dict', dict())

    # Initial t and s values
    t: 'NDArray' = kwargs.get('t', linspace(0.0, 1.0, numtgt))
    s: 'NDArray' = kwargs.get('s', ones(len(tgts_dict)))

    # Solution variables
    tol_dv: float = kwargs.get('tol_dv', 1e-12)
    tol_f: float = kwargs.get('tol_f', 1e-12)
    max_iter: int = kwargs.get('max_iter', 100)
    display: bool = kwargs.get('display', False)

    # Initializations
    tgts_target = Vector2D.from_iter(tgts_dict.values())
    ind_d_c = asarray(list(tgts_dict.keys()))

    # Point indices
    full_range = arange(numtgt)
    ind_p_f = setdiff1d(full_range, ind_p_c)
    ind_t_f = arange(1, numtgt - 1)

    count = 0

    while True:

        if display:
            print(f'Iteration {count}\n')

        t_p_c = t[ind_p_c]

        pnts_c = bspline.evaluate_points_at_t(t_p_c)

        Dx_c, Dy_c = (pnts_c - pnts_target[ind_p_c]).to_xy()

        t_p_f = t[ind_p_f]

        pnts_f = bspline.evaluate_points_at_t(t_p_f)

        Dx_f, Dy_f = (pnts_f - pnts_target[ind_p_f]).to_xy()

        t_d_c = t[ind_d_c]

        tgts_c = bspline.evaluate_first_derivatives_at_t(t_d_c)

        Du_c, Dv_c = (tgts_c - tgts_target*s).to_xy()

        f_f = concatenate((Dx_f, Dy_f))
        f_c = concatenate((Dx_c, Dy_c, Du_c, Dv_c))
        norm_f = norm(f_f) + norm(f_c)

        if display:
            print(f'f_f = {f_f}\n')
            print(f'f_c = {f_c}\n')
            print(f'norm_f = {norm_f}\n')

        if norm_f < tol_f:
            if display:
                print('Converged')
            break

        t_f = t[ind_t_f]

        Nt = bspline.basis_functions(t).transpose()
        # print(f'Nt = \n{Nt}\n')

        dNt = bspline.basis_first_derivatives(t).transpose()
        # print(f'dNt = \n{dNt}\n')

        dDxdX_f = Nt[ind_p_f, ...]
        dDydX_f = zeros(dDxdX_f.shape)
        dDydY_f = Nt[ind_p_f, ...]
        dDxdY_f = zeros(dDydY_f.shape)
        dDxds_f = zeros((Dx_f.size, s.size))
        dDyds_f = zeros((Dy_f.size, s.size))

        dfdXYs_f = vstack((hstack((dDxdX_f, dDxdY_f, dDxds_f)),
                        hstack((dDydX_f, dDydY_f, dDyds_f))))

        dDxdX_c = Nt[ind_p_c, ...]
        dDydX_c = zeros((Dy_c.size, numctl))
        dDxdY_c = zeros((Dx_c.size, numctl))
        dDydY_c = Nt[ind_p_c, ...]
        dDxds_c = zeros((Dx_c.size, s.size))
        dDyds_c = zeros((Dy_c.size, s.size))

        dDudX_c = dNt[ind_d_c, ...]
        dDvdX_c = zeros((Dv_c.size, numctl))
        dDudY_c = zeros((Du_c.size, numctl))
        dDvdY_c = dNt[ind_d_c, ...]
        dDuds_c = -diag(tgts_target.x)
        dDvds_c = -diag(tgts_target.y)

        dfdXYs_c = vstack((hstack((dDxdX_c, dDxdY_c, dDxds_c)),
                        hstack((dDydX_c, dDydY_c, dDyds_c)),
                        hstack((dDudX_c, dDudY_c, dDuds_c)),
                        hstack((dDvdX_c, dDvdY_c, dDvds_c))))

        dvXYs, _ = solve_clsq(dfdXYs_f, f_f, dfdXYs_c, f_c)

        if display:
            print(f'dvXYs = {dvXYs}\n')

        bspline.ctlpnts.x -= dvXYs[0:numctl]
        bspline.ctlpnts.y -= dvXYs[numctl:2*numctl]
        bspline.reset()

        s -= dvXYs[2*numctl:]

        t_f = t[ind_t_f]

        pnts = bspline.evaluate_points_at_t(t_f)
        # print(f'pnts = \n{pnts}\n')

        Dx, Dy = (pnts - pnts_target[ind_t_f]).to_xy()

        fxy = concatenate((Dx, Dy))

        if display:
            print(f'fxy = {fxy}\n')

        dDxdt, dDydt = bspline.evaluate_first_derivatives_at_t(t_f).to_xy()
        dDxdt = diag(dDxdt)
        dDydt = diag(dDydt)
        dfxydt = vstack((dDxdt, dDydt))

        dvt, _, _, _ = lstsq(dfxydt, fxy, rcond=None)

        if display:
            print(f'dvt = {dvt}\n')

        t[ind_t_f] -= dvt

        norm_dv = norm(dvXYs) + norm(dvt)

        if display:
            print(f'norm_dv = {norm_dv}\n')

        if norm_dv < tol_dv:
            if display:
                print('Converged')
            break

        if display:
            print(f'X = {bspline.ctlpnts.x}\n')
            print(f'Y = {bspline.ctlpnts.y}\n')
            print(f't = {t}\n')
            print(f't[1:] - t[:-1] = {t[1:] - t[:-1]}\n')
            print(f's = {s}\n')

        count += 1
        if count > max_iter:
            if display:
                print('Max Iterations Reached')
            break

    return bspline

class PolyFit:
    x: 'NDArray'
    y: 'NDArray'
    deg: int
    _coeffs: 'NDArray'
    _residuals: 'NDArray'
    _rank: int
    _singvals: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, x: 'NDArray', y: 'NDArray', deg: int) -> None:
        self.x = x
        self.y = y
        self.deg = deg
        self.reset()

    def reset(self) -> None:
        for attr in self.__slots__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def fit(self) -> None:
        Amat = zeros((self.x.size, self.deg + 1))
        for i in range(self.deg + 1):
            Amat[:, i] = self.x**i
        coeffs, residuals, rank, singvals = lstsq(Amat, self.y, rcond=None)
        self._coeffs = coeffs
        self._residuals = residuals
        self._rank = rank
        self._singvals = singvals

    @property
    def coeffs(self) -> 'NDArray':
        if self._coeffs is None:
            self.fit()
        return self._coeffs

    @property
    def residuals(self) -> 'NDArray':
        if self._residuals is None:
            self.fit()
        return self._residuals

    @property
    def rank(self) -> int:
        if self._rank is None:
            self.fit()
        return self._rank

    @property
    def singvals(self) -> 'NDArray':
        if self._singvals is None:
            self.fit()
        return self._singvals

    def __call__(self, x: 'NDArray') -> 'NDArray':
        y = zeros(x.size)
        for i in range(self.deg + 1):
            y += self.coeffs[i]*x**i
        return y

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y}, {self.deg})'

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y}, {self.deg})'
