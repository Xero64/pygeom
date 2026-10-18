from typing import TYPE_CHECKING

from numpy import (absolute, asarray, cumsum, fill_diagonal, full, imag, isnan,
                   logical_and, logical_or, real, reciprocal, unique, zeros)

from ..tools.basis import knot_linspace
from ..tools.roots import linear_roots

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LinearSpline1DSolver():
    u"""This class solves for the quadratic spline coefficients."""
    s: 'NDArray' = None
    _sa: 'NDArray' = None
    _sb: 'NDArray' = None
    _Ds: 'NDArray' = None
    _Dsr: 'NDArray' = None
    _a0: 'NDArray' = None
    _a1: 'NDArray' = None
    _b0: 'NDArray' = None
    _b1: 'NDArray' = None
    _iarr: 'NDArray' = None

    def __init__(self, s: 'NDArray', validate: bool = True) -> None:
        u"""This function initialises the object."""
        self.s = asarray(s)
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        if self.s.ndim != 1:
            raise ValueError('Input s must be a 1D ndarray.')

    def evaluate_t(self, num: int) -> 'NDArray':
        return knot_linspace(num, self.s)

    @property
    def sa(self) -> 'NDArray':
        if self._sa is None:
            self._sa = self.s[:-1]
        return self._sa

    @property
    def sb(self) -> 'NDArray':
        if self._sb is None:
            self._sb = self.s[1:]
        return self._sb

    @property
    def Ds(self) -> 'NDArray':
        if self._Ds is None:
            self._Ds = self.sb - self.sa
        return self._Ds

    @property
    def Dsr(self) -> 'NDArray':
        if self._Dsr is None:
            self._Dsr = reciprocal(self.Ds)
        return self._Dsr

    @property
    def a0(self) -> 'NDArray':
        if self._a0 is None:
            self._a0 = self.sb * self.Dsr
        return self._a0

    @property
    def a1(self) -> 'NDArray':
        if self._a1 is None:
            self._a1 = -self.Dsr
        return self._a1

    @property
    def b0(self) -> 'NDArray':
        if self._b0 is None:
            self._b0 = -self.sa * self.Dsr
        return self._b0

    @property
    def b1(self) -> 'NDArray':
        if self._b1 is None:
            self._b1 = self.Dsr
        return self._b1

    def calculate_integral_array(self) -> None:
        Dsio2 = self.Ds / 2.0
        iarr = zeros((self.s.size, self.s.size))
        fill_diagonal(iarr[1:, :-1], Dsio2)
        fill_diagonal(iarr[1:-1, 1:], Dsio2[:-1])
        self._iarr = cumsum(iarr, axis=0)

    @property
    def iarr(self) -> 'NDArray':
        if self._iarr is None:
            self.calculate_integral_array()
        return self._iarr

    def evaluate_points_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the spline at a given s."""
        s = asarray(s)
        z = zip(self.sa, self.sb,
                self.a0, self.a1,
                self.b0, self.b1)
        rarr = zeros((*s.shape, self.s.size))
        for i, zi in enumerate(z):
            sa, sb, a0, a1, b0, b1  = zi
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            av = a0 + a1 * sv
            bv = b0 + b1 * sv
            rarr[s_check, i] = av
            rarr[s_check, i + 1] = bv
        return rarr

    def evaluate_integral_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the integrals of the spline at a given s."""
        s = asarray(s)
        z = zip(self.sa, self.sb,
                self.a0, self.a1,
                self.b0, self.b1)
        iarr = zeros((*s.shape, self.s.size))
        for i, zi in enumerate(z):
            sa, sb, a0, a1, b0, b1 = zi
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            sav = sv - sa
            sav2 = (sv**2 - sa**2) / 2.0
            int_ya = a0 * sav + a1 * sav2
            int_yb = b0 * sav + b1 * sav2
            iarr[s_check, i] += int_ya
            iarr[s_check, i+1] += int_yb
            iarr[s_check, :] += self.iarr[i, :]
        return iarr

    def to_linear_spline_1d(self, r: 'NDArray', validate: bool = True) -> 'LinearSpline1D':
        u"""This function creates a LinearSpline1D object from the solver."""
        ls1d = LinearSpline1D(self.s, r, validate=validate)
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(ls1d, attr, getattr(self, attr))
        return ls1d

    def __repr__(self):
        return '<LinearSpline1DSolver>'


class LinearSpline1D(LinearSpline1DSolver):
    u"""This class stores a 1D parametric linear spline."""
    r: 'NDArray' = None
    _ra: 'NDArray' = None
    _rb: 'NDArray' = None
    _Dr: 'NDArray' = None
    _Ir: 'NDArray' = None

    def __init__(self, s: 'NDArray', r: 'NDArray',
                 validate: bool = True) -> None:
        u"""This function initialises the object."""
        super().__init__(s, validate=False)
        self.r = asarray(r)
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        super().validate()
        if self.r.ndim != 1:
            raise ValueError('Input r must be a 1D ndarray.')
        if self.r.size != self.s.size:
            raise ValueError('Input r must have the same size as s.')

    @property
    def ra(self) -> 'NDArray':
        if self._ra is None:
            self._ra = self.r[:-1]
        return self._ra

    @property
    def rb(self) -> 'NDArray':
        if self._rb is None:
            self._rb = self.r[1:]
        return self._rb

    @property
    def Dr(self) -> 'NDArray':
        if self._Dr is None:
            self._Dr = self.rb - self.ra
        return self._Dr

    @property
    def Ir(self) -> 'NDArray':
        if self._Ir is None:
            self._Ir = self.iarr @ self.r
        return self._Ir

    def evaluate_points_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the spline at a given s."""
        s = asarray(s)
        r = full(s.shape, float('nan'))
        z = zip(self.sa, self.sb,
                self.ra, self.rb,
                self.a0, self.a1,
                self.b0, self.b1)
        for zi in z:
            sa, sb, ra, rb, a0, a1, b0, b1 = zi
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            av = a0 + a1 * sv
            bv = b0 + b1 * sv
            r[s_check] = ra * av + rb * bv
        return r

    def evaluate_first_derivatives_at_t(self, s: 'NDArray',
                                        tol: float = 1e-12) -> 'NDArray':
        u"""This function evaluates the first derivatives of the spline at a given s."""
        s = asarray(s)
        dr = full(s.shape, float('nan'))
        z = zip(self.sa, self.sb,
                self.ra, self.rb,
                self.a1, self.b1)
        for zi in z:
            sa, sb, ra, rb, a1, b1 = zi
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            av = full(sv.shape, a1)
            bv = full(sv.shape, b1)
            val = ra * av + rb * bv
            if sa != self.s[0]:
                sav_check = absolute(sv - sa) < tol
            else:
                sav_check = full(sv.shape, False)
            if sb != self.s[-1]:
                sbv_check = absolute(sv - sb) < tol
            else:
                sbv_check = full(sv.shape, False)
            sab_check = logical_or(sav_check, sbv_check)
            val[sab_check] = val[sab_check] * 0.5
            nan_check = logical_and(s_check, isnan(dr))
            dr[nan_check] = 0.0
            dr[s_check] += val
        return dr

    def evaluate_points(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        points = self.evaluate_points_at_t(s)
        return points

    def evaluate_integrals_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the integrals of the spline at a given s."""
        s = asarray(s)
        Ir = full(s.shape, float('nan'))
        z = zip(self.sa, self.sb,
                self.ra, self.rb,
                self.a0, self.a1,
                self.b0, self.b1)
        for i, zi in enumerate(z):
            sa, sb, ra, rb, a0, a1, b0, b1 = zi
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            sav = sv - sa
            sav2o2 = (sv**2 - sa**2) / 2.0
            int_ya_bv = a0 * sav + a1 * sav2o2
            int_yb_bv = b0 * sav + b1 * sav2o2
            Ia = self.Ir[i]
            Ir[s_check] = Ia + ra * int_ya_bv + rb * int_yb_bv
        return Ir

    def evaluate_integrals(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        integrals = self.evaluate_integrals_at_t(s)
        return integrals

    def copy_with_new_input(self, new_input: 'NDArray') -> 'LinearSpline1D':
        new_spline = LinearSpline1D(self.s, new_input, validate=False)
        new_spline._sa = self._sa
        new_spline._sb = self._sb
        new_spline._Ds = self._Ds
        new_spline._iarr = self._iarr
        return new_spline

    def find_intercepts(self, value: float, tol: float = 1e-12) -> 'NDArray':
        s_int = []
        z = zip(self.sa, self.sb, self.Ds, self.ra, self.rb, self.Dr)
        for zi in z:
            sa, sb, Ds, ra, rb, Dr = zi
            ra = ra - value
            rb = rb - value
            a = Dr / Ds
            b = ra - sa * a
            root = linear_roots(a, b)
            if abs(imag(root)) < tol:
                root_sa = real(root) - sa
                root_sb = real(root) - sb
                if root_sa >= 0.0 and root_sb <= 0.0:
                    s_int.append(real(root))
                elif root_sa >= -tol and root_sa < 0.0:
                    s_int.append(real(sa))
                elif root_sb <= tol and root_sb > 0.0:
                    s_int.append(real(sb))
        return unique(asarray(s_int))

    def __repr__(self):
        return '<LinearSpline1D>'
