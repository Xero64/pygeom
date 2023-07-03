#%%
# Import Dependencies
from typing import TYPE_CHECKING
from numpy import linspace
from pygeom.geom2d import Vector2D
from pygeom.array2d import ArrayVector2D
from matplotlib.pyplot import figure

if TYPE_CHECKING:
    from numpy import ndarray

#%%
# Define Class
class CubicPiece2D():
    ra: 'Vector2D' = None
    rb: 'Vector2D' = None
    d2ra: 'Vector2D' = None
    d2rb: 'Vector2D' = None
    _vec: 'Vector2D' = None
    _length: float = None
    _uvec: 'Vector2D' = None
    _sa: float = None
    _sb: float = None

    def __init__(self, ra: 'Vector2D', rb: 'Vector2D',
                 d2ra: 'Vector2D', d2rb: 'Vector2D') -> None:
        self.ra = ra
        self.rb = rb
        self.d2ra = d2ra
        self.d2rb = d2rb

    @property
    def vec(self) -> 'Vector2D':
        if self._vec is None:
            self._vec = self.rb - self.ra
        return self._vec

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self.vec.return_magnitude()
        return self._length

    @property
    def uvec(self) -> 'Vector2D':
        if self._uvec is None:
            self._uvec = self.vec/self.length
        return self._uvec

    @property
    def sa(self) -> float:
        if self._sa is None:
            self._sa = 0.0
        return self._sa

    @sa.setter
    def sa(self, value: float) -> None:
        self._sa = value
        self._sb = value + self.length

    @property
    def sb(self) -> float:
        if self._sb is None:
            self._sb = self.length
        return self._sb

    @sb.setter
    def sb(self, value: float) -> None:
        self._sb = value
        self._sa = value - self.length

    def spline_points(self, s: 'ndarray') -> ArrayVector2D:
        Axl = (self.sb - s)
        Bxl = (s - self.sa)
        A = Axl/self.length
        B = Bxl/self.length
        J = -Axl*Bxl/6
        C = J*(2*A + B)
        D = J*(A + 2*B)
        x = A*self.ra.x + B*self.rb.x + C*self.d2ra.x + D*self.d2rb.x
        y = A*self.ra.y + B*self.rb.y + C*self.d2ra.y + D*self.d2rb.y
        return ArrayVector2D(x, y)

#%%
# Create Spline
s = linspace(0.0, 2.0, 100, dtype=float)

ra = Vector2D(1.0, 1.0)
rb = Vector2D(3.0, 1.0)
d2ra = Vector2D(1.0, -1.0)
d2rb = Vector2D(-1.0, -1.0)

splpc = CubicPiece2D(ra, rb, d2ra, d2rb)

r = splpc.spline_points(s)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.set_aspect('equal')
ax.grid(True)
_ = ax.plot(r.x, r.y)

#%%
# Checks
dxa = r.x[1] - r.x[0]
dya = r.y[1] - r.y[0]
dxb = r.x[-1] - r.x[-2]
dyb = r.y[-1] - r.y[-2]

print(f'Vector2D(dxa, dya).to_unit() = {Vector2D(dxa, dya).to_unit()}')

print(f'Vector2D(dxb, dyb).to_unit() = {Vector2D(dxb, dyb).to_unit()}')
