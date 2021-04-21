from matplotlib.pyplot import figure
from .linearspline import LinearSpline

class CubicSpline(LinearSpline):
    _grad = None # Gradient
    _curv = None # Curvature
    _pieces = None # Spline Pieces
    @property
    def curv(self):
        if self._curv is None:
            num = len(self.x)
            a = [0.0 for i in range(num)]
            b = [1.0 for i in range(num)]
            c = [0.0 for i in range(num)]
            r = [0.0 for i in range(num)]
            for i in range(1, num-1):
                dxA = self.x[i]-self.x[i-1]
                dxB = self.x[i+1]-self.x[i]
                dyA = self.y[i]-self.y[i-1]
                dyB = self.y[i+1]-self.y[i]
                a[i] = dxA/6
                b[i] = (dxA+dxB)/3
                c[i] = dxB/6
                r[i] = dyB/dxB-dyA/dxA
            gm = [0.0 for i in range(num)]
            bt = b[0]
            self._curv = [0.0 for i in range(num)]
            self._curv[0] = r[0]/bt
            for i in range(1, num):
                gm[i] = c[i-1]/bt
                bt = b[i]-a[i]*gm[i]
                self._curv[i] = (r[i]-a[i]*self._curv[i-1])/bt
            for i in range(num-2, 0, -1):
                self._curv[i] -= gm[i+1]*self._curv[i+1]
        return self._curv
    @property
    def grad(self):
        if self._grad is None:
            self._grad = [piece.interpolate_gradient(0.0) for piece in self.pieces]
            self._grad.append(self.pieces[-1].interpolate_gradient(self.pieces[-1].dx))
        return self._grad
    @property
    def pieces(self):
        if self._pieces is None:
            self._pieces = []
            for i in range(len(self.x)-1):
                xa = self.x[i]
                xb = self.x[i+1]
                ya = self.y[i]
                yb = self.y[i+1]
                d2ya = self.curv[i]
                d2yb = self.curv[i+1]
                self._pieces.append(CubicPiece(xa, xb, ya, yb, d2ya, d2yb))
        return self._pieces
    def single_interpolate_curvature(self, x: float):
        if x > self.xmax or x < self.xmin:
            return ValueError('Lookup value not in range.')
        y = None
        for piece in self.pieces:
            if piece.contains(x):
                y = piece.interpolate_curvature(x=x)
                break
        if y is None:
            return ValueError('Lookup value not found.')
        return y
    def plot_curvature(self, num: int, ax=None, **kwargs):
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
        x, d2y = [], []
        for piece in self.pieces:
            x = x + piece.x_list(num)
            d2y = d2y + piece.d2y_list(num)
        ax.plot(x, d2y, **kwargs)
        return ax
    def __str__(self):
        from py2md.classes import MDTable
        table = MDTable()
        table.add_column('x', '', data=self.x)
        table.add_column('y', '', data=self.y)
        table.add_column('Gradient', '', data=self.grad)
        table.add_column('Curvature', '', data=self.curv)
        return table.__str__()
    def _repr_markdown_(self):
        return self.__str__()
    def __repr__(self):
        return '<CubicSpline>'

class CubicPiece(object):
    xa = None
    xb = None
    ya = None
    yb = None
    d2ya = None
    d2yb = None
    _dx = None
    _dy = None
    _dydx = None
    def __init__(self, xa: float, xb: float,
                 ya: float, yb: float,
                 d2ya: float, d2yb: float):
        self.xa = xa
        self.xb = xb
        self.ya = ya
        self.yb = yb
        self.d2ya = d2ya
        self.d2yb = d2yb
    @property
    def dx(self):
        if self._dx is None:
            self._dx = self.xb-self.xa
        return self._dx
    @property
    def dy(self):
        if self._dy is None:
            self._dy = self.yb-self.ya
        return self._dy
    @property
    def dydx(self):
        if self._dydx is None:
            self._dydx = self.dy/self.dx
        return self._dydx
    def contains(self, x: float):
        if x >= self.xa and x <= self.xb:
            return True
        else:
            return False
    def interpolate_spline(self, s: float=None, x: float=None):
        if s is None:
            s = x-self.xa
        A = (self.dx-s)/self.dx
        B = s/self.dx
        C = (A**3-A)*self.dx**2/6
        D = (B**3-B)*self.dx**2/6
        return A*self.ya+B*self.yb+C*self.d2ya+D*self.d2yb
    def interpolate_gradient(self, s: float=None, x: float=None):
        if s is None:
            s = x-self.xa
        A = (self.dx-s)/self.dx
        B = s/self.dx
        E = 3*A**2-1
        F = 3*B**2-1
        return self.dydx+(F*self.d2yb-E*self.d2ya)*self.dx/6
    def interpolate_curvature(self, s: float=None, x: float=None):
        if s is None:
            s = x-self.xa
        A = (self.dx-s)/self.dx
        B = s/self.dx
        return A*self.d2ya+B*self.d2yb
    def s_list(self, num: int):
        return [i*self.dx/num for i in range(num+1)]
    def x_list(self, num: int):
        return [self.xa+si for si in self.s_list(num)]
    def y_list(self, num: int):
        return [self.interpolate_spline(s=si) for si in self.s_list(num)]
    def dydx_list(self, num: int):
        return [self.interpolate_gradient(s=si) for si in self.s_list(num)]
    def d2y_list(self, num: int):
        return [self.interpolate_curvature(s=si) for si in self.s_list(num)]
    def __repr__(self):
        return '<CubicPiece>'
