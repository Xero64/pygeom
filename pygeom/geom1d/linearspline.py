from matplotlib.pyplot import figure

class LinearSpline(object):
    x = None # X Value List
    y = None # Y Value List
    _pieces = None # Spline Pieces
    _xmin = None # Minimum X
    _xmax = None # Maximum X
    def __init__(self, x: list, y: list):
        num = min(len(x), len(y))
        self.x = [x[i] for i in range(num)]
        self.y = [y[i] for i in range(num)]
    @property
    def pieces(self):
        if self._pieces is None:
            self._pieces = []
            for i in range(len(self.x)-1):
                xa = self.x[i]
                xb = self.x[i+1]
                ya = self.y[i]
                yb = self.y[i+1]
                self._pieces.append(LinearPiece(xa, xb, ya, yb))
        return self._pieces
    @property
    def xmin(self):
        if self._xmin is None:
            self._xmin = min(self.x)
        return self._xmin
    @property
    def xmax(self):
        if self._xmax is None:
            self._xmax = max(self.x)
        return self._xmax
    def single_interpolate_spline(self, x: float):
        if x > self.xmax or x < self.xmin:
            return ValueError('Lookup value not in range.')
        y = None
        for piece in self.pieces:
            if piece.contains(x):
                y = piece.interpolate_spline(x=x)
                break
        if y is None:
            return ValueError('Lookup value not found.')
        return y
    def single_interpolate_gradient(self, x: float):
        if x > self.xmax or x < self.xmin:
            return ValueError('Lookup value not in range.')
        dydx = None
        for piece in self.pieces:
            if piece.contains(x):
                dydx = piece.interpolate_gradient(x=x)
                break
        if dydx is None:
            return ValueError('Lookup value not found.')
        return dydx
    def list_interpolation(self, x: list):
        if max(x) > self.xmax or min(x) < self.xmin:
            return ValueError('One or more lookup valus not in range.')
        num = len(self.pieces)
        y = []
        j = 0
        for xi in x:
            found = False
            if not found:
                for i in range(j, num):
                    piece = self.pieces[i]
                    if piece.contains(xi):
                        yi = piece.interpolate_spline(x=xi)
                        j = i
                        found = True
                        break
            if not found:
                for i in range(0, j):
                    piece = self.pieces[i]
                    if piece.contains(xi):
                        yi = piece.interpolate_spline(x=xi)
                        j = i
                        found = True
                        break
            if found:
                y.append(yi)
            else:
                y.append(float('nan'))
        return y
    def plot_spline(self, num: int, ax=None, **kwargs):
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
        x, y = [], []
        for piece in self.pieces:
            x = x + piece.x_list(num)
            y = y + piece.y_list(num)
        ax.plot(x, y, **kwargs)
        return ax
    def plot_gradient(self, num: int, ax=None, **kwargs):
        if ax is None:
            fig = figure(figsize=(12, 8))
            ax = fig.gca()
            ax.grid(True)
        x, dydx = [], []
        for piece in self.pieces:
            x = x + piece.x_list(num)
            dydx = dydx + piece.dydx_list(num)
        ax.plot(x, dydx, **kwargs)
        return ax
    def __str__(self):
        from py2md.classes import MDTable
        table = MDTable()
        table.add_column('x', ':', data=self.x)
        table.add_column('y', ':', data=self.y)
        return table.__str__()
    def _repr_markdown_(self):
        return self.__str__()
    def __repr__(self):
        return '<LinearSpline>'

class LinearPiece(object):
    xa = None
    xb = None
    ya = None
    yb = None
    _dx = None
    _dy = None
    _dydx = None
    def __init__(self, xa: float, xb: float, ya: float, yb: float):
        self.xa = xa
        self.xb = xb
        self.ya = ya
        self.yb = yb
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
        return self.ya+self.dydx*s
    def interpolate_gradient(self, s: float=None, x: float=None):
        if s is None:
            s = x-self.xa
        return self.dydx
    def x_list(self):
        return [self.xa, self.xb]
    def y_list(self):
        return [self.ya, self.yb]
    def dydx_list(self):
        return [self.dydx, self.dydx]
    def __repr__(self):
        return '<LinearPiece>'
