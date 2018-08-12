# from .vector2d import Vector2D

class Point2D(object):
    """Point2D Class"""
    x = None
    y = None
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_vector(self):
        """Returns the vector from origin to this point"""
        x = self.x
        y = self.y
        return Vector2D(x, y)
    def distance_from_origin(self):
        """Returns the distance from origin to this point"""
        vec = self.to_vector()
        return vec.return_magnitude()
    def __eq__(self, obj):
        if isinstance(obj, Point2D):
            return self.x == obj.x and self.y == obj.y
    def __add__(self, obj):
        if isinstance(obj, Vector2D):
            return Point2D(self.x+obj.x, self.y+obj.y)
    def __sub__(self, obj):
        if isinstance(obj, Vector2D):
            return Point2D(self.x-obj.x, self.y-obj.y)
    def __repr__(self):
        chx = isinstance(self.x, float)
        chy = isinstance(self.y, float)
        if chx and chy:
            frmstr = '<Point2D: {:.8g}, {:.8g}>'
        else:
            frmstr = '<Point2D: {:}, {:}>'
        return frmstr.format(self.x, self.y)
    def __str__(self):
        chx = isinstance(self.x, float)
        chy = isinstance(self.y, float)
        if chx and chy:
            frmstr = '{:.8g}\t{:.8g}'
        else:
            frmstr = '{:}\t{:}'
        return frmstr.format(self.x, self.y)

def point2d_from_lists(x, y):
    """Create a list of Point2D objects"""
    n = len(x)
    if len(y) == n:
        pnts = [Point2D(x[i], y[i]) for i in range(n)]
    return pnts

def point2d_from_complex(z):
    """Create a Point2D from a complex number"""
    x = z.real
    y = z.imag
    return Point2D(x, y)
