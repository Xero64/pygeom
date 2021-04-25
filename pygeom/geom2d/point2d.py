from .vector2d import Vector2D

class Point2D(Vector2D):
    def __init__(self, x, y):
        super().__init__(x, y)
    def __repr__(self):
        return '<Point2D: {:}, {:}>'.format(self.x, self.y)
    def __str__(self):
        frmstr = '({:}, {:})'
        return frmstr.format(self.x, self.y)
    def __format__(self, format_spec: str):
        frmstr = '({:'+format_spec+'}, {:'+format_spec+'})'
        return frmstr.format(self.x, self.y)

# class Point2D(object):
#     """Point2D Class"""
#     x = None
#     y = None
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def to_vector(self):
#         """Returns the vector from origin to this point"""
#         from .vector2d import Vector2D
#         x = self.x
#         y = self.y
#         return Vector2D(x, y)
#     def distance_from_origin(self):
#         """Returns the distance from origin to this point"""
#         vec = self.to_vector()
#         return vec.return_magnitude()
#     def to_xy(self):
#         """Returns the x, y values of this point"""
#         return self.x, self.y
#     def __eq__(self, obj):
#         if isinstance(obj, Point2D):
#             return self.x == obj.x and self.y == obj.y
#     def __add__(self, obj):
#         from .vector2d import Vector2D
#         if isinstance(obj, Vector2D):
#             return Point2D(self.x+obj.x, self.y+obj.y)
#     def __sub__(self, obj):
#         from .vector2d import Vector2D
#         if isinstance(obj, Vector2D):
#             return Point2D(self.x-obj.x, self.y-obj.y)
#         if isinstance(obj, Point2D):
#             return Vector2D(self.x-obj.x, self.y-obj.y)
#     def __repr__(self):
#         return '<Point2D: {:}, {:}>'.format(self.x, self.y)
#     def __str__(self):
#         frmstr = '({:}, {:})'
#         return frmstr.format(self.x, self.y)
#     def __format__(self, format_spec: str):
#         frmstr = '({:'+format_spec+'}, {:'+format_spec+'})'
#         return frmstr.format(self.x, self.y)

def point2d_from_lists(x, y):
    """Create a list of Point2D objects"""
    n = len(x)
    if len(y) == n:
        pnts = [Point2D(x[i], y[i]) for i in range(n)]
    return pnts

def midpoint_of_point2ds(pnts):
    num = len(pnts)
    x = sum(pnt.x for pnt in pnts)/num
    y = sum(pnt.y for pnt in pnts)/num
    return Point2D(x, y)

def point2d_from_complex(z):
    """Create a Point2D from a complex number"""
    x = z.real
    y = z.imag
    return Point2D(x, y)
