from .vector import Vector

class Point(Vector):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
    def __repr__(self):
        return '<Point: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __str__(self):
        frmstr = '({:}, {:}, {:})'
        return frmstr.format(self.x, self.y, self.z)
    def __format__(self, format_spec: str):
        frmstr = '({:'+format_spec+'}, {:'+format_spec+'}, {:'+format_spec+'})'
        return frmstr.format(self.x, self.y, self.z)

# class Point(object):
#     """Point Class"""
#     x = None
#     y = None
#     z = None
#     def __init__(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.z = z
#     def to_vector(self):
#         """Returns the vector from origin to this point"""
#         from .vector import Vector
#         x = self.x
#         y = self.y
#         z = self.z
#         return Vector(x, y, z)
#     def distance_from_origin(self):
#         """Returns the distance from origin to this point"""
#         vec = self.to_vector()
#         return vec.return_magnitude()
#     def to_xyz(self):
#         """Returns the x, y and z values of this point"""
#         return self.x, self.y, self.z
#     def __eq__(self, obj):
#         if isinstance(obj, Point):
#             return self.x == obj.x and self.y == obj.y and self.z == obj.z
#     def __add__(self, obj):
#         from .vector import Vector
#         if isinstance(obj, Vector):
#             return Point(self.x+obj.x, self.y+obj.y, self.z+obj.z)
#     def __sub__(self, obj):
#         from .vector import Vector
#         if isinstance(obj, Vector):
#             return Point(self.x-obj.x, self.y-obj.y, self.z-obj.z)
#         elif isinstance(obj, Point):
#             return Vector(self.x-obj.x, self.y-obj.y, self.z-obj.z)
#     def __repr__(self):
#         return '<Point: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
#     def __str__(self):
#         frmstr = '({:}, {:}, {:})'
#         return frmstr.format(self.x, self.y, self.z)
#     def __format__(self, format_spec: str):
#         frmstr = '({:'+format_spec+'}, {:'+format_spec+'}, {:'+format_spec+'})'
#         return frmstr.format(self.x, self.y, self.z)

def point_from_lists(x, y, z):
    """Create a list of Point objects"""
    return [Point(x[i], y[i], z[i]) for i in range(len(x))]

def midpoint_of_points(pnts):
    num = len(pnts)
    x = sum(pnt.x for pnt in pnts)/num
    y = sum(pnt.y for pnt in pnts)/num
    z = sum(pnt.z for pnt in pnts)/num
    return Point(x, y, z)
