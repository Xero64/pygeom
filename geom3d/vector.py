class Vector(object):
    """Vector Class"""
    x = None
    y = None
    z = None
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def to_unit(self):
        """Returns the unit vector of this vector"""
        mag = self.return_magnitude()
        if mag == 0:
            return Vector(self.x, self.y, self.z)
        else:
            x = self.x/mag
            y = self.y/mag
            z = self.z/mag
            return Vector(x, y, z)
    def to_point(self):
        """Returns the end point position of this vector"""
        from .point import Point
        return Point(self.x, self.y, self.z)
    def return_magnitude(self):
        """Returns the magnitude of this vector"""
        return (self.x**2+self.y**2+self.z**2)**0.5
    def __mul__(self, obj):
        if isinstance(obj, Vector):
            return self.x*obj.x+self.y*obj.y+self.z*obj.z
        else:
            x = obj*self.x
            y = obj*self.y
            z = obj*self.z
            return Vector(x, y, z)
    def __rmul__(self, obj):
        return self.__mul__(obj)
    def __truediv__(self, obj):
        if isinstance(obj, (int, float, complex)):
            x = self.x/obj
            y = self.y/obj
            z = self.z/obj
            return Vector(x, y, z)
    def __pow__(self, obj):
        if isinstance(obj, Vector):
            x = self.y*obj.z-self.z*obj.y
            y = self.z*obj.x-self.x*obj.z
            z = self.x*obj.y-self.y*obj.x
            return Vector(x, y, z)
    def __add__(self, obj):
        if isinstance(obj, Vector):
            x = self.x+obj.x
            y = self.y+obj.y
            z = self.z+obj.z
            return Vector(x, y, z)
    def __radd__(self, obj):
        if obj == 0:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj):
        if isinstance(obj, Vector):
            x = self.x-obj.x
            y = self.y-obj.y
            z = self.z-obj.z
            return Vector(x, y, z)
    def __pos__(self):
        return Vector(self.x, self.y, self.z)
    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)
    def __repr__(self):
        return '<Vector: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __str__(self):
        return '<{:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __format__(self, format_spec):
        frmstr = '<{:'+format_spec+'}, {:'+format_spec+'}, {:'+format_spec+'}>'
        return frmstr.format(self.x, self.y, self.z)

def vector_from_points(pnta, pntb):
    """Create a Vector from two Points"""
    x = pntb.x-pnta.x
    y = pntb.y-pnta.y
    z = pntb.z-pnta.z
    return Vector(x, y, z)

ihat = Vector(1.0, 0.0, 0.0)
jhat = Vector(0.0, 1.0, 0.0)
khat = Vector(0.0, 0.0, 1.0)
zerovec = Vector(0.0, 0.0, 0.0)
