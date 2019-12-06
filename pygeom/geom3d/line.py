from .point import Point
from .vector import vector_from_points

class Line(object):
    """Line Class"""
    pnta = None
    pntb = None
    vec = None
    length = None
    def __init__(self, pnta, pntb):
        self.pnta = pnta
        self.pntb = pntb
        self.update()
    def update(self):
        """Updates this line if the point definition changes"""
        self.vec = vector_from_points(self.pnta, self.pntb)
        self.length = self.vec.return_magnitude()
    def centre_point(self):
        """Returns the centre point of this line"""
        x = (self.pnta.x+self.pntb.x)/2
        y = (self.pnta.y+self.pntb.y)/2
        z = (self.pnta.z+self.pntb.z)/2
        return Point(x, y, z)
    def ratio_point(self, ratio):
        """Returns a point a certain ratio along the line"""
        vec = vector_from_points(self.pnta, self.pntb)
        return self.pnta+ratio*vec
    def __repr__(self):
        return '<Line>'
