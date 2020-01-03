from .point2d import Point2D
from .vector2d import vector2d_from_points

class Line2D(object):
    """Line2D Class"""
    pnta = None
    pntb = None
    vec = None
    length = None
    def __init__(self, pnta: Point2D, pntb: Point2D):
        self.pnta = pnta
        self.pntb = pntb
        self.update()
    def update(self):
        """Updates this line if the point definition changes"""
        self.vec = vector2d_from_points(self.pnta, self.pntb)
        self.length = self.vec.return_magnitude()
    def centre_point(self):
        """Returns the centre point of this line"""
        x = (self.pnta.x+self.pntb.x)/2
        y = (self.pnta.y+self.pntb.y)/2
        return Point2D(x, y)
    def ratio_point(self, ratio: float):
        """Returns a point a certain ratio along the line"""
        vec = vector2d_from_points(self.pnta, self.pntb)
        return self.pnta+ratio*vec
    def __repr__(self):
        return '<Line2D>'
