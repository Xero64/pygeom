from math import cos, sin

from .tensor2d import Tensor2D
from .vector2d import Vector2D


class Transform2D():
    """Transform2D Class"""
    dirx: Vector2D = None
    _diry: Vector2D = None
    def __init__(self, vecx: Vector2D) -> None:
        self.dirx = vecx.to_unit()
    @property
    def diry(self) -> Vector2D:
        if self._diry is None:
            self._diry = Vector2D(-self.dirx.y, self.dirx.x)
        return self._diry
    def vector_to_global(self, vec: Vector2D) -> Vector2D:
        """Transforms a vector from this local coordinate system to global."""
        dirx = Vector2D(self.dirx.x, self.diry.x)
        diry = Vector2D(self.dirx.y, self.diry.y)
        x = dirx*vec
        y = diry*vec
        return Vector2D(x, y)
    def vector_to_local(self, vec: Vector2D) -> Vector2D:
        """Transforms a vector from global to this local coordinate system."""
        dirx = self.dirx
        diry = self.diry
        x = dirx*vec
        y = diry*vec
        return Vector2D(x, y)
    def tensor_to_global(self, ten: 'Tensor2D') -> 'Tensor2D':
        """Transforms a tensor from this local coordinate system to global."""
        dirx = Vector2D(self.dirx.x, self.diry.x)
        diry = Vector2D(self.dirx.y, self.diry.y)
        xx = dirx*ten*dirx
        xy = dirx*ten*diry
        yx = diry*ten*dirx
        yy = diry*ten*diry
        return Tensor2D(xx, xy, yx, yy)
    def tensor_to_local(self, ten: 'Tensor2D') -> 'Tensor2D':
        """Transforms a tensor from global to this local coordinate system."""
        dirx = self.dirx
        diry = self.diry
        xx = dirx*ten*dirx
        xy = dirx*ten*diry
        yx = diry*ten*dirx
        yy = diry*ten*diry
        return Tensor2D(xx, xy, yx, yy)
    def rotate_about_z(self, angle: float) -> 'Transform2D':
        """Creates a transform that is rotated by an angle [radians]."""
        cos_ang = cos(angle)
        sin_ang = sin(angle)
        dirx = self.dirx*cos_ang + self.diry*sin_ang
        return Transform2D(dirx)
    def __repr__(self) -> str:
        return '<Transform2D>'
