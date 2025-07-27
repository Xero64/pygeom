from .tensor2d import Tensor2D
from .vector2d import Vector2D


class Transform2D():
    """Transform2D Class"""
    dirx: Vector2D = None
    diry: Vector2D = None

    def __init__(self, vecx: Vector2D) -> None:
        self.dirx = vecx.to_unit()
        self.diry = Vector2D(-self.dirx.y, self.dirx.x)

    def vector2d_to_global(self, vec: Vector2D) -> Vector2D:
        """Transforms a vector from this local coordinate system to global."""
        dirx = Vector2D(self.dirx.x, self.diry.x)
        diry = Vector2D(self.dirx.y, self.diry.y)
        x = vec.dot(dirx)
        y = vec.dot(diry)
        return vec.__class__(x, y)

    def vector2d_to_local(self, vec: Vector2D) -> Vector2D:
        """Transforms a vector from global to this local coordinate system."""
        x = vec.dot(self.dirx)
        y = vec.dot(self.diry)
        return vec.__class__(x, y)

    def tensor2d_to_global(self, tens: Tensor2D) -> Tensor2D:
        """Transforms a tensor from this local coordinate system to global"""
        sxx, sxy, syx, syy = tens.to_xy()
        qxx, qxy, qyx, qyy = self.dirx.x, self.dirx.y, self.diry.x, self.diry.y
        exx = qxx**2*sxx + qxx*qyx*sxy + qxx*qyx*syx + qyx**2*syy
        eyy = qxy**2*sxx + qxy*qyy*sxy + qxy*qyy*syx + qyy**2*syy
        exy = qxx*qxy*sxx + qxx*qyy*sxy + qxy*qyx*syx + qyx*qyy*syy
        eyx = qxx*qxy*sxx + qxx*qyy*syx + qxy*qyx*sxy + qyx*qyy*syy
        return tens.__class__(exx, exy, eyx, eyy)

    def tensor2d_to_local(self, tens: Tensor2D) -> Tensor2D:
        """Transforms a tensor from global to this local coordinate system"""
        sxx, sxy, syx, syy = tens.to_xy()
        qxx, qxy, qyx, qyy = self.dirx.x, self.dirx.y, self.diry.x, self.diry.y
        exx = qxx**2*sxx + qxx*qxy*sxy + qxx*qxy*syx + qxy**2*syy
        eyy = qyx**2*sxx + qyx*qyy*sxy + qyx*qyy*syx + qyy**2*syy
        exy = qxx*qyx*sxx + qxx*qyy*sxy + qxy*qyx*syx + qxy*qyy*syy
        eyx = qxx*qyx*sxx + qxx*qyy*syx + qxy*qyx*sxy + qxy*qyy*syy
        return tens.__class__(exx, exy, eyx, eyy)

    def __repr__(self) -> str:
        return '<Transform2D>'
