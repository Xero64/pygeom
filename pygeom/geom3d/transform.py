from .vector import Vector


class Transform():
    """3D Transform Class"""
    dirx: Vector = None
    diry: Vector = None
    dirz: Vector = None

    def __init__(self, vecx: Vector, vecxy: Vector) -> None:
        self.dirx = vecx.to_unit()
        vecz = vecx.cross(vecxy)
        vecy = vecz.cross(vecx)
        self.diry = vecy.to_unit()
        self.dirz = vecz.to_unit()

    def vector_to_global(self, vec: Vector) -> Vector:
        """Transforms a vector from this local coordinate to global."""
        dirx = Vector(self.dirx.x, self.diry.x, self.dirz.x)
        diry = Vector(self.dirx.y, self.diry.y, self.dirz.y)
        dirz = Vector(self.dirx.z, self.diry.z, self.dirz.z)
        x = vec.dot(dirx)
        y = vec.dot(diry)
        z = vec.dot(dirz)
        return vec.__class__(x, y, z)

    def vector_to_local(self, vec: Vector) -> Vector:
        """Transforms a vector from global to this local coordinate."""
        x = vec.dot(self.dirx)
        y = vec.dot(self.diry)
        z = vec.dot(self.dirz)
        return vec.__class__(x, y, z)

    def __repr__(self) -> str:
        return '<Transform>'
