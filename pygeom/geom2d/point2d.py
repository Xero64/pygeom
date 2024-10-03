from .vector2d import Vector2D


class Point2D(Vector2D):

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)

    def __repr__(self) -> str:
        return '<Point2D: {:}, {:}>'.format(self.x, self.y)

    def __str__(self) -> str:
        frmstr = '({:}, {:})'
        return frmstr.format(self.x, self.y)
    
    def __format__(self, frm: str) -> str:
        frmstr = '({:' + frm + '}, {:' + frm + '})'
        return frmstr.format(self.x, self.y)
