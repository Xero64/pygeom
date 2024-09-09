from .vector import Vector


class Point(Vector):
    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x, y, z)
    def __repr__(self) -> str:
        return '<Point: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __str__(self) -> str:
        frmstr = '({:}, {:}, {:})'
        return frmstr.format(self.x, self.y, self.z)
    def __format__(self, frm: str) -> str:
        frmstr = '({:' + frm + '}, {:' + frm + '}, {:' + frm + '})'
        return frmstr.format(self.x, self.y, self.z)
