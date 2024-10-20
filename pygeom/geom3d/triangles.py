from typing import TYPE_CHECKING

from ..geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class Triangle():
    """Triangle Class"""

    pnta: Vector = None
    pntb: Vector = None
    pntc: Vector = None
    _pnto: Vector = None
    _vecab: Vector = None
    _vecbc: Vector = None
    _nrm: Vector = None
    _jac: float = None

    def __init__(self, pnta: Vector, pntb: Vector, pntc: Vector) -> None:
        self.pnta = pnta
        self.pntb = pntb
        self.pntc = pntc

    @property
    def pnto(self) -> Vector:
        if self._pnto is None:
            self._pnto = (self.pnta + self.pntb + self.pntc)/3
        return self._pnto

    @pnto.setter
    def pnto(self, value: Vector) -> None:
        self._pnto = value

    @property
    def vecab(self) -> Vector:
        if self._vecab is None:
            self._vecab = self.pntb - self.pnta
        return self._vecab

    @vecab.setter
    def vecab(self, value: Vector) -> None:
        self._vecab = value

    @property
    def vecbc(self) -> Vector:
        if self._vecbc is None:
            self._vecbc = self.pntc - self.pntb
        return self._vecbc

    @vecbc.setter
    def vecbc(self, value: Vector) -> None:
        self._vecbc = value

    @property
    def nrm(self) -> Vector:
        if self._nrm is None:
            nrmvec = self.vecab.cross(self.vecbc)
            self._nrm, self._jac = nrmvec.to_unit(return_magnitude=True)
        return self._nrm

    @nrm.setter
    def nrm(self, value: Vector) -> None:
        self._nrm = value

    @property
    def jac(self) -> float:
        if self._jac is None:
            self.nrm
        return self._jac

    @jac.setter
    def jac(self, value: float) -> None:
        self._jac = value


class Triangles():
    """Triangles Class"""

    pnta: Vector = None
    pntb: Vector = None
    pntc: Vector = None
    _pnto: 'NDArray' = None
    _vecab: 'NDArray' = None
    _vecbc: 'NDArray' = None
    _nrm: 'NDArray' = None
    _jac: 'NDArray' = None

    def __init__(self, pnta: Vector, pntb: Vector, pntc: Vector) -> None:
        self.pnta = pnta
        self.pntb = pntb
        self.pntc = pntc

    @property
    def pnto(self) -> Vector:
        if self._pnto is None:
            self._pnto = (self.pnta + self.pntb + self.pntc)/3
        return self._pnto

    @property
    def vecab(self) -> Vector:
        if self._vecab is None:
            self._vecab = self.pntb - self.pnta
        return self._vecab

    @property
    def vecbc(self) -> Vector:
        if self._vecbc is None:
            self._vecbc = self.pntc - self.pntb
        return self._vecbc

    @property
    def nrm(self) -> Vector:
        if self._nrm is None:
            nrmvec = self.vecab.cross(self.vecbc)
            self._nrm, self._jac = nrmvec.to_unit(return_magnitude=True)
        return self._nrm

    @property
    def jac(self) -> 'NDArray':
        if self._jac is None:
            self.nrm
        return self._jac

    def __getitem__(self, key: int) -> 'Triangle | Triangles':
        pnta = self.pnta[key]
        pntb = self.pntb[key]
        pntc = self.pntc[key]

        if isinstance(pnta, Vector):
            output = Triangle(pnta, pntb, pntc)
        else:
            output = Triangles(pnta, pntb, pntc)

        for attr in self.__dict__:
            if attr[0] != '_':
                if self.__dict__[attr] is not None:
                 output.__dict__[attr] = self.__dict__[attr][key]

        return output

    def __setitem__(self, key, value: 'Triangle | Triangles') -> None:
        try:
            self.pnta[key] = value.pnta
            self.pntb[key] = value.pntb
            self.pntc[key] = value.pntc

            for attr in self.__dict__:
                if attr[0] != '_':
                    if self.__dict__[attr] is not None:
                        self.__dict__[attr][key] = value.__dict__[attr]

        except IndexError:
            err = 'Triangles index out of range.'
            raise IndexError(err)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.pnta.shape == self.pntb.shape and self.pntb.shape == self.pntc.shape:
            return self.pnta.shape
        else:
            raise ValueError('Triangle pnts should have the same shape.')

    @property
    def dtype(self) -> 'DTypeLike':
        if self.pnta.dtype is self.pntb.dtype and self.pntb.dtype is self.pntc.dtype:
            return self.pnta.dtype
        else:
            raise ValueError('Triangle pnts should have the same dtype.')

    @property
    def ndim(self) -> int:
        if self.pnta.ndim == self.pntb.ndim and self.pntb.ndim == self.pntc.ndim:
            return self.pnta.ndim
        else:
            raise ValueError('Triangle pnts should have the same ndim.')

    @property
    def size(self) -> int:
        if self.pnta.size == self.pntb.size and self.pntb.size == self.pntc.size:
            return self.pnta.size
        else:
            raise ValueError('Triangle pnts should have the same size.')

    def transpose(self) -> Vector:
        pnta = self.pnta.transpose()
        pntb = self.pntb.transpose()
        pntc = self.pntc.transpose()
        return Triangles(pnta, pntb, pntc)

    def sum(self, axis=None, dtype=None, out=None) -> 'Triangle | Triangles':
        pnta = self.pnta.sum(axis=axis, dtype=dtype, out=out)
        pntb = self.pntb.sum(axis=axis, dtype=dtype, out=out)
        pntc = self.pntc.sum(axis=axis, dtype=dtype, out=out)

        if isinstance(pnta, Vector) and isinstance(pntb, Vector) and isinstance(pntc, Vector):
            return Triangle(pnta, pntb, pntc)
        else:
            return Triangles(pnta, pntb, pntc)

    def repeat(self, repeats, axis=None) -> 'Triangles':
        pnta = self.pnta.repeat(repeats, axis=axis)
        pntb = self.pntb.repeat(repeats, axis=axis)
        pntc = self.pntc.repeat(repeats, axis=axis)
        return Triangles(pnta, pntb, pntc)

    def reshape(self, shape, order='C') -> 'Triangles':
        pnta = self.pnta.reshape(shape, order=order)
        pntb = self.pntb.reshape(shape, order=order)
        pntc = self.pntc.reshape(shape, order=order)
        return Triangles(pnta, pntb, pntc)
