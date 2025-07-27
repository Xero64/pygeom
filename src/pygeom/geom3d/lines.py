from typing import TYPE_CHECKING

from ..geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

class Line():
    """Line Class"""

    pnta: Vector = None
    pntb: Vector = None
    _lvec: Vector = None
    _lmag: 'NDArray' = None
    _ldir: Vector = None

    def __init__(self, pnta: Vector, pntb: Vector) -> None:
        self.pnta = pnta
        self.pntb = pntb

    @property
    def lvec(self) -> Vector:
        if self._lvec is None:
            self._lvec = self.pntb - self.pnta
        return self._lvec

    @property
    def lmag(self) -> 'NDArray':
        if self._lmag is None:
            self._lmag = self.lvec.return_magnitude()
        return self._lmag

    @property
    def ldir(self) -> Vector:
        if self._ldir is None:
            if self._lmag is None:
                self._ldir, self._lmag = self.lvec.to_unit(return_magnitude=True)
            else:
                self._ldir = self.lvec/self.lmag
        return self._ldir

class Lines():
    """Lines Class"""

    pnta: Vector = None
    pntb: Vector = None
    _lvec: Vector = None
    _lmag: 'NDArray' = None
    _ldir: Vector = None

    def __init__(self, pnta: Vector, pntb: Vector) -> None:
        self.pnta = pnta
        self.pntb = pntb

    @property
    def lvec(self) -> Vector:
        if self._lvec is None:
            self._lvec = self.pntb - self.pnta
        return self._lvec

    @property
    def lmag(self) -> 'NDArray':
        if self._lmag is None:
            self._lmag = self.lvec.return_magnitude()
        return self._lmag

    @property
    def ldir(self) -> Vector:
        if self._ldir is None:
            if self._lmag is None:
                self._ldir, self._lmag = self.lvec.to_unit(return_magnitude=True)
            else:
                self._ldir = self.lvec/self.lmag
        return self._ldir

    def __getitem__(self, key: int) -> 'Line | Lines':
        pnta = self.pnta[key]
        pntb = self.pntb[key]

        if isinstance(pnta, Vector) and isinstance(pntb, Vector):
            output = Line(pnta, pntb)
        else:
            output = Lines(pnta, pntb)

        for attr in self.__dict__:
            if attr[0] != '_':
                if self.__dict__[attr] is not None:
                 output.__dict__[attr] = self.__dict__[attr][key]

        return output

    def __setitem__(self, key, value: 'Line | Lines') -> None:
        try:
            self.pnta[key] = value.pnta
            self.pntb[key] = value.pntb

            for attr in self.__dict__:
                if attr[0] != '_':
                    if self.__dict__[attr] is not None:
                        self.__dict__[attr][key] = value.__dict__[attr]

        except IndexError:
            err = 'Lines index out of range.'
            raise IndexError(err)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.pnta.shape == self.pntb.shape:
            return self.pnta.shape
        else:
            raise ValueError('Line pnts should have the same shape.')

    @property
    def dtype(self) -> 'DTypeLike':
        if self.pnta.dtype is self.pntb.dtype:
            return self.pnta.dtype
        else:
            raise ValueError('Line pnts should have the same dtype.')

    @property
    def ndim(self) -> int:
        if self.pnta.ndim == self.pntb.ndim:
            return self.pnta.ndim
        else:
            raise ValueError('Line pnts should have the same ndim.')

    @property
    def size(self) -> int:
        if self.pnta.size == self.pntb.size:
            return self.pnta.size
        else:
            raise ValueError('Line pnts should have the same size.')

    def transpose(self) -> Vector:
        pnta = self.pnta.transpose()
        pntb = self.pntb.transpose()
        return Lines(pnta, pntb)

    def sum(self, axis=None, dtype=None, out=None) -> 'Line | Lines':
        pnta = self.pnta.sum(axis=axis, dtype=dtype, out=out)
        pntb = self.pntb.sum(axis=axis, dtype=dtype, out=out)

        if isinstance(pnta, Vector) and isinstance(pntb, Vector):
            return Line(pnta, pntb)
        else:
            return Lines(pnta, pntb)

    def repeat(self, repeats, axis=None) -> 'Lines':
        pnta = self.pnta.repeat(repeats, axis=axis)
        pntb = self.pntb.repeat(repeats, axis=axis)
        return Lines(pnta, pntb)

    def reshape(self, shape, order='C') -> 'Lines':
        pnta = self.pnta.reshape(shape, order=order)
        pntb = self.pntb.reshape(shape, order=order)
        return Lines(pnta, pntb)
