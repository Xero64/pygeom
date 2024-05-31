from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from numpy import (arange, argsort, array, bool_, float64, hstack, int64,
                   logical_and, take_along_axis, unique, vstack, zeros)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class MetaCache2D():
    key: str = None
    dtype: 'DTypeLike' = None
    default: Any = None
    data: List[Any] = None

    def __init__(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.key = key
        self.dtype = dtype
        self.default = default
        self.data = []

    def clear(self) -> None:
        self.data.clear()

    def append(self, value: Any) -> None:
        self.data.append(value)

    def asarray(self) -> 'NDArray':
        arr = array(self.data, dtype=self.dtype)
        if len(arr.shape) == 1:
            return arr.reshape(-1, 1)
        return arr


class MeshGrids2D():
    grids: 'NDArray[float64]' = None
    meta: Dict[str, 'NDArray'] = None
    grids_cache: List[Tuple[float, float, float]] = None
    meta_cache: Dict[str, MetaCache2D] = None

    def __init__(self) -> None:
        self.grids = zeros((0, 2), dtype=float64)
        self.meta = {}
        self.grids_cache = []
        self.meta_cache = {}

    def add_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.meta_cache[key] = MetaCache2D(key, dtype, default)

    def add(self, x: float, y: float, **kwargs: Dict[str, Any]) -> None:
        self.grids_cache.append((x, y))
        for key in self.meta_cache.keys():
            value = kwargs.get(key, self.meta_cache[key].default)
            self.meta_cache[key].append(value)

    def clear_cache(self) -> None:
        self.grids_cache.clear()
        for value in self.meta_cache.values():
            value.clear()

    def resolve_cache(self) -> None:
        self.grids = array(self.grids_cache, dtype=float64)
        for key, value in self.meta_cache.items():
            self.meta[key] = value.asarray()
        self.clear_cache()

    def duplicate_indices(self) -> Tuple['NDArray[int64]',
                                         'NDArray[int64]']:
        if self.size == 0:
            return zeros(0, dtype=int64), zeros(0, dtype=int64)
        data = self.grids
        if len(self.meta) > 0:
            data = [data]
            for val in self.meta.values():
                if len(val.shape) == 1:
                    val = val.reshape(-1, 1)
                data.append(val)
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        return unind, invind

    def apply_inverse(self, invind: 'NDArray[int64]', key: str) -> None:
        if key == 'grids':
            raise ValueError('Cannot apply inverse to grids.')
        if key in self.meta:
            self.meta[key] = invind[self.meta[key]]

    def __getitem__(self, index: Any) -> 'MeshGrids2D':
        meshgrids = MeshGrids2D()
        meshgrids.grids = self.grids[index]
        meshgrids.meta = {}
        for key in self.meta.keys():
            meshgrids.meta[key] = self.meta[key][index]
        return meshgrids

    def __setitem__(self, index: Any, value: 'MeshGrids2D') -> None:
        try:
            self.grids[index] = value.grids
            for key in self.meta.keys():
                self.meta[key][index] = value.meta[key]
        except IndexError:
            err = 'MeshGrids2D index out of range.'
            raise IndexError(err)

    @property
    def size(self) -> int:
        return self.grids.shape[0]

    def __str__(self) -> str:
        outstr = f'MeshGrids2D: size = {self.size:d}, dtype = {self.grids.dtype}\n'
        outstr += f'grids: \n{self.grids:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

    def __repr__(self) -> str:
        return f'<MeshGrids2D: size = {self.size:d}>'


class MeshVectors2D():
    label: str = None
    name: str = None
    vecs: 'NDArray[float64]' = None
    meta: Dict[str, 'NDArray'] = None
    vecs_cache: List[Tuple[float, float, float]] = None
    meta_cache: Dict[str, MetaCache2D] = None

    def __init__(self, label: str, name: str = 'MeshVectors2D') -> None:
        self.label = label
        self.name = name
        self.vecs = zeros((0, 2), dtype=float64)
        self.meta = {}
        self.vecs_cache = []
        self.meta_cache = {}

    def add_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.meta_cache[key] = MetaCache2D(key, dtype, default)

    def add(self, x: float, y: float, **kwargs: Dict[str, Any]) -> None:
        self.vecs_cache.append((x, y))
        for key in self.meta_cache.keys():
            value = kwargs.get(key, self.meta_cache[key].default)
            self.meta_cache[key].append(value)

    def clear_cache(self) -> None:
        self.vecs_cache.clear()
        for value in self.meta_cache.values():
            value.clear()

    def resolve_cache(self) -> None:
        self.vecs = array(self.vecs_cache, dtype=float64)
        for key, value in self.meta_cache.items():
            self.meta[key] = value.asarray()
        self.clear_cache()

    def duplicate_indices(self) -> Tuple['NDArray[int64]',
                                         'NDArray[int64]']:
        if self.size == 0:
            return zeros(0, dtype=int64), zeros(0, dtype=int64)
        data = self.vecs
        if len(self.meta) > 0:
            data = [data]
            for val in self.meta.values():
                if len(val.shape) == 1:
                    val = val.reshape(-1, 1)
                data.append(val)
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        return unind, invind

    def apply_inverse(self, invind: 'NDArray[int64]', key: str) -> None:
        if key == self.label:
            raise ValueError(f'Cannot apply inverse to {self.label:s}.')
        if key in self.meta:
            self.meta[key] = invind[self.meta[key]]

    def __getitem__(self, index: Any) -> 'MeshVectors2D':
        meshvecs = MeshVectors2D(self.label, self.name)
        meshvecs.vecs = self.vecs[index]
        meshvecs.meta = {}
        for key in self.meta.keys():
            meshvecs.meta[key] = self.meta[key][index]
        return meshvecs

    def __setitem__(self, index: Any, value: 'MeshVectors2D') -> None:
        try:
            self.vecs[index] = value.vecs
            for key in self.meta.keys():
                self.meta[key][index] = value.meta[key]
        except IndexError:
            err = f'{self.name:s} index out of range.'
            raise IndexError(err)

    @property
    def size(self) -> int:
        return self.vecs.shape[0]

    def __str__(self) -> str:
        outstr = f'{self.name:s}: size = {self.size:d}, dtype = {self.vecs.dtype}\n'
        outstr += f'{self.label}: \n{self.vecs:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

    def __repr__(self) -> str:
        return f'<{self.name:s}: size = {self.size:d}>'


class MeshElems2D():
    name: str = 'MeshElems2D'
    desc: str = 'elems'
    numg: int = 0
    grids: 'NDArray[int64]' = None
    meta: Dict[str, 'NDArray'] = None
    grids_cache: List[Tuple[int, ...]] = None
    meta_cache: Dict[str, MetaCache2D] = None

    def __init__(self) -> None:
        self.grids = zeros((0, self.numg), dtype=int64)
        self.meta = {}
        self.grids_cache = []
        self.meta_cache = {}

    def add_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.meta_cache[key] = MetaCache2D(key, dtype, default)

    def add(self, *grids: int, **kwargs: Dict[str, Any]) -> None:
        self.grids_cache.append(grids)
        for key in self.meta_cache.keys():
            value = kwargs.get(key, self.meta_cache[key].default)
            self.meta_cache[key].append(value)

    def clear_cache(self) -> None:
        self.grids_cache.clear()
        for value in self.meta_cache.values():
            value.clear()

    def resolve_cache(self) -> None:
        self.grids = array(self.grids_cache, dtype=int64)
        for key, value in self.meta_cache.items():
            self.meta[key] = value.asarray()
        self.clear_cache()

    def duplicate_indices(self) -> Tuple['NDArray[int64]',
                                         'NDArray[int64]']:
        if self.size == 0:
            return None
        data = self.grids
        minindax1 = argsort(data, axis=1)
        data = take_along_axis(data, minindax1, axis=1)
        if len(self.meta) > 0:
            data = [data]
            for value in self.meta.values():
                if value.shape[1] == self.numg:
                    data.append(take_along_axis(value, minindax1, axis=1))
                else:
                    data.append(value)
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        return unind, invind

    def apply_inverse(self, invind: 'NDArray[int64]', key: str) -> None:
        if key == self.desc:
            raise ValueError(f'Cannot apply inverse to {self.desc:s}.')
        if key == 'grids':
            self.grids = invind[self.grids]
        if key in self.meta:
            self.meta[key] = invind[self.meta[key]]

    def remove_collapsed(self) -> None:
        if self.size == 0:
            return None
        ind1 = arange(-1, self.numg-1)
        ind2 = arange(0, self.numg)
        grids1 = self.grids[:, ind1]
        grids2 = self.grids[:, ind2]
        checkedge: 'NDArray[bool_]' = grids1 == grids2
        countelem: 'NDArray[int64]' = checkedge.sum(axis=1)
        count = countelem.flatten()
        if self.numg == 4:
            check = count < 2
            ind1 = array([0, 1], dtype=int64)
            ind2 = array([2, 3], dtype=int64)
            grids1 = self.grids[:, ind1]
            grids2 = self.grids[:, ind2]
            checkedge: 'NDArray[bool_]' = grids1 == grids2
            countelem: 'NDArray[int64]' = checkedge.sum(axis=1)
            count = countelem.flatten()
            checkdiag = count == 0
            check = logical_and(checkdiag, check)
        else:
            check = count == 0
        self.grids = self.grids[check, :]
        for key in self.meta.keys():
            self.meta[key] = self.meta[key][check, :]

    def __getitem__(self, index: Any) -> 'MeshElems2D':
        meshelems = self.__class__()
        meshelems.grids = self.grids[index, :]
        meshelems.meta = {}
        for key in self.meta.keys():
            meshelems.meta[key] = self.meta[key][index, :]
        return meshelems

    def __setitem__(self, index: Any, value: 'MeshElems2D') -> None:
        try:
            self.grids[index, :] = value.grids
            for key in self.meta.keys():
                self.meta[key][index, :] = value.meta[key]
        except IndexError:
            err = f'{self.name:s} index out of range.'
            raise IndexError(err)

    @property
    def size(self) -> int:
        return self.grids.shape[0]

    def __str__(self) -> str:
        outstr = f'{self.__class__.__qualname__:s}: size = {self.size:d}\n'
        outstr += f'grids: \n{self.grids:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

    def __repr__(self) -> str:
        return f'<{self.__class__.__qualname__:s}: size = {self.size:d}>'


class MeshLines2D(MeshElems2D):
    desc: str = 'lines'
    numg: int = 2


class MeshTrias2D(MeshElems2D):
    desc: str = 'trias'
    numg: int = 3


class MeshQuads2D(MeshElems2D):
    desc: str = 'quads'
    numg: int = 4


class Mesh2D():
    grids: MeshGrids2D = None
    lines: MeshLines2D = None
    trias: MeshTrias2D = None
    quads: MeshQuads2D = None
    mvecs: Dict[str, MeshVectors2D] = None

    def __init__(self) -> None:
        self.grids = MeshGrids2D()
        self.lines = MeshLines2D()
        self.trias = MeshTrias2D()
        self.quads = MeshQuads2D()
        self.mvecs = {}

    def getattr(self, key: str) -> Any:
        if key not in self.__dict__:
            if key in self.mvecs:
                return self.mvecs[key]
            else:
                raise AttributeError(f'Mesh has no attribute {key:s}.')
        else:
            return self.__dict__[key]

    def add_mesh_vectors(self, label: str, name: str) -> None:
        self.mvecs[label] = MeshVectors2D(label, name)

    def resolve_cache(self) -> None:
        self.grids.resolve_cache()
        self.lines.resolve_cache()
        self.trias.resolve_cache()
        self.quads.resolve_cache()
        for mvec in self.mvecs.values():
            mvec.resolve_cache()

    def remove_duplicate_grids(self) -> None:
        if self.grids.size == 0:
            return None
        unind, invind = self.grids.duplicate_indices()
        self.grids = self.grids[unind]
        self.lines.apply_inverse(invind, 'grids')
        self.trias.apply_inverse(invind, 'grids')
        self.quads.apply_inverse(invind, 'grids')
        for mvec in self.mvecs.values():
            mvec.apply_inverse(invind, 'grids')

    def remove_duplicate_vectors(self, label: str) -> None:
        mvec = self.mvecs[label]
        if mvec.size == 0:
            return None
        unind, invind = mvec.duplicate_indices()
        self.mvecs[label] = mvec[unind]
        self.grids.apply_inverse(invind, label)
        self.lines.apply_inverse(invind, label)
        self.trias.apply_inverse(invind, label)
        self.quads.apply_inverse(invind, label)
        for mlbl, mvec in self.mvecs.items():
            if mlbl != label:
                mvec.apply_inverse(invind, label)

    def remove_duplicate_lines(self) -> None:
        if self.lines.size == 0:
            return None
        unind, invind = self.lines.duplicate_indices()
        self.lines = self.lines[unind]
        self.grids.apply_inverse(invind, 'lines')
        self.trias.apply_inverse(invind, 'lines')
        self.quads.apply_inverse(invind, 'lines')
        for mvec in self.mvecs.values():
            mvec.apply_inverse(invind, 'lines')

    def remove_duplicate_trias(self) -> None:
        if self.trias.size == 0:
            return None
        unind, invind = self.trias.duplicate_indices()
        self.trias = self.trias[unind]
        self.grids.apply_inverse(invind, 'trias')
        self.lines.apply_inverse(invind, 'trias')
        self.quads.apply_inverse(invind, 'trias')
        for mvec in self.mvecs.values():
            mvec.apply_inverse(invind, 'trias')

    def remove_duplicate_quads(self) -> None:
        if self.quads.size == 0:
            return None
        unind, invind = self.quads.duplicate_indices()
        self.quads = self.quads[unind]
        self.grids.apply_inverse(invind, 'quads')
        self.lines.apply_inverse(invind, 'quads')
        self.trias.apply_inverse(invind, 'quads')
        for mvec in self.mvecs.values():
            mvec.apply_inverse(invind, 'quads')

    def collapse_quads_to_trias(self) -> None:
        ind1 = arange(-1, 3)
        ind2 = arange(0, 4)
        grids1 = self.quads.grids[:, ind1]
        grids2 = self.quads.grids[:, ind2]
        checkedge: 'NDArray[bool_]' = grids1 == grids2
        countelem: 'NDArray[int64]' = checkedge.sum(axis=1)
        count = countelem.flatten()
        triacheck = count == 1
        quadcheck = count != 1
        triagrids = self.quads.grids[triacheck, :]
        triameta = {}
        for key in self.quads.meta.keys():
            triameta[key] = self.quads.meta[key][triacheck, :]
        check_ab = triagrids[:, 0] == triagrids[:, 1]
        check_bc = triagrids[:, 1] == triagrids[:, 2]
        check_cd = triagrids[:, 2] == triagrids[:, 3]
        check_da = triagrids[:, 3] == triagrids[:, 0]
        tria_ab = triagrids[check_ab, :][:, (1, 2, 3)]
        tria_bc = triagrids[check_bc, :][:, (2, 3, 0)]
        tria_cd = triagrids[check_cd, :][:, (3, 0, 1)]
        tria_da = triagrids[check_da, :][:, (0, 1, 2)]
        trias = vstack((tria_ab, tria_bc, tria_cd, tria_da))
        for key in triameta.keys():
            metadata: 'NDArray' = triameta[key]
            if metadata.shape[1] == 4:
                meta_ab = metadata[check_ab, :][:, (1, 2, 3)]
                meta_bc = metadata[check_bc, :][:, (2, 3, 0)]
                meta_cd = metadata[check_cd, :][:, (3, 0, 1)]
                meta_da = metadata[check_da, :][:, (0, 1, 2)]
                metadata = vstack((meta_ab, meta_bc, meta_cd, meta_da))
            triameta[key] = metadata
        if self.trias.size == 0:
            self.trias.grids = trias
            self.trias.meta = triameta
        else:
            self.trias.grids = vstack((self.trias.grids, trias))
            for key in self.trias.meta.keys():
                self.trias.meta[key] = vstack((self.trias.meta[key], triameta[key]))
        self.quads.grids = self.quads.grids[quadcheck, :]
        for key in self.quads.meta.keys():
            self.quads.meta[key] = self.quads.meta[key][quadcheck, :]

    def remove_unreferenced_grids(self) -> None:
        if self.grids.size == 0:
            return None
        refind = []
        refind.append(self.lines.grids.flatten())
        refind.append(self.trias.grids.flatten())
        refind.append(self.quads.grids.flatten())
        if 'grids' in self.lines.meta:
            refind.append(self.lines.meta['grids'].flatten())
        if 'grids' in self.trias.meta:
            refind.append(self.trias.meta['grids'].flatten())
        if 'grids' in self.quads.meta:
            refind.append(self.quads.meta['grids'].flatten())
        for mvec in self.mvecs.values():
            if 'grids' in mvec.meta:
                refind.append(mvec.meta['grids'].flatten())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        invind = zeros(self.grids.size, dtype=int64)
        invind[refind] = arange(refind.size)
        self.grids = self.grids[refind]
        self.lines.apply_inverse(invind, 'grids')
        self.trias.apply_inverse(invind, 'grids')
        self.quads.apply_inverse(invind, 'grids')
        for mvec in self.mvecs.values():
            mvec.apply_inverse(invind, 'grids')

    def remove_unreferenced_vectors(self, label: str) -> None:
        mvec = self.mvecs[label]
        if mvec.size == 0:
            return None
        refind = []
        if mvec.label in self.grids.meta:
            refind.append(self.grids.meta[mvec.label].flatten())
        if mvec.label in self.lines.meta:
            refind.append(self.lines.meta[mvec.label].flatten())
        if mvec.label in self.trias.meta:
            refind.append(self.trias.meta[mvec.label].flatten())
        if mvec.label in self.quads.meta:
            refind.append(self.quads.meta[mvec.label].flatten())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        invind = zeros(mvec.size, dtype=int64)
        invind[refind] = arange(refind.size)
        self.mvecs[label] = mvec[refind]
        self.grids.apply_inverse(invind, mvec.label)
        self.lines.apply_inverse(invind, mvec.label)
        self.trias.apply_inverse(invind, mvec.label)
        self.quads.apply_inverse(invind, mvec.label)

    def merge(self, mesh: 'Mesh2D') -> 'Mesh2D':

        mergedmesh = Mesh2D()

        # Grids
        mergedmesh.grids.grids = vstack((self.grids.grids, mesh.grids.grids))

        # Grid Meta
        mergedmesh.grids.meta = {}
        for key, value in self.grids.meta.items():
            meshvalue = mesh.grids.meta[key]
            if key == 'lines':
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + self.lines.size))
            elif key == 'trias':
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + self.trias.size))
            elif key == 'quads':
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + self.quads.size))
            else:
                mergedmesh.grids.meta[key] = vstack((value, meshvalue))
            if key in self.mvecs:
                mvec = self.mvecs[key]
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + mvec.size))

        # Vectors
        for label, mvec in self.mvecs.items():

            mvecself = self.mvecs[label]
            mvecmesh = mesh.mvecs[label]

            mergedmesh.add_mesh_vectors(label, mvec.name)
            mvecmerged = mergedmesh.mvecs[label]
            mvecmerged.vecs = vstack((mvecself.vecs, mvecmesh.vecs))

            # Mesh Vectors Meta
            mvecmerged.meta = {}
            for key, value in mvecself.meta.items():
                meshvalue = mvecmesh.meta[key]
                if key == 'grids':
                    mvecmerged.meta[key] = vstack((value, meshvalue + self.grids.size))
                elif key == 'lines':
                    mvecmerged.meta[key] = vstack((value, meshvalue + self.lines.size))
                elif key == 'trias':
                    mvecmerged.meta[key] = vstack((value, meshvalue + self.trias.size))
                elif key == 'quads':
                    mvecmerged.meta[key] = vstack((value, meshvalue + self.quads.size))
                else:
                    mvecmerged.meta[key] = vstack((value, meshvalue))
                if key in self.mvecs:
                    mvec = self.mvecs[key]
                    mvecmerged.meta[key] = vstack((value, meshvalue + mvec.size))

        # Lines
        mesh_lines_grids = mesh.lines.grids + self.grids.size
        mesh_lines_grids = mesh_lines_grids.reshape(-1, 2)
        mergedmesh.lines.grids = vstack((self.lines.grids, mesh_lines_grids))

        # Line Meta
        mergedmesh.lines.meta = {}
        for key, value in self.lines.meta.items():
            meshvalue = mesh.lines.meta[key]
            if key == 'grids':
                mergedmesh.lines.meta[key] = vstack((value, meshvalue + self.grids.size))
            elif key == 'trias':
                mergedmesh.lines.meta[key] = vstack((value, meshvalue + self.trias.size))
            elif key == 'quads':
                mergedmesh.lines.meta[key] = vstack((value, meshvalue + self.quads.size))
            else:
                mergedmesh.lines.meta[key] = vstack((value, meshvalue))
            if key in self.mvecs:
                mvec = self.mvecs[key]
                mergedmesh.lines.meta[key] = vstack((value, meshvalue + mvec.size))

        # Trias
        mesh_trias_grids = mesh.trias.grids + self.grids.size
        mesh_trias_grids = mesh_trias_grids.reshape(-1, 3)
        mergedmesh.trias.grids = vstack((self.trias.grids, mesh_trias_grids))

        # Tria Meta
        mergedmesh.trias.meta = {}
        for key, value in self.trias.meta.items():
            meshvalue = mesh.trias.meta[key]
            if key == 'grids':
                mergedmesh.trias.meta[key] = vstack((value, meshvalue + self.grids.size))
            elif key == 'lines':
                mergedmesh.trias.meta[key] = vstack((value, meshvalue + self.lines.size))
            elif key == 'quads':
                mergedmesh.trias.meta[key] = vstack((value, meshvalue + self.quads.size))
            else:
                mergedmesh.trias.meta[key] = vstack((value, meshvalue))
            if key in self.mvecs:
                mvec = self.mvecs[key]
                mergedmesh.trias.meta[key] = vstack((value, meshvalue + mvec.size))

        # Quads
        mesh_quads_grids = mesh.quads.grids + self.grids.size
        mesh_quads_grids = mesh_quads_grids.reshape(-1, 4)
        mergedmesh.quads.grids = vstack((self.quads.grids, mesh_quads_grids))

        # Quad Meta
        mergedmesh.quads.meta = {}
        for key, value in self.quads.meta.items():
            meshvalue = mesh.quads.meta[key]
            if key == 'grids':
                mergedmesh.quads.meta[key] = vstack((value, meshvalue + self.grids.size))
            elif key == 'lines':
                mergedmesh.quads.meta[key] = vstack((value, meshvalue + self.lines.size))
            elif key == 'trias':
                mergedmesh.quads.meta[key] = vstack((value, meshvalue + self.trias.size))
            else:
                mergedmesh.quads.meta[key] = vstack((value, meshvalue))
            if key in self.mvecs:
                mvec = self.mvecs[key]
                mergedmesh.quads.meta[key] = vstack((value, meshvalue + mvec.size))

        return mergedmesh

    def __str__(self) -> str:
        outstr = ''
        if self.grids is not None:
            outstr += f'{self.grids}\n'
        if self.lines is not None:
            outstr += f'{self.lines}\n'
        if self.trias is not None:
            outstr += f'{self.trias}\n'
        if self.quads is not None:
            outstr += f'{self.quads}\n'
        for mvec in self.mvecs.values():
            outstr += f'{mvec}\n'
        outstr += '\n'
        return outstr

    def __repr__(self) -> str:
        return '<Mesh2D>'
