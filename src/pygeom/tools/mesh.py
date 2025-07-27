from typing import TYPE_CHECKING, Any

from numpy import (arange, argsort, asarray, bool_, hstack, int64, logical_and,
                   round, take_along_axis, unique, vstack, zeros)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class MetaCache():
    key: str = None
    dtype: 'DTypeLike' = None
    default: Any = None
    data: list[Any] = None

    def __init__(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.key = key
        self.dtype = dtype
        self.default = default
        self.data = []

    @property
    def size(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.data.clear()

    def append(self, value: Any) -> None:
        self.data.append(value)

    def asarray(self) -> 'NDArray':
        arr = asarray(self.data, dtype=self.dtype)
        if len(arr.shape) == 1:
            return arr.reshape(-1, 1)
        return arr

    def __repr__(self) -> str:
        outstr = '<MetaCache'
        outstr += f': key = {self.key:s}'
        outstr += f', dtype = {self.dtype}'
        outstr += f', default = {self.default}'
        outstr += f', size = {self.size:d}'
        outstr += '>'
        return outstr

    def __str__(self) -> str:
        outstr = 'MetaCache'
        outstr += f': key = {self.key:s}'
        outstr += f', dtype = {self.dtype}'
        outstr += f', default = {self.default}'
        outstr += f', size = {self.size:d}'
        outstr += '\n'
        return outstr


class MeshObject():
    meta: dict[str, 'NDArray'] = None
    meta_cache: dict[str, MetaCache] = None

    def __init__(self) -> None:
        self.meta = {}
        self.meta_cache = {}

    def add_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        if key in self.__dict__:
            raise ValueError(f'Cannot add meta with key {key:s}.')
        self.meta_cache[key] = MetaCache(key, dtype, default)
        self.meta[key] = zeros(0, dtype=dtype)


class MeshVectors(MeshObject):
    ndim: int = 3
    name: str = None
    label: str = None
    vecs: 'NDArray' = None
    vecs_cache: list[tuple[float, float, float]] = None

    def __init__(self, label: str, name: str = 'MeshVectors') -> None:
        self.name = name
        self.label = label
        super().__init__()
        self.vecs = zeros((0, self.ndim))
        self.vecs_cache = []

    def add(self, x: float, y: float, z: float, **kwargs: dict[str, Any]) -> None:
        self.vecs_cache.append((x, y, z))
        for key in self.meta_cache.keys():
            value = kwargs.get(key, self.meta_cache[key].default)
            self.meta_cache[key].append(value)

    def clear_cache(self) -> None:
        self.vecs_cache.clear()
        for value in self.meta_cache.values():
            value.clear()

    def resolve_cache(self) -> None:
        self.vecs = asarray(self.vecs_cache)
        for key, value in self.meta_cache.items():
            self.meta[key] = value.asarray()
        self.clear_cache()

    def append_cache(self) -> None:
        vecs = asarray(self.vecs_cache)
        meta = {}
        for key, value in self.meta_cache.items():
            meta[key] = value.asarray()
        vec_data = []
        meta_data = {mkey: [] for mkey in meta.keys()}
        if self.size > 0:
            vec_data.append(self.vecs)
            for key in self.meta.keys():
                meta_data[key].append(self.meta[key])
        if len(vecs) > 0:
            vec_data.append(vecs)
            for key in meta.keys():
                meta_data[key].append(meta[key])
        if len(vec_data) > 0:
            self.vecs = vstack(tuple(vec_data))
            for key in meta.keys():
                if len(meta_data[key]) > 0:
                    self.meta[key] = vstack(tuple(meta_data[key]))
        self.clear_cache()

    def duplicate_indices(self, decimals: int | None = None) -> tuple['NDArray[int64]',
                                                                      'NDArray[int64]']:
        if self.size == 0:
            return zeros(0, dtype=int64), zeros(0, dtype=int64)
        if decimals is not None:
            data = round(self.vecs, decimals=decimals)
        else:
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

    def __getitem__(self, index: Any) -> 'MeshVectors':
        meshvecs = self.__class__(self.label, self.name)
        meshvecs.vecs_cache = self.vecs_cache
        meshvecs.meta_cache = self.meta_cache
        meshvecs.vecs = self.vecs[index]
        meshvecs.meta = {}
        for key in self.meta.keys():
            meshvecs.meta[key] = self.meta[key][index]
        return meshvecs

    def __setitem__(self, index: Any, value: 'MeshVectors') -> None:
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

    @property
    def full_str(self) -> str:
        outstr = f'{self.name:s}: size = {self.size:d}, dtype = {self.vecs.dtype}\n'
        outstr += f'{self.label}: \n{self.vecs:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

    def __str__(self) -> str:
        return f'{self.name:s}: size = {self.size:d}, dtype = {self.vecs.dtype}\n'

    def __repr__(self) -> str:
        return f'<{self.name:s}: size = {self.size:d}>'


class MeshVectors2D(MeshVectors):
    ndim: int = 2

    def __init__(self, label: str, name: str = 'MeshVectors2D') -> None:
        super().__init__(label, name)

    def add(self, x: float, y: float, **kwargs: dict[str, Any]) -> None:
        self.vecs_cache.append((x, y))
        for key in self.meta_cache.keys():
            value = kwargs.get(key, self.meta_cache[key].default)
            self.meta_cache[key].append(value)

    def __getitem__(self, index: Any) -> 'MeshVectors2D':
        return super().__getitem__(index)


class MeshGrids(MeshVectors):
    name: str = 'MeshGrids'
    label: str = 'grids'

    def __init__(self, label: str = 'grids', name: str = 'MeshGrids') -> None:
        super().__init__(label, name)


class MeshGrids2D(MeshVectors2D):
    name: str = 'MeshGrids2D'
    label: str = 'grids'

    def __init__(self, label: str = 'grids', name: str = 'MeshGrids') -> None:
        super().__init__(label, name)


class MeshElems(MeshObject):
    name: str = 'MeshElems'
    desc: str = 'elems'
    numg: int = 0
    grids: 'NDArray[int64]' = None
    grids_cache: list[tuple[int, ...]] = None

    def __init__(self) -> None:
        super().__init__()
        self.grids = zeros((0, self.numg), dtype=int64)
        self.grids_cache = []

    def add(self, *grids: int, **kwargs: dict[str, Any]) -> None:
        self.grids_cache.append(grids)
        for key in self.meta_cache.keys():
            value = kwargs.get(key, self.meta_cache[key].default)
            self.meta_cache[key].append(value)

    def clear_cache(self) -> None:
        self.grids_cache.clear()
        for value in self.meta_cache.values():
            value.clear()

    def resolve_cache(self) -> None:
        self.grids = asarray(self.grids_cache, dtype=int64)
        for key, value in self.meta_cache.items():
            self.meta[key] = value.asarray()
        self.clear_cache()

    def append_cache(self) -> None:
        grids = asarray(self.grids_cache, dtype=int64)
        meta = {}
        for key, value in self.meta_cache.items():
            meta[key] = value.asarray()
        grid_data = []
        meta_data = {mkey: [] for mkey in meta.keys()}
        if self.size > 0:
            grid_data.append(self.grids)
            for key in self.meta.keys():
                meta_data[key].append(self.meta[key])
        if len(grids) > 0:
            grid_data.append(grids)
            for key in meta.keys():
                meta_data[key].append(meta[key])
        if len(grid_data) > 0:
            self.grids = vstack(tuple(grid_data))
            for key in meta.keys():
                if len(meta_data[key]) > 0:
                    self.meta[key] = vstack(tuple(meta_data[key]))
        self.clear_cache()

    def duplicate_indices(self) -> tuple['NDArray[int64]',
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
        count = countelem.ravel()
        if self.numg == 4:
            check = count < 2
            ind1 = asarray([0, 1], dtype=int64)
            ind2 = asarray([2, 3], dtype=int64)
            grids1 = self.grids[:, ind1]
            grids2 = self.grids[:, ind2]
            checkedge: 'NDArray[bool_]' = grids1 == grids2
            countelem: 'NDArray[int64]' = checkedge.sum(axis=1)
            count = countelem.ravel()
            checkdiag = count == 0
            check = logical_and(checkdiag, check)
        else:
            check = count == 0
        self.grids = self.grids[check, :]
        for key in self.meta.keys():
            self.meta[key] = self.meta[key][check, :]

    def __getitem__(self, index: Any) -> 'MeshElems':
        meshelems = self.__class__()
        meshelems.grids_cache = self.grids_cache
        meshelems.meta_cache = self.meta_cache
        meshelems.grids = self.grids[index, :]
        meshelems.meta = {}
        for key in self.meta.keys():
            meshelems.meta[key] = self.meta[key][index, :]
        return meshelems

    def __setitem__(self, index: Any, value: 'MeshElems') -> None:
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

    @property
    def full_str(self) -> str:
        outstr = f'{self.name:s}: size = {self.size:d}\n'
        outstr += f'grids: \n{self.grids:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

    def __str__(self) -> str:
        return f'{self.name:s}: size = {self.size:d}\n'

    def __repr__(self) -> str:
        return f'<{self.name:s}: size = {self.size:d}>'


class MeshLines(MeshElems):
    name: str = 'MeshLines'
    desc: str = 'lines'
    numg: int = 2


class MeshTrias(MeshElems):
    name: str = 'MeshTrias'
    desc: str = 'trias'
    numg: int = 3


class MeshQuads(MeshElems):
    name: str = 'MeshQuads'
    desc: str = 'quads'
    numg: int = 4


class Mesh():
    ndim: int = 3
    grids: MeshGrids = None
    lines: MeshLines = None
    trias: MeshTrias = None
    quads: MeshQuads = None
    attrs: dict[str, MeshVectors] = None

    def __init__(self) -> None:
        if self.ndim == 3:
            self.grids = MeshGrids()
        elif self.ndim == 2:
            self.grids = MeshGrids2D()
        else:
            raise ValueError('Invalid ndim.')
        self.lines = MeshLines()
        self.trias = MeshTrias()
        self.quads = MeshQuads()
        self.attrs = {}

    def getattr(self, key: str) -> Any:
        if key not in self.__dict__:
            if key in self.attrs:
                return self.attrs[key]
            else:
                raise AttributeError(f'Mesh has no attribute {key:s}.')
        else:
            return self.__dict__[key]

    def add_mesh_vectors(self, label: str, name: str) -> None:
        if self.ndim == 3:
            self.attrs[label] = MeshVectors(label, name)
        elif self.ndim == 2:
            self.attrs[label] = MeshVectors2D(label, name)
        else:
            raise ValueError('Invalid ndim.')

    def resolve_cache(self) -> None:
        self.grids.resolve_cache()
        self.lines.resolve_cache()
        self.trias.resolve_cache()
        self.quads.resolve_cache()
        for attr in self.attrs.values():
            attr.resolve_cache()

    def append_cache(self) -> None:
        self.grids.append_cache()
        self.lines.append_cache()
        self.trias.append_cache()
        self.quads.append_cache()
        for attr in self.attrs.values():
            attr.append_cache()

    def remove_duplicate_grids(self, decimals: int | None = None) -> None:
        if self.grids.size == 0:
            return None
        unind, invind = self.grids.duplicate_indices(decimals=decimals)
        self.grids = self.grids[unind]
        self.lines.apply_inverse(invind, 'grids')
        self.trias.apply_inverse(invind, 'grids')
        self.quads.apply_inverse(invind, 'grids')
        for attr in self.attrs.values():
            attr.apply_inverse(invind, 'grids')

    def remove_duplicate_vectors(self, label: str,
                                 decimals: int | None = None) -> None:
        attr = self.attrs[label]
        if attr.size == 0:
            return None
        unind, invind = attr.duplicate_indices(decimals=decimals)
        self.attrs[label] = attr[unind]
        self.grids.apply_inverse(invind, label)
        self.lines.apply_inverse(invind, label)
        self.trias.apply_inverse(invind, label)
        self.quads.apply_inverse(invind, label)
        for akey, attr in self.attrs.items():
            if akey != label:
                attr.apply_inverse(invind, label)

    def remove_duplicate_lines(self) -> None:
        if self.lines.size == 0:
            return None
        unind, invind = self.lines.duplicate_indices()
        self.lines = self.lines[unind]
        self.grids.apply_inverse(invind, 'lines')
        self.trias.apply_inverse(invind, 'lines')
        self.quads.apply_inverse(invind, 'lines')
        for attr in self.attrs.values():
            attr.apply_inverse(invind, 'lines')

    def remove_duplicate_trias(self) -> None:
        if self.trias.size == 0:
            return None
        unind, invind = self.trias.duplicate_indices()
        self.trias = self.trias[unind]
        self.grids.apply_inverse(invind, 'trias')
        self.lines.apply_inverse(invind, 'trias')
        self.quads.apply_inverse(invind, 'trias')
        for attr in self.attrs.values():
            attr.apply_inverse(invind, 'trias')

    def remove_duplicate_quads(self) -> None:
        if self.quads.size == 0:
            return None
        unind, invind = self.quads.duplicate_indices()
        self.quads = self.quads[unind]
        self.grids.apply_inverse(invind, 'quads')
        self.lines.apply_inverse(invind, 'quads')
        self.trias.apply_inverse(invind, 'quads')
        for attr in self.attrs.values():
            attr.apply_inverse(invind, 'quads')

    def collapse_quads_to_trias(self) -> None:
        ind1 = arange(-1, 3)
        ind2 = arange(0, 4)
        grids1 = self.quads.grids[:, ind1]
        grids2 = self.quads.grids[:, ind2]
        checkedge: 'NDArray[bool_]' = grids1 == grids2
        countelem: 'NDArray[int64]' = checkedge.sum(axis=1)
        count = countelem.ravel()
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
        refind.append(self.lines.grids.ravel())
        refind.append(self.trias.grids.ravel())
        refind.append(self.quads.grids.ravel())
        if 'grids' in self.lines.meta:
            refind.append(self.lines.meta['grids'].ravel())
        if 'grids' in self.trias.meta:
            refind.append(self.trias.meta['grids'].ravel())
        if 'grids' in self.quads.meta:
            refind.append(self.quads.meta['grids'].ravel())
        for attr in self.attrs.values():
            if 'grids' in attr.meta:
                refind.append(attr.meta['grids'].ravel())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        invind = zeros(self.grids.size, dtype=int64)
        invind[refind] = arange(refind.size)
        self.grids = self.grids[refind]
        self.lines.apply_inverse(invind, 'grids')
        self.trias.apply_inverse(invind, 'grids')
        self.quads.apply_inverse(invind, 'grids')
        for attr in self.attrs.values():
            attr.apply_inverse(invind, 'grids')

    def remove_unreferenced_vectors(self, label: str) -> None:
        attr = self.attrs[label]
        if attr.size == 0:
            return None
        refind = []
        if attr.label in self.grids.meta:
            refind.append(self.grids.meta[attr.label].ravel())
        if attr.label in self.lines.meta:
            refind.append(self.lines.meta[attr.label].ravel())
        if attr.label in self.trias.meta:
            refind.append(self.trias.meta[attr.label].ravel())
        if attr.label in self.quads.meta:
            refind.append(self.quads.meta[attr.label].ravel())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        invind = zeros(attr.size, dtype=int64)
        invind[refind] = arange(refind.size)
        self.attrs[label] = attr[refind]
        self.grids.apply_inverse(invind, attr.label)
        self.lines.apply_inverse(invind, attr.label)
        self.trias.apply_inverse(invind, attr.label)
        self.quads.apply_inverse(invind, attr.label)

    def merge(self, mesh: 'Mesh | Mesh2D') -> 'Mesh | Mesh2D':

        mergedmesh = merge_meshes(self, mesh)

        return mergedmesh

    def new_mesh_from_template(self) -> 'Mesh | Mesh2D':

        newmesh = self.__class__()
        for key in self.grids.meta:
            meta_cache = self.grids.meta_cache[key]
            newmesh.grids.add_meta(meta_cache.key, meta_cache.dtype,
                                   meta_cache.default)
        for key in self.lines.meta:
            meta_cache = self.lines.meta_cache[key]
            newmesh.lines.add_meta(meta_cache.key, meta_cache.dtype,
                                   meta_cache.default)
        for key in self.trias.meta:
            meta_cache = self.trias.meta_cache[key]
            newmesh.trias.add_meta(meta_cache.key, meta_cache.dtype,
                                   meta_cache.default)
        for key in self.quads.meta:
            meta_cache = self.quads.meta_cache[key]
            newmesh.quads.add_meta(meta_cache.key, meta_cache.dtype,
                                   meta_cache.default)
        for mkey in self.attrs:
            newmesh.add_mesh_vectors(self.attrs[mkey].label,
                                     self.attrs[mkey].name)
            for key in self.attrs[mkey].meta:
                meta_cache = self.attrs[mkey].meta_cache[key]
                newmesh.attrs[mkey].add_meta(meta_cache.key, meta_cache.dtype,
                                             meta_cache.default)
        newmesh.resolve_cache()

        return newmesh

    def compare_mesh_template(self, mesh: 'Mesh | Mesh2D') -> bool:

        if self.ndim != mesh.ndim:
            return False
        if self.grids.meta.keys() != mesh.grids.meta.keys():
            return False
        if self.lines.meta.keys() != mesh.lines.meta.keys():
            return False
        if self.trias.meta.keys() != mesh.trias.meta.keys():
            return False
        if self.quads.meta.keys() != mesh.quads.meta.keys():
            return False
        for mkey in self.attrs:
            if mkey not in mesh.attrs:
                return False
            if self.attrs[mkey].meta.keys() != mesh.attrs[mkey].meta.keys():
                return False

        return True

    @property
    def mesh_template(self) -> str:
        outstr = 'Mesh Template:\n'
        outstr += 'grids:\n'
        for mkey in self.grids.meta.keys():
            outstr += f'  {mkey:s}\n'
        outstr += 'lines:\n'
        for mkey in self.lines.meta.keys():
            outstr += f'  {mkey:s}\n'
        outstr += 'trias:\n'
        for mkey in self.trias.meta.keys():
            outstr += f'  {mkey:s}\n'
        outstr += 'quads:\n'
        for mkey in self.quads.meta.keys():
            outstr += f'  {mkey:s}\n'
        for akey in self.attrs:
            outstr += f'{akey:s}:\n'
            for mkey in self.attrs[akey].meta.keys():
                outstr += f'  {mkey:s}\n'
        outstr += '\n'
        return outstr

    @property
    def full_str(self) -> str:
        outstr = ''
        if self.grids is not None:
            outstr += f'{self.grids.full_str}\n'
        if self.lines is not None:
            outstr += f'{self.lines.full_str}\n'
        if self.trias is not None:
            outstr += f'{self.trias.full_str}\n'
        if self.quads is not None:
            outstr += f'{self.quads.full_str}\n'
        for attr in self.attrs.values():
            outstr += f'{attr.full_str}\n'
        outstr += '\n'
        return outstr

    def __str__(self) -> str:
        outstr = ''
        if self.grids is not None:
            outstr += f'{self.grids}'
        if self.lines is not None:
            outstr += f'{self.lines}'
        if self.trias is not None:
            outstr += f'{self.trias}'
        if self.quads is not None:
            outstr += f'{self.quads}'
        for attr in self.attrs.values():
            outstr += f'{attr}'
        outstr += '\n'
        return outstr

    def __repr__(self) -> str:
        return '<Mesh>'


class Mesh2D(Mesh):
    ndim: int = 2
    grids: MeshGrids2D = None
    attrs: dict[str, MeshVectors2D] = None

    def add_mesh_vectors(self, label: str, name: str) -> None:
        self.attrs[label] = MeshVectors2D(label, name)


def merge_meshes(*meshes: Mesh) -> Mesh:

    if len(meshes) == 0:
        raise ValueError('No meshes to merge.')

    # Create new mesh from template
    mergedmesh = meshes[0].new_mesh_from_template()

    # Check if meshes have the same template
    for mesh in meshes[1:]:
        if not mergedmesh.compare_mesh_template(mesh):
            raise ValueError('Meshes have different templates.')

    grids_offset = 0
    lines_offset = 0
    trias_offset = 0
    quads_offset = 0
    attrs_offset = {akey: 0 for akey in mergedmesh.attrs.keys()}

    grids = []
    lines = []
    trias = []
    quads = []
    attrs = {akey: [] for akey in mergedmesh.attrs.keys()}

    grids_meta = {mkey: [] for mkey in mergedmesh.grids.meta.keys()}
    lines_meta = {mkey: [] for mkey in mergedmesh.lines.meta.keys()}
    trias_meta = {mkey: [] for mkey in mergedmesh.trias.meta.keys()}
    quads_meta = {mkey: [] for mkey in mergedmesh.quads.meta.keys()}
    attrs_meta = {akey: {mkey: [] for mkey in attr.meta.keys()} for akey, attr in mergedmesh.attrs.items()}

    for mesh in meshes:
        if mesh.grids.size > 0:
            grids.append(mesh.grids.vecs)
            for mkey in grids_meta:
                if mkey == 'grids':
                    grids_meta[mkey].append(mesh.grids.meta[mkey] + grids_offset)
                elif mkey == 'lines':
                    grids_meta[mkey].append(mesh.grids.meta[mkey] + lines_offset)
                elif mkey == 'trias':
                    grids_meta[mkey].append(mesh.grids.meta[mkey] + trias_offset)
                elif mkey == 'quads':
                    grids_meta[mkey].append(mesh.grids.meta[mkey] + quads_offset)
                elif mkey in mesh.attrs:
                    grids_meta[mkey].append(mesh.grids.meta[mkey] + attrs_offset[mkey])
                else:
                    grids_meta[mkey].append(mesh.grids.meta[mkey])
        if mesh.lines.size > 0:
            lines.append(mesh.lines.grids + grids_offset)
            for mkey in lines_meta:
                if mkey == 'grids':
                    lines_meta[mkey].append(mesh.lines.meta[mkey] + grids_offset)
                elif mkey == 'lines':
                    lines_meta[mkey].append(mesh.lines.meta[mkey] + lines_offset)
                elif mkey == 'trias':
                    lines_meta[mkey].append(mesh.lines.meta[mkey] + trias_offset)
                elif mkey == 'quads':
                    lines_meta[mkey].append(mesh.lines.meta[mkey] + quads_offset)
                elif mkey in mesh.attrs:
                    lines_meta[mkey].append(mesh.lines.meta[mkey] + attrs_offset[mkey])
                else:
                    lines_meta[mkey].append(mesh.lines.meta[mkey])
        if mesh.trias.size > 0:
            trias.append(mesh.trias.grids + grids_offset)
            for mkey in trias_meta:
                if mkey == 'grids':
                    trias_meta[mkey].append(mesh.trias.meta[mkey] + grids_offset)
                elif mkey == 'lines':
                    trias_meta[mkey].append(mesh.trias.meta[mkey] + lines_offset)
                elif mkey == 'trias':
                    trias_meta[mkey].append(mesh.trias.meta[mkey] + trias_offset)
                elif mkey == 'quads':
                    trias_meta[mkey].append(mesh.trias.meta[mkey] + quads_offset)
                elif mkey in mesh.attrs:
                    trias_meta[mkey].append(mesh.trias.meta[mkey] + attrs_offset[mkey])
                else:
                    trias_meta[mkey].append(mesh.trias.meta[mkey])
        if mesh.quads.size > 0:
            quads.append(mesh.quads.grids + grids_offset)
            for mkey in quads_meta:
                if mkey == 'grids':
                    quads_meta[mkey].append(mesh.quads.meta[mkey] + grids_offset)
                elif mkey == 'lines':
                    quads_meta[mkey].append(mesh.quads.meta[mkey] + lines_offset)
                elif mkey == 'trias':
                    quads_meta[mkey].append(mesh.quads.meta[mkey] + trias_offset)
                elif mkey == 'quads':
                    quads_meta[mkey].append(mesh.quads.meta[mkey] + quads_offset)
                elif mkey in mesh.attrs:
                    quads_meta[mkey].append(mesh.quads.meta[mkey] + attrs_offset[mkey])
                else:
                    quads_meta[mkey].append(mesh.quads.meta[mkey])
        for akey in mergedmesh.attrs.keys():
            if mesh.attrs[akey].size > 0:
                attrs[akey].append(mesh.attrs[akey].vecs)
                for mkey in attrs_meta[akey]:
                    if mkey == 'grids':
                        attrs_meta[akey][mkey].append(mesh.attrs[akey].meta[mkey] + grids_offset)
                    elif mkey == 'lines':
                        attrs_meta[akey][mkey].append(mesh.attrs[akey].meta[mkey] + lines_offset)
                    elif mkey == 'trias':
                        attrs_meta[akey][mkey].append(mesh.attrs[akey].meta[mkey] + trias_offset)
                    elif mkey == 'quads':
                        attrs_meta[akey][mkey].append(mesh.attrs[akey].meta[mkey] + quads_offset)
                    elif mkey in mesh.attrs[akey].meta:
                        attrs_meta[akey][mkey].append(mesh.attrs[akey].meta[mkey] + attrs_offset[akey])
                    else:
                        attrs_meta[akey][mkey].append(mesh.attrs[akey].meta[mkey])
        grids_offset += mesh.grids.size
        lines_offset += mesh.lines.size
        trias_offset += mesh.trias.size
        quads_offset += mesh.quads.size
        for akey in mergedmesh.attrs.keys():
            attrs_offset[akey] += mesh.attrs[akey].size

    if len(grids) > 0:
        mergedmesh.grids.vecs = vstack(tuple(grids))
        for mkey in grids_meta:
            mergedmesh.grids.meta[mkey] = vstack(tuple(grids_meta[mkey]))
    if len(lines) > 0:
        mergedmesh.lines.grids = vstack(tuple(lines))
        for mkey in lines_meta:
            mergedmesh.lines.meta[mkey] = vstack(tuple(lines_meta[mkey]))
    if len(trias) > 0:
        mergedmesh.trias.grids = vstack(tuple(trias))
        for mkey in trias_meta:
            mergedmesh.trias.meta[mkey] = vstack(tuple(trias_meta[mkey]))
    if len(quads) > 0:
        mergedmesh.quads.grids = vstack(tuple(quads))
        for mkey in quads_meta:
            mergedmesh.quads.meta[mkey] = vstack(tuple(quads_meta[mkey]))
    for akey in mergedmesh.attrs.keys():
        if len(attrs[akey]) > 0:
            mergedmesh.attrs[akey].vecs = vstack(tuple(attrs[akey]))
            for mkey in attrs_meta[akey]:
                mergedmesh.attrs[akey].meta[mkey] = vstack(tuple(attrs_meta[akey][mkey]))

    return mergedmesh
