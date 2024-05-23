from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from numpy import (arange, argsort, array, bool_, float64, hstack, int64,
                   logical_and, ndarray, take_along_axis, unique, vstack,
                   zeros)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class MetaCache():
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


class MeshGrids():
    grids: 'NDArray[float64]' = None
    meta: Dict[str, 'NDArray'] = None
    grids_cache: List[Tuple[float, float, float]] = None
    meta_cache: Dict[str, MetaCache] = None

    def __init__(self) -> None:
        self.grids = zeros((0, 3), dtype=float64)
        self.meta = {}
        self.grids_cache = []
        self.meta_cache = {}

    def add_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.meta_cache[key] = MetaCache(key, dtype, default)

    def add(self, x: float, y: float, z: float, **kwargs: Dict[str, Any]) -> None:
        self.grids_cache.append((x, y, z))
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

    def __getitem__(self, index: Any) -> 'MeshGrids':
        meshgrids = MeshGrids()
        meshgrids.grids = self.grids[index]
        meshgrids.meta = {}
        for key in self.meta.keys():
            meshgrids.meta[key] = self.meta[key][index]
        return meshgrids

    def __setitem__(self, index: Any, value: 'MeshGrids') -> None:
        try:
            self.grids[index] = value.grids
            for key in self.meta.keys():
                self.meta[key][index] = value.meta[key]
        except IndexError:
            err = 'MeshGrids index out of range.'
            raise IndexError(err)

    @property
    def size(self) -> int:
        return self.grids.shape[0]

    def __str__(self) -> str:
        outstr = f'MeshGrids: size = {self.size:d}, dtype = {self.grids.dtype}\n'
        outstr += f'grids: \n{self.grids:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

    def __repr__(self) -> str:
        return f'<MeshGrids: size = {self.size:d}>'


class MeshNorms():
    norms: 'NDArray[float64]' = None
    meta: Dict[str, 'NDArray'] = None
    norms_cache: List[Tuple[float, float, float]] = None
    meta_cache: Dict[str, MetaCache] = None

    def __init__(self) -> None:
        self.norms = zeros((0, 3), dtype=float64)
        self.meta = {}
        self.norms_cache = []
        self.meta_cache = {}

    def add_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.meta_cache[key] = MetaCache(key, dtype, default)

    def add(self, x: float, y: float, z: float, **kwargs: Dict[str, Any]) -> None:
        self.norms_cache.append((x, y, z))
        for key in self.meta_cache.keys():
            value = kwargs.get(key, self.meta_cache[key].default)
            self.meta_cache[key].append(value)

    def clear_cache(self) -> None:
        self.norms_cache.clear()
        for value in self.meta_cache.values():
            value.clear()

    def resolve_cache(self) -> None:
        self.norms = array(self.norms_cache, dtype=float64)
        for key, value in self.meta_cache.items():
            self.meta[key] = value.asarray()
        self.clear_cache()

    def duplicate_indices(self) -> Tuple['NDArray[int64]',
                                         'NDArray[int64]']:
        if self.size == 0:
            return zeros(0, dtype=int64), zeros(0, dtype=int64)
        data = self.norms
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
        if key == 'norms':
            raise ValueError('Cannot apply inverse to norms.')
        if key in self.meta:
            self.meta[key] = invind[self.meta[key]]

    def __getitem__(self, index: Any) -> 'MeshNorms':
        meshnorms = MeshNorms()
        meshnorms.norms = self.norms[index]
        meshnorms.meta = {}
        for key in self.meta.keys():
            meshnorms.meta[key] = self.meta[key][index]
        return meshnorms

    def __setitem__(self, index: Any, value: 'MeshNorms') -> None:
        try:
            self.norms[index] = value.norms
            for key in self.meta.keys():
                self.meta[key][index] = value.meta[key]
        except IndexError:
            err = 'MeshNorms index out of range.'
            raise IndexError(err)

    @property
    def size(self) -> int:
        return self.norms.shape[0]

    def __str__(self) -> str:
        outstr = f'MeshNorms: size = {self.size:d}, dtype = {self.norms.dtype}\n'
        outstr += f'norms: \n{self.norms:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

    def __repr__(self) -> str:
        return f'<MeshNorms: size = {self.size:d}>'


class MeshElems():
    name: str = 'MeshElems'
    desc: str = 'elems'
    numg: int = 0
    grids: 'NDArray[int64]' = None
    meta: Dict[str, 'NDArray'] = None
    grids_cache: List[Tuple[int, ...]] = None
    meta_cache: Dict[str, MetaCache] = None

    def __init__(self) -> None:
        self.grids = zeros((0, self.numg), dtype=int64)
        self.meta = {}
        self.grids_cache = []
        self.meta_cache = {}

    def add_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.meta_cache[key] = MetaCache(key, dtype, default)

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
                if isinstance(value, ndarray):
                    if value.shape[1] == 2:
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
        print(f'self.grids: \n{self.grids}')
        inda = arange(-1, self.numg-2)
        print(f'inda: {inda}')
        grids1 = self.grids[:, inda]
        print(f'grids1: \n{grids1}')
        indb = arange(0, self.numg-1)
        print(f'indb: {indb}')
        grids2 = self.grids[:, indb]
        print(f'grids2: \n{grids2}')
        checkedge: 'NDArray[bool_]' = grids1 != grids2
        print(f'checkedge: \n{checkedge}')
        countelem: 'NDArray[int64]' = checkedge.sum(axis=1)
        count = countelem.flatten()
        print(f'count: \n{count}')
        check = count > 0
        print(f'check: \n{check}')
        # meshelems = self.__class__()
        self.grids = self.grids[check, :]
        # meshelems.meta = {}
        for key in self.meta.keys():
            self.meta[key] = self.meta[key][check, :]
        # print(f'meshelems: \n{meshelems}')
        # self = meshelems

    def __getitem__(self, index: Any) -> 'MeshElems':
        meshelems = self.__class__()
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

    def __str__(self) -> str:
        outstr = f'{self.name:s}: size = {self.size:d}\n'
        outstr += f'grids: \n{self.grids:}\n'
        for key, value in self.meta.items():
            outstr += f'{key}: \n{value:}\n'
        return outstr

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
    grids: MeshGrids = None
    norms: MeshNorms = None
    lines: MeshLines = None
    trias: MeshTrias = None
    quads: MeshQuads = None

    def __init__(self) -> None:
        self.grids = MeshGrids()
        self.norms = MeshNorms()
        self.lines = MeshLines()
        self.trias = MeshTrias()
        self.quads = MeshQuads()

    def resolve_cache(self) -> None:
        self.grids.resolve_cache()
        self.norms.resolve_cache()
        self.lines.resolve_cache()
        self.trias.resolve_cache()
        self.quads.resolve_cache()

    def remove_duplicate_grids(self) -> None:
        if self.grids.size == 0:
            return None
        unind, invind = self.grids.duplicate_indices()
        self.grids = self.grids[unind]
        self.norms.apply_inverse(invind, 'grids')
        self.lines.apply_inverse(invind, 'grids')
        self.trias.apply_inverse(invind, 'grids')
        self.quads.apply_inverse(invind, 'grids')

    def remove_duplicate_norms(self) -> None:
        if self.norms.size == 0:
            return None
        unind, invind = self.norms.duplicate_indices()
        self.norms = self.norms[unind]
        self.grids.apply_inverse(invind, 'norms')
        self.lines.apply_inverse(invind, 'norms')
        self.trias.apply_inverse(invind, 'norms')
        self.quads.apply_inverse(invind, 'norms')

    def remove_duplicate_lines(self) -> None:
        if self.lines.size == 0:
            return None
        unind, invind = self.lines.duplicate_indices()
        self.lines = self.lines[unind]
        self.grids.apply_inverse(invind, 'lines')
        self.norms.apply_inverse(invind, 'lines')
        self.trias.apply_inverse(invind, 'lines')
        self.quads.apply_inverse(invind, 'lines')

    def remove_duplicate_trias(self) -> None:
        if self.trias.size == 0:
            return None
        unind, invind = self.trias.duplicate_indices()
        self.trias = self.trias[unind]
        self.grids.apply_inverse(invind, 'trias')
        self.norms.apply_inverse(invind, 'trias')
        self.lines.apply_inverse(invind, 'trias')
        self.quads.apply_inverse(invind, 'trias')

    def remove_duplicate_quads(self) -> None:
        if self.quads.size == 0:
            return None
        unind, invind = self.quads.duplicate_indices()
        self.quads = self.quads[unind]
        self.grids.apply_inverse(invind, 'quads')
        self.norms.apply_inverse(invind, 'quads')
        self.lines.apply_inverse(invind, 'quads')
        self.trias.apply_inverse(invind, 'quads')

    # def remove_collapsed_trias(self) -> None:
    #     if self.trias.size == 0:
    #         return None
    #     check: 'NDArray[bool_]' = self.trias[:, (0, 1, 2)] != self.trias[:, (1, 2, 0)]
    #     sumax1 = check.sum(axis=1)
    #     check = sumax1 == 0
    #     self.trias = self.trias[check, ...]
    #     for key in self.triameta:
    #         self.triameta[key] = self.triameta[key][check, ...]
    #     if 'trias' in self.gridmeta:
    #         self.gridmeta['trias'] = self.gridmeta['trias'][check, ...]
    #     if 'trias' in self.normmeta:
    #         self.normmeta['trias'] = self.normmeta['trias'][check, ...]
    #     if 'trias' in self.linemeta:
    #         self.linemeta['trias'] = self.linemeta['trias'][check, ...]
    #     if 'trias' in self.quadmeta:
    #         self.quadmeta['trias'] = self.quadmeta['trias'][check, ...]

    # def remove_collapsed_quads(self) -> None:
    #     if self.quads.size == 0:
    #         return None
    #     check: 'NDArray[bool_]' = self.quads[:, (0, 1, 2, 3, 0, 1)] == self.quads[:, (1, 2, 3, 0, 2, 3)]
    #     sumax1 = check.sum(axis=1)
    #     chkeq0 = sumax1 == 0
    #     chkeq1 = sumax1 == 1

    #     # Reduce Quads to Trias
    #     checkab = logical_and(chkeq1, check[:, 0])
    #     checkbc = logical_and(chkeq1, check[:, 1])
    #     checkcd = logical_and(chkeq1, check[:, 2])
    #     checkda = logical_and(chkeq1, check[:, 3])
    #     quad_ab = self.quads[checkab, :]
    #     quad_bc = self.quads[checkbc, :]
    #     quad_cd = self.quads[checkcd, :]
    #     quad_da = self.quads[checkda, :]
    #     tria_ab = quad_ab[:, (1, 2, 3)]
    #     tria_bc = quad_bc[:, (2, 3, 0)]
    #     tria_cd = quad_cd[:, (3, 0, 1)]
    #     tria_da = quad_da[:, (0, 1, 2)]
    #     if self.trias.size == 0:
    #         self.trias = vstack((tria_ab, tria_bc, tria_cd, tria_da))
    #     else:
    #         self.trias = vstack((self.trias, tria_ab, tria_bc, tria_cd, tria_da))
    #     for key in self.quadmeta:
    #         meta = []
    #         if key in self.triameta:
    #             if self.triameta[key].size > 0:
    #                 meta.append(self.triameta[key])
    #         if self.quadmeta[key].shape[1] == 4:
    #             quad_ab_meta = self.quadmeta[key][checkab, ...]
    #             quad_bc_meta = self.quadmeta[key][checkbc, ...]
    #             quad_cd_meta = self.quadmeta[key][checkcd, ...]
    #             quad_da_meta = self.quadmeta[key][checkda, ...]
    #             tria_ab_meta = quad_ab_meta[:, (1, 2, 3)]
    #             tria_bc_meta = quad_bc_meta[:, (2, 3, 0)]
    #             tria_cd_meta = quad_cd_meta[:, (3, 0, 1)]
    #             tria_da_meta = quad_da_meta[:, (0, 1, 2)]
    #         else:
    #             tria_ab_meta = self.quadmeta[key][checkab, ...]
    #             tria_bc_meta = self.quadmeta[key][checkbc, ...]
    #             tria_cd_meta = self.quadmeta[key][checkcd, ...]
    #             tria_da_meta = self.quadmeta[key][checkda, ...]
    #         if tria_ab_meta.size > 0:
    #             meta.append(tria_ab_meta)
    #         if tria_bc_meta.size > 0:
    #             meta.append(tria_bc_meta)
    #         if tria_cd_meta.size > 0:
    #             meta.append(tria_cd_meta)
    #         if tria_da_meta.size > 0:
    #             meta.append(tria_da_meta)
    #         if len(meta) > 0:
    #             self.triameta[key] = vstack(tuple(meta))

    #     # Reduce to only valid quads
    #     self.quads = self.quads[chkeq0, ...]
    #     for key in self.quadmeta:
    #         self.quadmeta[key] = self.quadmeta[key][chkeq0, ...]
    #     if 'quads' in self.gridmeta:
    #         self.gridmeta['quads'] = self.gridmeta['quads'][chkeq0, ...]
    #     if 'quads' in self.normmeta:
    #         self.normmeta['quads'] = self.normmeta['quads'][chkeq0, ...]
    #     if 'quads' in self.linemeta:
    #         self.linemeta['quads'] = self.linemeta['quads'][chkeq0, ...]
    #     if 'quads' in self.triameta:
    #         self.triameta['quads'] = self.triameta['quads'][chkeq0, ...]

    def remove_unreferenced_grids(self) -> None:
        if self.grids.size == 0:
            return None
        refind = []
        if 'grids' in self.norms.meta:
            refind.append(self.norms.meta['grids'].flatten())
        if 'grids' in self.lines.meta:
            refind.append(self.lines.meta['grids'].flatten())
        if 'grids' in self.trias.meta:
            refind.append(self.trias.meta['grids'].flatten())
        if 'grids' in self.quads.meta:
            refind.append(self.quads.meta['grids'].flatten())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        invind = zeros(self.grids.size, dtype=int64)
        invind[refind] = arange(refind.size)
        self.grids = self.grids[refind]
        self.lines.apply_inverse(invind, 'grids')
        self.trias.apply_inverse(invind, 'grids')
        self.quads.apply_inverse(invind, 'grids')
        self.norms.apply_inverse(invind, 'grids')

    def remove_unreferenced_norms(self) -> None:
        if self.norms.size == 0:
            return None
        refind = []
        if 'norms' in self.grids.meta:
            refind.append(self.grids.meta['norms'].flatten())
        if 'norms' in self.lines.meta:
            refind.append(self.lines.meta['norms'].flatten())
        if 'norms' in self.trias.meta:
            refind.append(self.trias.meta['norms'].flatten())
        if 'norms' in self.quads.meta:
            refind.append(self.quads.meta['norms'].flatten())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        invind = zeros(self.norms.size, dtype=int64)
        invind[refind] = arange(refind.size)
        self.norms = self.norms[refind]
        self.grids.apply_inverse(invind, 'norms')
        self.lines.apply_inverse(invind, 'norms')
        self.trias.apply_inverse(invind, 'norms')
        self.quads.apply_inverse(invind, 'norms')

    def merge(self, mesh: 'Mesh') -> 'Mesh':

        mergedmesh = Mesh()

        # Grids
        mergedmesh.grids.grids = vstack((self.grids.grids, mesh.grids.grids))

        # Grid Meta
        mergedmesh.grids.meta = {}
        for key, value in self.grids.meta.items():
            meshvalue = mesh.grids.meta[key]
            if key == 'norms':
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + self.norms.size))
            elif key == 'lines':
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + self.lines.size))
            elif key == 'trias':
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + self.trias.size))
            elif key == 'quads':
                mergedmesh.grids.meta[key] = vstack((value, meshvalue + self.quads.size))
            else:
                mergedmesh.grids.meta[key] = vstack((value, meshvalue))

        # Norms
        mergedmesh.norms.norms = vstack((self.norms.norms, mesh.norms.norms))

        # Norm Meta
        mergedmesh.norms.meta = {}
        for key, value in self.norms.meta.items():
            meshvalue = mesh.norms.meta[key]
            if key == 'grids':
                mergedmesh.norms.meta[key] = vstack((value, meshvalue + self.grids.size))
            elif key == 'lines':
                mergedmesh.norms.meta[key] = vstack((value, meshvalue + self.lines.size))
            elif key == 'trias':
                mergedmesh.norms.meta[key] = vstack((value, meshvalue + self.trias.size))
            elif key == 'quads':
                mergedmesh.norms.meta[key] = vstack((value, meshvalue + self.quads.size))
            else:
                mergedmesh.norms.meta[key] = vstack((value, meshvalue))

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
            elif key == 'norms':
                mergedmesh.lines.meta[key] = vstack((value, meshvalue + self.norms.size))
            elif key == 'trias':
                mergedmesh.lines.meta[key] = vstack((value, meshvalue + self.trias.size))
            elif key == 'quads':
                mergedmesh.lines.meta[key] = vstack((value, meshvalue + self.quads.size))
            else:
                mergedmesh.lines.meta[key] = vstack((value, meshvalue))

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
            elif key == 'norms':
                mergedmesh.trias.meta[key] = vstack((value, meshvalue + self.norms.size))
            elif key == 'lines':
                mergedmesh.trias.meta[key] = vstack((value, meshvalue + self.lines.size))
            elif key == 'quads':
                mergedmesh.trias.meta[key] = vstack((value, meshvalue + self.quads.size))
            else:
                mergedmesh.trias.meta[key] = vstack((value, meshvalue))

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
            elif key == 'norms':
                mergedmesh.quads.meta[key] = vstack((value, meshvalue + self.norms.size))
            elif key == 'lines':
                mergedmesh.quads.meta[key] = vstack((value, meshvalue + self.lines.size))
            elif key == 'trias':
                mergedmesh.quads.meta[key] = vstack((value, meshvalue + self.trias.size))
            else:
                mergedmesh.quads.meta[key] = vstack((value, meshvalue))

        return mergedmesh

    def __str__(self) -> str:
        outstr = ''
        if self.grids is not None:
            outstr += f'{self.grids}\n'
        if self.norms is not None:
            outstr += f'{self.norms}\n'
        if self.lines is not None:
            outstr += f'{self.lines}\n'
        if self.trias is not None:
            outstr += f'{self.trias}\n'
        if self.quads is not None:
            outstr += f'{self.quads}\n'
        outstr += '\n'
        return outstr

    def __repr__(self) -> str:
        return '<Mesh>'
