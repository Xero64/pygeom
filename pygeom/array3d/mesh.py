from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from numpy import (arange, argsort, array, bool_, float64, hstack, int64,
                   logical_and, take_along_axis, unique, vstack, zeros)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class Mesh():
    grids: 'NDArray[float64]' = None
    norms: 'NDArray[float64]' = None
    lines: 'NDArray[int64]' = None
    trias: 'NDArray[int64]' = None
    quads: 'NDArray[int64]' = None
    gridmeta: Dict[str, 'NDArray[int64]'] = None
    normmeta: Dict[str, 'NDArray[int64]'] = None
    linemeta: Dict[str, 'NDArray[int64]'] = None
    triameta: Dict[str, 'NDArray[int64]'] = None
    quadmeta: Dict[str, 'NDArray[int64]'] = None

    def __init__(self) -> None:
        self.grids = zeros((0, 3), dtype=float64)
        self.norms = zeros((0, 3), dtype=float64)
        self.lines = zeros((0, 2), dtype=int64)
        self.trias = zeros((0, 3), dtype=int64)
        self.quads = zeros((0, 4), dtype=int64)
        self.gridmeta = {}
        self.normmeta = {}
        self.linemeta = {}
        self.triameta = {}
        self.quadmeta = {}

    @property
    def num_grids(self) -> int:
        return self.grids.shape[0]

    @property
    def num_norms(self) -> int:
        return self.norms.shape[0]

    @property
    def num_lines(self) -> int:
        return self.lines.shape[0]

    @property
    def num_trias(self) -> int:
        return self.trias.shape[0]

    @property
    def num_quads(self) -> int:
        return self.quads.shape[0]

    def remove_duplicate_grids(self) -> None:
        if self.grids.size == 0:
            return None
        data = self.grids
        if len(self.gridmeta) > 0:
            data = [data]
            for val in self.gridmeta.values():
                data.append(val.reshape(-1, 1))
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        self.grids = self.grids[unind, ...]
        for key in self.gridmeta.keys():
            self.gridmeta[key] = self.gridmeta[key][unind, ...]
        self.lines = invind[self.lines]
        self.trias = invind[self.trias]
        self.quads = invind[self.quads]
        if 'grids' in self.normmeta:
            self.normmeta['grids'] = invind[self.normmeta['grids']]
        if 'grids' in self.linemeta:
            self.linemeta['grids'] = invind[self.linemeta['grids']]
        if 'grids' in self.triameta:
            self.triameta['grids'] = invind[self.triameta['grids']]
        if 'grids' in self.quadmeta:
            self.quadmeta['grids'] = invind[self.quadmeta['grids']]

    def remove_duplicate_norms(self) -> None:
        if self.norms.size == 0:
            return None
        data = self.norms
        if len(self.normmeta) > 0:
            data = [data]
            for val in self.normmeta.values():
                data.append(val.reshape(-1, 1))
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        self.norms = self.norms[unind, ...]
        for key in self.normmeta:
            self.normmeta[key] = self.normmeta[key][unind, ...]
        if 'norms' in self.gridmeta:
            self.gridmeta['norms'] = invind[self.gridmeta['norms']]
        if 'norms' in self.linemeta:
            self.linemeta['norms'] = invind[self.linemeta['norms']]
        if 'norms' in self.triameta:
            self.triameta['norms'] = invind[self.triameta['norms']]
        if 'norms' in self.quadmeta:
            self.quadmeta['norms'] = invind[self.quadmeta['norms']]

    def remove_duplicate_lines(self) -> None:
        if self.lines.size == 0:
            return None
        minindax1 = argsort(self.lines, axis=1)
        data = take_along_axis(self.lines, minindax1, axis=1)
        if len(self.linemeta) > 0:
            data = [data]
            for value in self.linemeta.values():
                if value.shape[1] == 2:
                    data.append(take_along_axis(value, minindax1, axis=1))
                else:
                    data.append(value)
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        self.lines = self.lines[unind, ...]
        for key in self.linemeta:
            self.linemeta[key] = self.linemeta[key][unind, ...]
        if 'lines' in self.gridmeta:
            self.gridmeta['lines'] = invind[self.gridmeta['lines']]
        if 'lines' in self.normmeta:
            self.normmeta['lines'] = invind[self.normmeta['lines']]
        if 'lines' in self.triameta:
            self.triameta['lines'] = invind[self.triameta['lines']]
        if 'lines' in self.quadmeta:
            self.quadmeta['lines'] = invind[self.quadmeta['lines']]

    def remove_duplicate_trias(self) -> None:
        if self.trias.size == 0:
            return None
        minindax1 = argsort(self.trias, axis=1)
        data = take_along_axis(self.trias, minindax1, axis=1)
        if len(self.triameta) > 0:
            data = [data]
            for value in self.triameta.values():
                if value.shape[1] == 3:
                    data.append(take_along_axis(value, minindax1, axis=1))
                else:
                    data.append(value)
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        self.trias = self.trias[unind, ...]
        for key in self.triameta:
            self.triameta[key] = self.triameta[key][unind, ...]
        if 'trias' in self.gridmeta:
            self.gridmeta['trias'] = invind[self.gridmeta['trias']]
        if 'trias' in self.normmeta:
            self.normmeta['trias'] = invind[self.normmeta['trias']]
        if 'trias' in self.linemeta:
            self.linemeta['trias'] = invind[self.linemeta['trias']]
        if 'trias' in self.quadmeta:
            self.quadmeta['trias'] = invind[self.quadmeta['trias']]

    def remove_duplicate_quads(self) -> None:
        if self.quads.size == 0:
            return None
        minindax1 = argsort(self.quads, axis=1)
        data = take_along_axis(self.quads, minindax1, axis=1)
        if len(self.quadmeta) > 0:
            data = [data]
            for value in self.quadmeta.values():
                if value.shape[1] == 4:
                    data.append(take_along_axis(value, minindax1, axis=1))
                else:
                    data.append(value)
            data = hstack(tuple(data))
        _, unind, invind = unique(data, return_index=True,
                                  return_inverse=True, axis=0)
        self.quads = self.quads[unind, ...]
        for key in self.quadmeta:
            self.quadmeta[key] = self.quadmeta[key][unind, ...]
        if 'quads' in self.gridmeta:
            self.gridmeta['quads'] = invind[self.gridmeta['quads']]
        if 'quads' in self.normmeta:
            self.normmeta['quads'] = invind[self.normmeta['quads']]
        if 'quads' in self.linemeta:
            self.linemeta['quads'] = invind[self.linemeta['quads']]
        if 'quads' in self.triameta:
            self.triameta['quads'] = invind[self.triameta['quads']]

    def remove_collapsed_lines(self) -> None:
        if self.lines.size == 0:
            return None
        check =  self.lines[:, 0] != self.lines[:, 1]
        self.lines = self.lines[check, ...]
        for key in self.linemeta:
            self.linemeta[key] = self.linemeta[key][check, ...]
        if 'lines' in self.gridmeta:
            self.gridmeta['lines'] = self.gridmeta['lines'][check, ...]
        if 'lines' in self.normmeta:
            self.normmeta['lines'] = self.normmeta['lines'][check, ...]
        if 'lines' in self.triameta:
            self.triameta['lines'] = self.triameta['lines'][check, ...]
        if 'lines' in self.quadmeta:
            self.quadmeta['lines'] = self.quadmeta['lines'][check, ...]

    def remove_collapsed_trias(self) -> None:
        if self.trias.size == 0:
            return None
        check: 'NDArray[bool_]' = self.trias[:, (0, 1, 2)] != self.trias[:, (1, 2, 0)]
        sumax1 = check.sum(axis=1)
        check = sumax1 == 0
        self.trias = self.trias[check, ...]
        for key in self.triameta:
            self.triameta[key] = self.triameta[key][check, ...]
        if 'trias' in self.gridmeta:
            self.gridmeta['trias'] = self.gridmeta['trias'][check, ...]
        if 'trias' in self.normmeta:
            self.normmeta['trias'] = self.normmeta['trias'][check, ...]
        if 'trias' in self.linemeta:
            self.linemeta['trias'] = self.linemeta['trias'][check, ...]
        if 'trias' in self.quadmeta:
            self.quadmeta['trias'] = self.quadmeta['trias'][check, ...]

    def remove_collapsed_quads(self) -> None:
        if self.quads.size == 0:
            return None
        check: 'NDArray[bool_]' = self.quads[:, (0, 1, 2, 3, 0, 1)] == self.quads[:, (1, 2, 3, 0, 2, 3)]
        sumax1 = check.sum(axis=1)
        chkeq0 = sumax1 == 0
        chkeq1 = sumax1 == 1

        # Reduce Quads to Trias
        checkab = logical_and(chkeq1, check[:, 0])
        checkbc = logical_and(chkeq1, check[:, 1])
        checkcd = logical_and(chkeq1, check[:, 2])
        checkda = logical_and(chkeq1, check[:, 3])
        quad_ab = self.quads[checkab, :]
        quad_bc = self.quads[checkbc, :]
        quad_cd = self.quads[checkcd, :]
        quad_da = self.quads[checkda, :]
        tria_ab = quad_ab[:, (1, 2, 3)]
        tria_bc = quad_bc[:, (2, 3, 0)]
        tria_cd = quad_cd[:, (3, 0, 1)]
        tria_da = quad_da[:, (0, 1, 2)]
        if self.trias.size == 0:
            self.trias = vstack((tria_ab, tria_bc, tria_cd, tria_da))
        else:
            self.trias = vstack((self.trias, tria_ab, tria_bc, tria_cd, tria_da))
        for key in self.quadmeta:
            meta = []
            if key in self.triameta:
                if self.triameta[key].size > 0:
                    meta.append(self.triameta[key])
            if self.quadmeta[key].shape[1] == 4:
                quad_ab_meta = self.quadmeta[key][checkab, ...]
                quad_bc_meta = self.quadmeta[key][checkbc, ...]
                quad_cd_meta = self.quadmeta[key][checkcd, ...]
                quad_da_meta = self.quadmeta[key][checkda, ...]
                tria_ab_meta = quad_ab_meta[:, (1, 2, 3)]
                tria_bc_meta = quad_bc_meta[:, (2, 3, 0)]
                tria_cd_meta = quad_cd_meta[:, (3, 0, 1)]
                tria_da_meta = quad_da_meta[:, (0, 1, 2)]
            else:
                tria_ab_meta = self.quadmeta[key][checkab, ...]
                tria_bc_meta = self.quadmeta[key][checkbc, ...]
                tria_cd_meta = self.quadmeta[key][checkcd, ...]
                tria_da_meta = self.quadmeta[key][checkda, ...]
            if tria_ab_meta.size > 0:
                meta.append(tria_ab_meta)
            if tria_bc_meta.size > 0:
                meta.append(tria_bc_meta)
            if tria_cd_meta.size > 0:
                meta.append(tria_cd_meta)
            if tria_da_meta.size > 0:
                meta.append(tria_da_meta)
            if len(meta) > 0:
                self.triameta[key] = vstack(tuple(meta))

        # Reduce to only valid quads
        self.quads = self.quads[chkeq0, ...]
        for key in self.quadmeta:
            self.quadmeta[key] = self.quadmeta[key][chkeq0, ...]
        if 'quads' in self.gridmeta:
            self.gridmeta['quads'] = self.gridmeta['quads'][chkeq0, ...]
        if 'quads' in self.normmeta:
            self.normmeta['quads'] = self.normmeta['quads'][chkeq0, ...]
        if 'quads' in self.linemeta:
            self.linemeta['quads'] = self.linemeta['quads'][chkeq0, ...]
        if 'quads' in self.triameta:
            self.triameta['quads'] = self.triameta['quads'][chkeq0, ...]

    def remove_unreferenced_grids(self) -> None:
        if self.grids.size == 0:
            return None
        refind = []
        if 'grids' in self.normmeta:
            refind.append(self.normmeta['grids'].flatten())
        if 'grids' in self.linemeta:
            refind.append(self.linemeta['grids'].flatten())
        if 'grids' in self.triameta:
            refind.append(self.triameta['grids'].flatten())
        if 'grids' in self.quadmeta:
            refind.append(self.quadmeta['grids'].flatten())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        revind = zeros(self.grids.shape[0], dtype=int64)
        revind[refind] = arange(refind.size)
        self.grids = self.grids[refind, ...]
        self.lines = revind[self.lines]
        self.trias = revind[self.trias]
        self.quads = revind[self.quads]
        for key in self.gridmeta.keys():
            self.gridmeta[key] = self.gridmeta[key][refind, ...]
        if 'grids' in self.normmeta:
            self.normmeta['grids'] = revind[self.normmeta['grids']]
        if 'grids' in self.linemeta:
            self.linemeta['grids'] = revind[self.linemeta['grids']]
        if 'grids' in self.triameta:
            self.triameta['grids'] = revind[self.triameta['grids']]
        if 'grids' in self.quadmeta:
            self.quadmeta['grids'] = revind[self.quadmeta['grids']]

    def remove_unreferenced_norms(self) -> None:
        if self.norms.size == 0:
            return None
        refind = []
        if 'norms' in self.gridmeta:
            refind.append(self.gridmeta['norms'].flatten())
        if 'norms' in self.linemeta:
            refind.append(self.linemeta['norms'].flatten())
        if 'norms' in self.triameta:
            refind.append(self.triameta['norms'].flatten())
        if 'norms' in self.quadmeta:
            refind.append(self.quadmeta['norms'].flatten())
        refind = hstack(tuple(refind))
        refind = unique(refind)
        revind = zeros(self.norms.shape[0], dtype=int64)
        revind[refind] = arange(refind.size)
        self.norms = self.norms[refind, ...]
        for key in self.normmeta:
            self.normmeta[key] = self.normmeta[key][refind, ...]
        if 'norms' in self.gridmeta:
            self.gridmeta['norms'] = revind[self.gridmeta['norms']]
        if 'norms' in self.linemeta:
            self.linemeta['norms'] = revind[self.linemeta['norms']]
        if 'norms' in self.triameta:
            self.triameta['norms'] = revind[self.triameta['norms']]

    def merge(self, mesh: 'Mesh') -> 'Mesh':

        mergedmesh = Mesh()

        # Grids
        if self.grids.shape[0] != 0 and mesh.grids.shape[0] != 0:
            mergedmesh.grids = vstack((self.grids, mesh.grids))
        elif self.grids.shape[0] == 0:
            mergedmesh.grids = mesh.grids
        elif mesh.grids.shape[0] == 0:
            mergedmesh.grids = self.grids
        else:
            mergedmesh.grids = zeros((0, 3), dtype=float64)

        # Grid Meta
        mergedmesh.gridmeta = {}
        for key, val in self.gridmeta.items():
            if key == 'norms':
                mergedmesh.gridmeta[key] = vstack((val, mesh.gridmeta[key] + self.num_norms))
            elif key == 'lines':
                mergedmesh.gridmeta[key] = vstack((val, mesh.gridmeta[key] + self.num_lines))
            elif key == 'trias':
                mergedmesh.gridmeta[key] = vstack((val, mesh.gridmeta[key] + self.num_trias))
            elif key == 'quads':
                mergedmesh.gridmeta[key] = vstack((val, mesh.gridmeta[key] + self.num_quads))
            else:
                mergedmesh.gridmeta[key] = vstack((val, mesh.gridmeta[key]))

        # Norms
        if self.norms.shape[0] != 0 and mesh.norms.shape[0] != 0:
            mergedmesh.norms = vstack((self.norms, mesh.norms))
        elif self.norms.shape[0] == 0:
            mergedmesh.norms = mesh.norms
        elif mesh.norms.shape[0] == 0:
            mergedmesh.norms = self.norms
        else:
            mergedmesh.norms = zeros((0, 3), dtype=float64)

        # Norm Meta
        mergedmesh.normmeta = {}
        for key, val in self.normmeta.items():
            if key == 'grids':
                mergedmesh.normmeta[key] = vstack((val, mesh.normmeta[key] + self.num_grids))
            elif key == 'lines':
                mergedmesh.normmeta[key] = vstack((val, mesh.normmeta[key] + self.num_lines))
            elif key == 'trias':
                mergedmesh.normmeta[key] = vstack((val, mesh.normmeta[key] + self.num_trias))
            elif key == 'quads':
                mergedmesh.normmeta[key] = vstack((val, mesh.normmeta[key] + self.num_quads))
            else:
                mergedmesh.normmeta[key] = vstack((val, mesh.normmeta[key]))

        # Lines
        if self.lines.shape[0] != 0 and mesh.lines.shape[0] != 0:
            mergedmesh.lines = vstack((self.lines, mesh.lines + self.num_grids))
        elif self.lines.shape[0] == 0:
            mergedmesh.lines = mesh.lines + self.num_grids
        elif mesh.lines.shape[0] == 0:
            mergedmesh.lines = self.lines
        else:
            mergedmesh.lines = zeros((0, 2), dtype=int64)

        # Line Meta
        mergedmesh.linemeta = {}
        for key, val in self.linemeta.items():
            if key == 'grids':
                mergedmesh.linemeta[key] = vstack((val, mesh.linemeta[key] + self.num_grids))
            elif key == 'norms':
                mergedmesh.linemeta[key] = vstack((val, mesh.linemeta[key] + self.num_norms))
            elif key == 'trias':
                mergedmesh.linemeta[key] = vstack((val, mesh.linemeta[key] + self.num_trias))
            elif key == 'quads':
                mergedmesh.linemeta[key] = vstack((val, mesh.linemeta[key] + self.num_quads))
            else:
                mergedmesh.linemeta[key] = vstack((val, mesh.linemeta[key]))

        # Trias
        if self.trias.shape[0] != 0 and mesh.trias.shape[0] != 0:
            mergedmesh.trias = vstack((self.trias, mesh.trias + self.num_grids))
        elif self.trias.shape[0] == 0:
            mergedmesh.trias = mesh.trias + self.num_grids
        elif mesh.trias.shape[0] == 0:
            mergedmesh.trias = self.trias
        else:
            mergedmesh.trias = zeros((0, 3), dtype=int64)

        # Tria Meta
        mergedmesh.triameta = {}
        for key, val in self.triameta.items():
            if key == 'grids':
                mergedmesh.triameta[key] = vstack((val, mesh.triameta[key] + self.num_grids))
            elif key == 'norms':
                mergedmesh.triameta[key] = vstack((val, mesh.triameta[key] + self.num_norms))
            elif key == 'lines':
                mergedmesh.triameta[key] = vstack((val, mesh.triameta[key] + self.num_lines))
            elif key == 'quads':
                mergedmesh.triameta[key] = vstack((val, mesh.triameta[key] + self.num_quads))
            else:
                mergedmesh.triameta[key] = vstack((val, mesh.triameta[key]))

        # Quads
        if self.quads.shape[0] != 0 and mesh.quads.shape[0] != 0:
            mergedmesh.quads = vstack((self.quads, mesh.quads + self.num_grids))
        elif self.quads.shape[0] == 0:
            mergedmesh.quads = mesh.quads + self.num_grids
        elif mesh.quads.shape[0] == 0:
            mergedmesh.quads = self.quads
        else:
            mergedmesh.quads = zeros((0, 4), dtype=int64)

        # Quad Meta
        mergedmesh.quadmeta = {}
        for key, val in self.quadmeta.items():
            if key == 'grids':
                mergedmesh.quadmeta[key] = vstack((val, mesh.quadmeta[key] + self.num_grids))
            elif key == 'norms':
                mergedmesh.quadmeta[key] = vstack((val, mesh.quadmeta[key] + self.num_norms))
            elif key == 'lines':
                mergedmesh.quadmeta[key] = vstack((val, mesh.quadmeta[key] + self.num_lines))
            elif key == 'trias':
                mergedmesh.quadmeta[key] = vstack((val, mesh.quadmeta[key] + self.num_trias))
            else:
                mergedmesh.quadmeta[key] = vstack((val, mesh.quadmeta[key]))

        return mergedmesh

    def __str__(self) -> str:
        outstr = ''
        if self.grids is not None:
            outstr += f'grids = \n{self.grids}\n'
            for key, val in self.gridmeta.items():
                outstr += f'grids.{key} = \n{val}\n'
            outstr += '\n'
        if self.norms is not None:
            outstr += f'norms = \n{self.norms}\n'
            for key, val in self.normmeta.items():
                outstr += f'norms.{key} = \n{val}\n'
            outstr += '\n'
        if self.lines is not None:
            outstr += f'lines = \n{self.lines}\n'
            for key, val in self.linemeta.items():
                outstr += f'lines.{key} = \n{val}\n'
            outstr += '\n'
        if self.trias is not None:
            outstr += f'trias = \n{self.trias}\n'
            for key, val in self.triameta.items():
                outstr += f'trias.{key} = \n{val}\n'
            outstr += '\n'
        if self.quads is not None:
            outstr += f'quads = \n{self.quads}\n'
            for key, val in self.quadmeta.items():
                outstr += f'quads.{key} = \n{val}\n'
            outstr += '\n'
        return outstr

    def __repr__(self) -> str:
        return '<Mesh>'


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
            arr = arr.reshape(-1, 1)
        return arr


class MeshCache():
    grids: List[Tuple[float, float, float]] = None
    norms: List[Tuple[float, float, float]] = None
    lines: List[Tuple[int, int]] = None
    trias: List[Tuple[int, int, int]] = None
    quads: List[Tuple[int, int, int, int]] = None
    gridmeta: Dict[str, MetaCache] = None
    normmeta: Dict[str, MetaCache] = None
    linemeta: Dict[str, MetaCache] = None
    triameta: Dict[str, MetaCache] = None
    quadmeta: Dict[str, MetaCache] = None

    def __init__(self) -> None:
        self.grids = []
        self.norms = []
        self.lines = []
        self.trias = []
        self.quads = []
        self.gridmeta = {}
        self.normmeta = {}
        self.linemeta = {}
        self.triameta = {}
        self.quadmeta = {}

    def clear(self) -> None:
        self.grids.clear()
        self.norms.clear()
        self.lines.clear()
        self.trias.clear()
        self.quads.clear()
        self.gridmeta.clear()
        self.normmeta.clear()
        self.linemeta.clear()
        self.triameta.clear()
        self.quadmeta.clear()

    def add_grid_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.gridmeta[key] = MetaCache(key, dtype, default)

    def add_norm_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.normmeta[key] = MetaCache(key, dtype, default)

    def add_line_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.linemeta[key] = MetaCache(key, dtype, default)

    def add_tria_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.triameta[key] = MetaCache(key, dtype, default)

    def add_quad_meta(self, key: str, dtype: 'DTypeLike', default: Any) -> None:
        self.quadmeta[key] = MetaCache(key, dtype, default)

    def add_grid(self, x: float, y: float, z: float, **kwargs: Dict[str, Any]) -> None:
        for key in self.gridmeta.keys():
            value = kwargs.get(key, self.gridmeta[key].default)
            self.gridmeta[key].append(value)
        self.grids.append((x, y, z))

    def add_norm(self, x: float, y: float, z: float, **kwargs: Dict[str, Any]) -> None:
        for key in self.normmeta.keys():
            value = kwargs.get(key, self.normmeta[key].default)
            self.normmeta[key].append(value)
        self.norms.append((x, y, z))

    def add_line(self, a: int, b: int, **kwargs: Dict[str, Any]) -> None:
        for key in self.linemeta.keys():
            value = kwargs.get(key, self.linemeta[key].default)
            self.linemeta[key].append(value)
        self.lines.append((a, b))

    def add_tria(self, a: int, b: int, c: int, **kwargs: Dict[str, Any]) -> None:
        for key in self.triameta.keys():
            value = kwargs.get(key, self.triameta[key].default)
            self.triameta[key].append(value)
        self.trias.append((a, b, c))

    def add_quad(self, a: int, b: int, c: int, d: int, **kwargs: Dict[str, Any]) -> None:
        for key in self.quadmeta.keys():
            value = kwargs.get(key, self.quadmeta[key].default)
            self.quadmeta[key].append(value)
        self.quads.append((a, b, c, d))

    def to_mesh(self) -> Mesh:
        mesh = Mesh()
        mesh.grids = array(self.grids, dtype=float64).reshape(-1, 3)
        mesh.norms = array(self.norms, dtype=float64).reshape(-1, 3)
        mesh.lines = array(self.lines, dtype=int64).reshape(-1, 2)
        mesh.trias = array(self.trias, dtype=int64).reshape(-1, 3)
        mesh.quads = array(self.quads, dtype=int64).reshape(-1, 4)
        for key, val in self.gridmeta.items():
            valarr = val.asarray()
            if len(valarr.shape) == 1:
                valarr = valarr.reshape(-1, 1)
            mesh.gridmeta[key] = valarr
        for key, val in self.normmeta.items():
            valarr = val.asarray()
            if len(valarr.shape) == 1:
                valarr = valarr.reshape(-1, 1)
            mesh.normmeta[key] = valarr
        for key, val in self.linemeta.items():
            valarr = val.asarray()
            if len(valarr.shape) == 1:
                valarr = valarr.reshape(-1, 1)
            mesh.linemeta[key] = valarr
        for key, val in self.triameta.items():
            valarr = val.asarray()
            if len(valarr.shape) == 1:
                valarr = valarr.reshape(-1, 1)
            mesh.triameta[key] = valarr
        for key, val in self.quadmeta.items():
            valarr = val.asarray()
            if len(valarr.shape) == 1:
                valarr = valarr.reshape(-1, 1)
            mesh.quadmeta[key] = val.asarray()
        return mesh

    def __str__(self) -> str:
        outstr = ''
        if self.grids is not None:
            outstr += f'grids = \n{self.grids}\n'
            for key, val in self.gridmeta.items():
                outstr += f'grids.{key} = \n{val}\n'
            outstr += '\n'
        if self.norms is not None:
            outstr += f'norms = \n{self.norms}\n'
            for key, val in self.normmeta.items():
                outstr += f'norms.{key} = \n{val}\n'
            outstr += '\n'
        if self.lines is not None:
            outstr += f'lines = \n{self.lines}\n'
            for key, val in self.linemeta.items():
                outstr += f'lines.{key} = \n{val}\n'
            outstr += '\n'
        if self.trias is not None:
            outstr += f'trias = \n{self.trias}\n'
            for key, val in self.triameta.items():
                outstr += f'trias.{key} = \n{val}\n'
            outstr += '\n'
        if self.quads is not None:
            outstr += f'quads = \n{self.quads}\n'
            for key, val in self.quadmeta.items():
                outstr += f'quads.{key} = \n{val}\n'
            outstr += '\n'
        return outstr

    def __repr__(self) -> str:
        return '<MeshCache>'
