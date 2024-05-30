#%%
# Import Dependencies
from pygeom.array3d import Mesh

#%%
# Create a Mesh object
mesh1 = Mesh()
mesh1.norms.add_meta('grids', int, -1)
mesh1.quads.add_meta('norms', int, (-1, -1, -1, -1))
mesh1.trias.add_meta('norms', int, (-1, -1, -1))
mesh1.grids.add(-1.0, -1.0, 0.0)
mesh1.grids.add(0.0, -1.0, 0.0)
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(-1.0, 0.0, 0.0)
mesh1.norms.add(0.0, 0.0, 1.0, grids=0)
mesh1.norms.add(0.0, 0.0, 1.0, grids=1)
mesh1.norms.add(0.0, 0.0, 1.0, grids=2)
mesh1.norms.add(0.0, 0.0, 1.0, grids=3)
mesh1.quads.add(0, 1, 2, 3, norms=[0, 1, 2, 3])
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(0.0, -1.0, 0.0)
mesh1.grids.add(1.0, -1.0, 0.0)
mesh1.norms.add(0.0, 0.0, 1.0, grids=4)
mesh1.norms.add(0.0, 0.0, 1.0, grids=5)
mesh1.norms.add(0.0, 0.0, 1.0, grids=6)
mesh1.trias.add(4, 5, 6, norms=[4, 5, 6])
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(1.0, -1.0, 0.0)
mesh1.grids.add(1.0, 0.0, 0.0)
mesh1.norms.add(0.0, 0.0, 1.0, grids=7)
mesh1.norms.add(0.0, 0.0, 1.0, grids=8)
mesh1.norms.add(0.0, 0.0, 1.0, grids=9)
mesh1.trias.add(7, 8, 9, norms=[7, 8, 9])
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(1.0, 0.0, 0.0)
mesh1.lines.add(10, 11)
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(1.0, 0.0, 0.0)
mesh1.lines.add(12, 13)
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.lines.add(14, 14)

mesh1.resolve_cache()

print(mesh1)

mesh1.remove_duplicate_grids()

print('Duplicate Grids Removed\n')

print(mesh1)

mesh1.remove_duplicate_norms()

print('Duplicate Norms Removed\n')

print(mesh1)

mesh1.remove_duplicate_lines()

print('Duplicate Lines Removed\n')

print(mesh1)

mesh1.lines.remove_collapsed()

print('Collapsed Lines Removed\n')

print(mesh1)

#%%
# Create a Mesh object
mesh2 = Mesh()
mesh2.norms.add_meta('grids', int, -1)
mesh2.quads.add_meta('norms', int, (-1, -1, -1, -1))
mesh2.trias.add_meta('norms', int, (-1, -1, -1))
mesh2.grids.add(0.0, 0.0, 0.0)
mesh2.grids.add(1.0, 0.0, 0.0)
mesh2.grids.add(1.0, 1.0, 0.0)
mesh2.grids.add(0.0, 1.0, 0.0)
mesh2.norms.add(0.0, 0.0, 1.0, grids=0)
mesh2.norms.add(0.0, 0.0, 1.0, grids=1)
mesh2.norms.add(0.0, 0.0, 1.0, grids=2)
mesh2.norms.add(0.0, 0.0, 1.0, grids=3)
mesh2.quads.add(0, 1, 2, 3, norms=[0, 1, 2, 3])
mesh2.grids.add(0.0, 0.0, 0.0)
mesh2.grids.add(0.0, 1.0, 0.0)
mesh2.grids.add(-1.0, 1.0, 0.0)
mesh2.norms.add(0.0, 0.0, 1.0, grids=4)
mesh2.norms.add(0.0, 0.0, 1.0, grids=5)
mesh2.norms.add(0.0, 0.0, 1.0, grids=6)
mesh2.trias.add(4, 5, 6, norms=[4, 5, 6])
mesh2.grids.add(0.0, 0.0, 0.0)
mesh2.grids.add(-1.0, 1.0, 0.0)
mesh2.grids.add(-1.0, 0.0, 0.0)
mesh2.norms.add(0.0, 0.0, 1.0, grids=7)
mesh2.norms.add(0.0, 0.0, 1.0, grids=8)
mesh2.norms.add(0.0, 0.0, 1.0, grids=9)
mesh2.trias.add(7, 8, 9, norms=[7, 8, 9])

mesh2.resolve_cache()

print(mesh2)

mesh2.remove_duplicate_grids()

print('Duplicate Grids Removed\n')

print(mesh2)

mesh2.remove_duplicate_norms()

print('Duplicate Norms Removed\n')

print(mesh2)

mesh2.remove_duplicate_lines()

print('Duplicate Lines Removed\n')

print(mesh2)

mesh2.lines.remove_collapsed()

print('Collapsed Lines Removed\n')

print(mesh2)

#%%
# Append Mesh2 to Mesh1
mesh3 = mesh1.merge(mesh2)

print(mesh3)

mesh3.remove_duplicate_grids()

print('Duplicate Grids Removed\n')

print(mesh3)

mesh3.remove_duplicate_norms()

print('Duplicate Norms Removed\n')

print(mesh3)

mesh3.remove_duplicate_lines()

print('Duplicate Lines Removed\n')

print(mesh3)

mesh3.lines.remove_collapsed()

print('Collapsed Lines Removed\n')

print(mesh3)

#%%
# Create a Mesh object
mesh4 = Mesh()
mesh4.norms.add_meta('grids', int, -1)
mesh4.quads.add_meta('norms', int, (-1, -1, -1, -1))
mesh4.trias.add_meta('norms', int, (-1, -1, -1))
mesh4.grids.add(0.0, 0.0, 0.0)
mesh4.grids.add(1.0, 0.0, 0.0)
mesh4.grids.add(1.0, 1.0, 0.0)
mesh4.grids.add(0.0, 1.0, 0.0)
mesh4.norms.add(0.0, 0.0, 1.0, grids=0)
mesh4.norms.add(0.0, 0.0, 1.0, grids=1)
mesh4.norms.add(0.0, 0.0, 1.0, grids=2)
mesh4.norms.add(0.0, 0.0, 1.0, grids=3)
mesh4.quads.add(0, 0, 1, 2, norms=[0, 0, 1, 2])
mesh4.quads.add(0, 1, 1, 2, norms=[0, 1, 1, 2])
mesh4.quads.add(0, 1, 2, 2, norms=[0, 1, 2, 2])
mesh4.quads.add(0, 1, 2, 2, norms=[0, 1, 2, 2])
mesh4.quads.add(0, 1, 0, 2, norms=[0, 1, 0, 2])
mesh4.quads.add(0, 1, 2, 1, norms=[0, 1, 2, 1])
mesh4.quads.add(0, 1, 2, 3, norms=[0, 1, 2, 3])

mesh4.resolve_cache()

print(mesh4)

mesh4.remove_duplicate_grids()

print('Duplicate Grids Removed\n')

print(mesh4)

mesh4.quads.remove_collapsed()

print('Collapsed Quads Removed\n')

print(mesh4)

mesh4.remove_duplicate_quads()

print('Duplicate Quads Removed\n')

print(mesh4)

mesh4.remove_duplicate_trias()

print('Duplicate Trias Removed\n')

print(mesh4)

mesh4.remove_unreferenced_grids()

print('Unreferenced Grids Removed\n')

print(mesh4)

mesh4.remove_unreferenced_norms()

print('Unreferenced Norms Removed\n')

print(mesh4)

mesh4.remove_unreferenced_grids()

print('Unreferenced Grids Removed\n')

print(mesh4)

mesh4.collapse_quads_to_trias()

print('Quads Collapsed to Trias\n')

print(mesh4)
