#%%
# Import Dependencies
from pygeom.tools.mesh import Mesh, merge_meshes

#%%
# Create a Mesh Template
mesht = Mesh()
mesht.add_mesh_vectors('norms', 'MeshNorms')
mesht.attrs['norms'].add_meta('grids', int, -1)
mesht.quads.add_meta('norms', int, (-1, -1, -1, -1))
mesht.trias.add_meta('norms', int, (-1, -1, -1))
mesht.resolve_cache()
print(mesht.mesh_template)

#%%
# Create a Mesh object
mesh1 = mesht.new_mesh_from_template()
mesh1.grids.add(-1.0, -1.0, 0.0)
mesh1.grids.add(0.0, -1.0, 0.0)
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(-1.0, 0.0, 0.0)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=0)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=1)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=2)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=3)
mesh1.quads.add(0, 1, 2, 3, norms=[0, 1, 2, 3])
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(0.0, -1.0, 0.0)
mesh1.grids.add(1.0, -1.0, 0.0)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=4)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=5)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=6)
mesh1.trias.add(4, 5, 6, norms=[4, 5, 6])
mesh1.grids.add(0.0, 0.0, 0.0)
mesh1.grids.add(1.0, -1.0, 0.0)
mesh1.grids.add(1.0, 0.0, 0.0)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=7)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=8)
mesh1.attrs['norms'].add(0.0, 0.0, 1.0, grids=9)
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

mesh1.remove_duplicate_vectors('norms')

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
mesh2 = mesht.new_mesh_from_template()
mesh2.grids.add(0.0, 0.0, 0.0)
mesh2.grids.add(1.0, 0.0, 0.0)
mesh2.grids.add(1.0, 1.0, 0.0)
mesh2.grids.add(0.0, 1.0, 0.0)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=0)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=1)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=2)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=3)
mesh2.quads.add(0, 1, 2, 3, norms=[0, 1, 2, 3])
mesh2.grids.add(0.0, 0.0, 0.0)
mesh2.grids.add(0.0, 1.0, 0.0)
mesh2.grids.add(-1.0, 1.0, 0.0)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=4)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=5)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=6)
mesh2.trias.add(4, 5, 6, norms=[4, 5, 6])
mesh2.grids.add(0.0, 0.0, 0.0)
mesh2.grids.add(-1.0, 1.0, 0.0)
mesh2.grids.add(-1.0, 0.0, 0.0)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=7)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=8)
mesh2.attrs['norms'].add(0.0, 0.0, 1.0, grids=9)
mesh2.trias.add(7, 8, 9, norms=[7, 8, 9])

mesh2.resolve_cache()

print(mesh2)

mesh2.remove_duplicate_grids()

print('Duplicate Grids Removed\n')

print(mesh2)

mesh2.remove_duplicate_vectors('norms')

print('Duplicate Norms Removed\n')

print(mesh2)

mesh2.remove_duplicate_lines()

print('Duplicate Lines Removed\n')

print(mesh2)

mesh2.lines.remove_collapsed()

print('Collapsed Lines Removed\n')

print(mesh2)

#%%
# Merge Mesh 1 and Mesh 2
mesh3 = merge_meshes(mesh1, mesh2)

print(mesh3)

mesh3.remove_duplicate_grids()

print('Duplicate Grids Removed\n')

print(mesh3)

mesh3.remove_duplicate_vectors('norms')

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
mesh4 = mesht.new_mesh_from_template()
mesh4.grids.add(0.0, 0.0, 0.0)
mesh4.grids.add(1.0, 0.0, 0.0)
mesh4.grids.add(1.0, 1.0, 0.0)
mesh4.grids.add(0.0, 1.0, 0.0)
mesh4.attrs['norms'].add(0.0, 0.0, 1.0, grids=0)
mesh4.attrs['norms'].add(0.0, 0.0, 1.0, grids=1)
mesh4.attrs['norms'].add(0.0, 0.0, 1.0, grids=2)
mesh4.attrs['norms'].add(0.0, 0.0, 1.0, grids=3)
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

mesh4.remove_unreferenced_vectors('norms')

print('Unreferenced Norms Removed\n')

print(mesh4)

mesh4.remove_unreferenced_grids()

print('Unreferenced Grids Removed\n')

print(mesh4)

mesh4.collapse_quads_to_trias()

print('Quads Collapsed to Trias\n')

print(mesh4)
