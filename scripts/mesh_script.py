#%%
# Import Dependencies
from pygeom.array3d.mesh import MeshCache

#%%
# Create a MeshCache object
mesh_cache = MeshCache()
mesh_cache.add_norm_meta('grids', int, -1)
mesh_cache.add_quad_meta('norms', int, -1)
mesh_cache.add_tria_meta('norms', int, -1)
mesh_cache.add_grid(-1.0, -1.0, 0.0)
mesh_cache.add_grid(0.0, -1.0, 0.0)
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(-1.0, 0.0, 0.0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=1)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=2)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=3)
mesh_cache.add_quad(0, 1, 2, 3, norms=[0, 1, 2, 3])
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(0.0, -1.0, 0.0)
mesh_cache.add_grid(1.0, -1.0, 0.0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=4)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=5)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=6)
mesh_cache.add_tria(4, 5, 6, norms=[4, 5, 6])
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(1.0, -1.0, 0.0)
mesh_cache.add_grid(1.0, 0.0, 0.0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=7)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=8)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=9)
mesh_cache.add_tria(7, 8, 9, norms=[7, 8, 9])
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(1.0, 0.0, 0.0)
mesh_cache.add_line(10, 11)
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(1.0, 0.0, 0.0)
mesh_cache.add_line(12, 13)
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_line(14, 14)

#%%
# Convert MeshCache to Mesh
mesh1 = mesh_cache.to_mesh()

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

mesh1.remove_collapsed_lines()

print('Collapsed Lines Removed\n')

print(mesh1)

#%%
# Create a MeshCache object
mesh_cache = MeshCache()
mesh_cache.add_norm_meta('grids', int, -1)
mesh_cache.add_quad_meta('norms', int, -1)
mesh_cache.add_tria_meta('norms', int, -1)
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(1.0, 0.0, 0.0)
mesh_cache.add_grid(1.0, 1.0, 0.0)
mesh_cache.add_grid(0.0, 1.0, 0.0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=1)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=2)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=3)
mesh_cache.add_quad(0, 1, 2, 3, norms=[0, 1, 2, 3])
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(0.0, 1.0, 0.0)
mesh_cache.add_grid(-1.0, 1.0, 0.0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=4)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=5)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=6)
mesh_cache.add_tria(4, 5, 6, norms=[4, 5, 6])
mesh_cache.add_grid(0.0, 0.0, 0.0)
mesh_cache.add_grid(-1.0, 1.0, 0.0)
mesh_cache.add_grid(-1.0, 0.0, 0.0)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=7)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=8)
mesh_cache.add_norm(0.0, 0.0, 1.0, grids=9)
mesh_cache.add_tria(7, 8, 9, norms=[7, 8, 9])

#%%
# Convert MeshCache to Mesh
mesh2 = mesh_cache.to_mesh()

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

mesh2.remove_collapsed_lines()

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

mesh3.remove_collapsed_lines()

print('Collapsed Lines Removed\n')

print(mesh3)
