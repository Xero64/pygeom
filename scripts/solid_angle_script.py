#%%
# Import Dependencies
from math import pi

from pygeom.geom3d import (Vector, solid_angle_apex_trapzpyr,
                           solid_angle_tetrahedron)

#%%
# Calculate Solid Angle Attempt 1
v1 = Vector(1.0, 0.0, 0.0).to_unit()
v2 = Vector(0.0, 1.0, 0.0).to_unit()
v3 = Vector(0.0, 1.0, 1.0).to_unit()
v4 = Vector(1.0, 0.0, 1.0).to_unit()

sang = solid_angle_apex_trapzpyr(v1, v2, v3, v4)

print(f'Solid Angle: {sang:.6f} sterad')

val = sang/4/pi

print(f'Value: {val:.6f}')

#%%
# Calculate Solid Angle Attempt 2
v1 = Vector(1.0, 0.0, 1.0).to_unit()
v2 = Vector(0.0, 1.0, 1.0).to_unit()
v3 = Vector(0.0, 0.0, 1.0).to_unit()

sang = solid_angle_tetrahedron(v1, v2, v3)

print(f'Solid Angle: {sang:.6f} sterad')

val = sang/4/pi

print(f'Value: {val:.6f}')

#%%
# Calculate Solid Angle Attempt 3
v1 = Vector(1.0, 0.0, 0.0).to_unit()
v2 = Vector(0.0, 1.0, 0.0).to_unit()
v3 = Vector(0.0, 0.0, 1.0).to_unit()
v4 = Vector(0.0, 0.0, 1.0).to_unit()

sang = solid_angle_apex_trapzpyr(v1, v2, v3, v4)

print(f'Solid Angle: {sang:.6f} sterad')

val = sang/4/pi

print(f'Value: {val:.6f}')

#%%
# Calculate Solid Angle Attempt 4
v1 = Vector(1.0, 0.0, 0.0).to_unit()
v2 = Vector(0.0, 1.0, 0.0).to_unit()
v3 = Vector(0.0, 0.0, 1.0).to_unit()

sang = solid_angle_tetrahedron(v1, v2, v3)

print(f'Solid Angle: {sang:.6f} sterad')

val = sang/4/pi

print(f'Value: {val:.6f}')
