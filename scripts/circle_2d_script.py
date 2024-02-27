#%%
# Import Dependencies
from pygeom.geom2d import Vector2D
from pygeom.geom2d.circle2d import circle2d_from_3_points

#%%
# Create Circle from 3 Points
pnta = Vector2D(-1.0, 0.0)
pntb = Vector2D(1.0, 0.0)
pntc = Vector2D(0.0, 2.0)

circle2d = circle2d_from_3_points(pnta, pntb, pntc)
print(circle2d)
