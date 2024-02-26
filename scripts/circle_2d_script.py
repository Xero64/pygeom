#%%
# Import Dependencies
from numpy import sqrt
from pygeom.geom2d import Vector2D, Circle2D

#%%
# Create Circle from 3 Points
pnta = Vector2D(-1.0, 0.0)
pntb = Vector2D(1.0, 0.0)
pntc = Vector2D(0.0, 2.0)

circle2d = Circle2D.from_3_points(pnta, pntb, pntc)
print(circle2d)
