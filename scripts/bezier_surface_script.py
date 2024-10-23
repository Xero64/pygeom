#%%
# Import Dependencies
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import sqrt, zeros

from pygeom.geom3d import RationalBezierSurface, Vector

#%%
# Define the control points and weights
numu = 19
numv = 19

ro = 3.0
ri = 2.0

ctlpts = Vector.zeros((3, 3))
ctlpts[0, 0] = Vector(ro, 0.0, 0.0)
ctlpts[1, 0] = Vector(ro, ro, 0.0)
ctlpts[2, 0] = Vector(0.0, ro, 0.0)
ctlpts[0, 1] = Vector(ro, 0.0, ro - ri)
ctlpts[1, 1] = Vector(ro, ro, ro - ri)
ctlpts[2, 1] = Vector(0.0, ro, ro - ri)
ctlpts[0, 2] = Vector(ri, 0.0, ro - ri)
ctlpts[1, 2] = Vector(ri, ri, ro - ri)
ctlpts[2, 2] = Vector(0.0, ri, ro - ri)

weights = zeros((3, 3))
weights[0, 0] = 1.0
weights[1, 0] = 1.0/sqrt(2.0)
weights[2, 0] = 1.0
weights[0, 1] = 1.0/sqrt(2.0)
weights[1, 1] = 0.5
weights[2, 1] = 1.0/sqrt(2.0)
weights[0, 2] = 1.0
weights[1, 2] = 1.0/sqrt(2.0)
weights[2, 2] = 1.0

print(f'Control Points:\n{ctlpts}\n')
print(f'Weights:\n{weights}\n')

beziersurface = RationalBezierSurface(ctlpts, weights)

pnts = beziersurface.evaluate_points(numu, numv)

#%%
# Plot the NURBS Surface
fig = figure(figsize=(12, 8))
ax = Axes3D(fig)
fig.add_axes(ax)
ax.view_init(elev=45, azim=45, roll=0)
ax.grid(True)
ax.plot(ctlpts.x, ctlpts.y, ctlpts.z, 'ro', label='Control Points')
ax.plot_wireframe(pnts.x, pnts.y, pnts.z, label='NURBS Surface')
ax.set_box_aspect([ro, ro, ro-ri])
_ = ax.legend()
