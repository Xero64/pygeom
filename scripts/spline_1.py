#%%
# Import Dependencies
from math import pi, cos, sin
from pygeom.geom2d.vector2d import Vector2D
from pygeom.geom2d import point2d_from_lists
from pygeom.geom2d.spline2d import Spline2D

#%%
# Create Spline 1
num = 36
R = 2.0
th = [i/num*2*pi for i in range(num)]
x = [R*cos(thi) for thi in th]
y = [R*sin(thi) for thi in th]

pnts = point2d_from_lists(x, y)

TestSpline = Spline2D(pnts)
ax = TestSpline.scatter(label=True)
ax = TestSpline.plot_spline(50, ax=ax)

#%%
# Create Spline 2
thA = pi/2
tanA = Vector2D(cos(thA), sin(thA))
thB = pi/2-2*pi/num
tanB = Vector2D(cos(thB), sin(thB))

TestSpline2 = Spline2D(pnts, tanA=tanA, tanB=tanB)
ax = TestSpline2.plot_spline(50, ax=ax, color='green')

ax1 = TestSpline2.plot_gradient()
ax2 = TestSpline2.plot_curvature()

#%%
# Create Spline 3
TestSpline3 = Spline2D(pnts, closed=True)
ax = TestSpline3.plot_spline(50, ax=ax, color='red')

ax3 = TestSpline3.plot_gradient()
ax4 = TestSpline3.plot_curvature()

ax = TestSpline3.plot_spline(10)
ax = TestSpline3.quiver_normal(ax=ax)
ax = TestSpline3.quiver_tangent(ax=ax)
