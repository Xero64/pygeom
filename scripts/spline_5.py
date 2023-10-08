#%%
# Import Dependencies
from math import pi, cos, sin
from pygeom.geom3d import Vector, point_from_lists
from pygeom.geom3d.cubicspline import CubicSpline
from pygeom.geom3d.spline import Spline

#%%
# Create Spline 1
num = 4
R = 2.0
th = [i/num*2*pi for i in range(num)]
x = [0.0 for _ in th]
y = [R*cos(thi) for thi in th]
z = [R*sin(thi) for thi in th]

pnts = point_from_lists(x, y, z)
# pnts.append(pnts[0])

TestSpline = CubicSpline(pnts)

ax1 = TestSpline.scatter(label=True)
ax1 = TestSpline.plot_spline(50, ax=ax1)

TestSpline4 = Spline(pnts)

ax4 = TestSpline4.scatter(label=True)
ax4 = TestSpline4.plot_spline(50, ax=ax4)

#%%
# Create Spline 2
thA = pi/2
tanA = Vector(0.0, cos(thA), sin(thA))*10.0
thB = pi/2 - 2*pi/num
tanB = Vector(0.0, cos(thB), sin(thB))*0.5

TestSpline2 = CubicSpline(pnts, tanA=tanA, tanB=tanB)

ax2 = TestSpline2.plot_spline(50)

ax2 = TestSpline2.plot_gradient()
ax2 = TestSpline2.plot_curvature()

ax2 = TestSpline2.scatter(label=True)
ax2 = TestSpline2.plot_spline(50, ax=ax2)

TestSpline5 = Spline(pnts, tanA=tanA, tanB=tanB)

ax5 = TestSpline5.plot_spline(50)

ax5 = TestSpline5.plot_gradient()
ax5 = TestSpline5.plot_curvature()
ax5 = TestSpline5.plot_inverse_radius()
ax5.set_ylim(0.0, 1.0)

ax5 = TestSpline5.scatter(label=True)
ax5 = TestSpline5.plot_spline(50, ax=ax5)

#%%
# Create Spline 3
TestSpline3 = CubicSpline(pnts, clsd=True)

ax3 = TestSpline3.plot_spline(50)

ax3 = TestSpline3.plot_gradient()
ax3 = TestSpline3.plot_curvature()

ax3 = TestSpline3.scatter(label=True)
ax3 = TestSpline3.plot_spline(50, ax=ax3)

TestSpline6 = Spline(pnts, closed=True)

ax6 = TestSpline6.plot_spline(50)

ax6 = TestSpline6.plot_gradient()
ax6 = TestSpline6.plot_curvature()
ax6 = TestSpline6.plot_inverse_radius(50)
ax6.set_ylim(0.0, 1.0)

ax6 = TestSpline6.scatter(label=True)
ax6 = TestSpline6.plot_spline(50, ax=ax6)
