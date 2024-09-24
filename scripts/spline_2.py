#%%
# Import Dependencies
from numpy import cos, linspace, pi, sin, zeros
from pygeom.geom3d import Vector
from pygeom.geom3d.cubicspline import CubicSpline
from pygeom.geom3d.spline import Spline
from pygeom.tools.mpl import (plot_curvature, plot_curve,
                              plot_first_derivatives, plot_points,
                              plot_second_derivatives)

#%%
# Create Spline 1
num = 36
r = 2.0
th = linspace(0.0, 2*pi, num)
x = zeros(th.shape)
y = r*cos(th)
z = r*sin(th)

pnts = Vector(x, y, z)

TestSpline = CubicSpline(pnts)

ax1 = plot_curve(TestSpline, label='Cubic Spline')
ax1.scatter(pnts.x, pnts.y, pnts.z, label='Spline Points')

ax1 = plot_points(TestSpline, label='Cubic Spline')
ax1 = plot_first_derivatives(TestSpline, label='Cubic Spline')
ax1 = plot_second_derivatives(TestSpline, label='Cubic Spline')
ax1 = plot_curvature(TestSpline, label='Cubic Spline')

TestSpline4 = Spline(pnts)

ax4 = TestSpline4.scatter(label=True)
ax4 = TestSpline4.plot_spline(50, ax=ax4)

#%%
# Create Spline 2
thA = pi/2
tanA = Vector(0.0, cos(thA), sin(thA))
thB = pi/2# - 2*pi/num
tanB = Vector(0.0, cos(thB), sin(thB))

bctype = ((1, tanA), (1, tanB))

TestSpline2 = CubicSpline(pnts, bctype=bctype)

ax2 = plot_curve(TestSpline2, label='Cubic Spline')
ax2.scatter(pnts.x, pnts.y, pnts.z, label='Spline Points')

ax2 = plot_points(TestSpline2, label='Cubic Spline')
ax2 = plot_first_derivatives(TestSpline2, label='Cubic Spline')
ax2 = plot_second_derivatives(TestSpline2, label='Cubic Spline')
ax2 = plot_curvature(TestSpline2, label='Cubic Spline')

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
TestSpline3 = CubicSpline(pnts, bctype='periodic')

ax3 = plot_curve(TestSpline3, label='Cubic Spline')
ax3.scatter(pnts.x, pnts.y, pnts.z, label='Spline Points')

ax3 = plot_points(TestSpline3, label='Cubic Spline')
ax3 = plot_first_derivatives(TestSpline3, label='Cubic Spline')
ax3 = plot_second_derivatives(TestSpline3, label='Cubic Spline')
ax3 = plot_curvature(TestSpline3, label='Cubic Spline')

TestSpline6 = Spline(pnts, closed=True)

ax6 = TestSpline6.plot_spline(50)

ax6 = TestSpline6.plot_gradient()
ax6 = TestSpline6.plot_curvature()
ax6 = TestSpline6.plot_inverse_radius()
ax6.set_ylim(0.0, 1.0)

ax6 = TestSpline6.scatter(label=True)
ax6 = TestSpline6.plot_spline(50, ax=ax6)
