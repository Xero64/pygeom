from .line2d import Line2D
from .point2d import Point2D
from .vector2d import Vector2D
from numpy.matlib import zeros
from numpy.linalg import solve
from matplotlib.pyplot import figure

class CubicSpline2D(object):
    u"""This class stores a 2D parametric cubic spline."""
    pnts = None
    npnts = None
    clsd = False
    pnls = None
    npnls = None
    d2r = None
    dr = None
    tanA = None
    tanB = None
    R = None
    def __init__(self, pnts, clsd=False, tanA=None, tanB=None):
        u"""This function initialises the object."""
        self.pnts = pnts
        self.clsd = clsd
        self.tanA = tanA
        self.tanB = tanB
        self.update()
    def update(self):
        u"""This function calculates the other parameters of the object."""
        self.npnts = len(self.pnts)
        if self.clsd:
            indas = [i for i in range(self.npnts)]
            indbs = [i+1 for i in range(self.npnts)]
            indbs[-1] = 0
            self.npnls = self.npnts
        else:
            indas = [i for i in range(self.npnts-1)]
            indbs = [i+1 for i in range(self.npnts-1)]
            self.npnls = self.npnts-1
        self.pnls = []
        for i in range(self.npnls):
            inda = indas[i]
            indb = indbs[i]
            self.pnls.append(Line2D(self.pnts[inda], self.pnts[indb]))
        if self.clsd:
            self.d2r = self.calc_d2r_closed()
            self.dr = self.calc_dr_closed()
        else:
            self.d2r = self.calc_d2r_open(tanA=self.tanA, tanB=self.tanB)
            self.dr = self.calc_dr_open()
        self.R = self.calc_R()
    def calc_d2r_open(self, tanA=None, tanB=None):
        u"""This function calculates the curvature of an open ended spline."""
        pnl_dx = [pnl.vec.x/pnl.length for pnl in self.pnls]
        pnl_dy = [pnl.vec.y/pnl.length for pnl in self.pnls]
        del_dx = [0.]*self.npnts
        del_dy = [0.]*self.npnts
        if tanA != None:
            dxA = tanA.x
            dyA = tanA.y
            del_dx[0] = pnl_dx[0]-dxA
            del_dy[0] = pnl_dy[0]-dyA
        for i in range(1, self.npnts-1):
            del_dx[i] = pnl_dx[i]-pnl_dx[i-1]
            del_dy[i] = pnl_dy[i]-pnl_dy[i-1]
        if tanB != None:
            dxB = tanB.x
            dyB = tanB.y
            del_dx[-1] = dxB-pnl_dx[-1]
            del_dy[-1] = dyB-pnl_dy[-1]
        a = [0.]*self.npnts
        b = [1.]*self.npnts
        c = [0.]*self.npnts
        rx = [0.]*self.npnts
        ry = [0.]*self.npnts
        if tanA != None:
            sB = self.pnls[0].length
            b[0] = sB/3
            c[0] = sB/6
            rx[0] = del_dx[0]
            ry[0] = del_dy[0]
        for i in range(1, self.npnts-1):
            sA = self.pnls[i-1].length
            sB = self.pnls[i].length
            a[i] = sA/6
            b[i] = (sA+sB)/3
            c[i] = sB/6
            rx[i] = del_dx[i]
            ry[i] = del_dy[i]
        if tanB != None:
            sA = self.pnls[-1].length
            a[-1] = sA/6
            b[-1] = sA/3
            rx[-1] = del_dx[-1]
            ry[-1] = del_dy[-1]
        Γ = [0.]*self.npnts
        d2x = [0.]*self.npnts
        d2y = [0.]*self.npnts
        β = b[0]
        d2x[0] = rx[0]/β
        d2y[0] = ry[0]/β
        for i in range(1, self.npnts):
            Γ[i] = c[i-1]/β
            β = b[i]-a[i]*Γ[i]
            d2x[i] = (rx[i]-a[i]*d2x[i-1])/β
            d2y[i] = (ry[i]-a[i]*d2y[i-1])/β
        for i in range(self.npnts-2, -1, -1):
            d2x[i] -= Γ[i+1]*d2x[i+1]
            d2y[i] -= Γ[i+1]*d2y[i+1]
        d2r = [Vector2D(d2x[i], d2y[i]) for i in range(self.npnts)]
        return d2r
    def calc_d2r_closed(self):
        u"""This function calculates the curvature of a closed spline."""
        n = self.npnts
        inda = [i-1 for i in range(n)]
        indb = [i for i in range(n)]
        inda[0] = n-1
        pnl_dx = [pnl.vec.x/pnl.length for pnl in self.pnls]
        pnl_dy = [pnl.vec.y/pnl.length for pnl in self.pnls]
        del_dx = [0.]*n
        del_dy = [0.]*n
        for i in range(n):
            del_dx[i] = pnl_dx[indb[i]]-pnl_dx[inda[i]]
            del_dy[i] = pnl_dy[indb[i]]-pnl_dy[inda[i]]
        A = zeros((n, n))
        B = zeros((n, 2))
        for i in range(n):
            sA = self.pnls[inda[i]].length
            sB = self.pnls[indb[i]].length
            if i-1 < 0:
                A[i, n-1] = sA/6
            else:
                A[i, i-1] = sA/6
            A[i, i] = (sA+sB)/3
            if i+1 > n-1:
                A[i, 0] = sB/6
            else:
                A[i,i+1] = sB/6
            B[i, 0] = del_dx[i]
            B[i, 1] = del_dy[i]
        X = solve(A, B)
        d2x = [X[i, 0] for i in range(n)]
        d2y = [X[i, 1] for i in range(n)]
        d2r = [Vector2D(d2x[i], d2y[i]) for i in range(self.npnts)]
        return d2r
    def calc_dr_open(self):
        u"""This function calculates the gradient of an open ended spline."""
        dx = []
        dy = []
        for i in range(self.npnls):
            xA = self.pnts[i].x
            xB = self.pnts[i+1].x
            d2xA = self.d2r[i].x
            d2xB = self.d2r[i+1].x
            yA = self.pnts[i].y
            yB = self.pnts[i+1].y
            d2yA = self.d2r[i].y
            d2yB = self.d2r[i+1].y
            sP = self.pnls[i].length
            dxA = (xB-xA)/sP-sP/3*d2xA-sP/6*d2xB
            dyA = (yB-yA)/sP-sP/3*d2yA-sP/6*d2yB
            dx.append(dxA)
            dy.append(dyA)
        dxB = (xB-xA)/sP+sP/6*d2xA+sP/3*d2xB
        dyB = (yB-yA)/sP+sP/6*d2yA+sP/3*d2yB
        dx.append(dxB)
        dy.append(dyB)
        dr = [Vector2D(dx[i], dy[i]) for i in range(self.npnts)]
        return dr
    def calc_dr_closed(self):
        u"""This function calculates the gradient of a closed spline."""
        n = self.npnts
        inda = [i for i in range(n)]
        indb = [i+1 for i in range(n)]
        indb[-1] = 0
        dx = []
        dy = []
        for i in range(self.npnls):
            ia = inda[i]
            ib = indb[i]
            xA = self.pnts[ia].x
            xB = self.pnts[ib].x
            d2xA = self.d2r[ia].x
            d2xB = self.d2r[ib].x
            yA = self.pnts[ia].y
            yB = self.pnts[ib].y
            d2yA = self.d2r[ia].y
            d2yB = self.d2r[ib].y
            sP = self.pnls[i].length
            dxA = (xB-xA)/sP-sP/3*d2xA-sP/6*d2xB
            dyA = (yB-yA)/sP-sP/3*d2yA-sP/6*d2yB
            dx.append(dxA)
            dy.append(dyA)
        dr = [Vector2D(dx[i], dy[i]) for i in range(self.npnts)]
        return dr
    def calc_R(self):
        u"""This function calculates the radius of curvature of the spline."""
        R = []
        for i in range(self.npnts):
            dx = self.dr[i].x
            dy = self.dr[i].y
            d2x = self.d2r[i].x
            d2y = self.d2r[i].y
            k = (dx*d2y-dy*d2x)/(dx**2+dy**2)**1.5
            if k == 0.:
                R.append(float('inf'))
            else:
                R.append(1/k)
        return R
    def spline_points(self, num):
        u"""This function interpolates the spline with a number of points."""
        x = []
        y = []
        for i in range(self.npnls):
            ia = i
            xA = self.pnts[ia].x
            d2xA = self.d2r[ia].x
            yA = self.pnts[ia].y
            d2yA = self.d2r[ia].y
            ib = i+1
            if ib == self.npnts:
                ib = 0
            xB = self.pnts[ib].x
            d2xB = self.d2r[ib].x
            yB = self.pnts[ib].y
            d2yB = self.d2r[ib].y
            sP = self.pnls[i].length
            for j in range(num):
                s = j*sP/num
                A = (sP-s)/sP
                B = s/sP
                C = (A**3-A)*sP**2/6
                D = (B**3-B)*sP**2/6
                x.append(A*xA+B*xB+C*d2xA+D*d2xB)
                y.append(A*yA+B*yB+C*d2yA+D*d2yB)
        if self.clsd:
            x.append(self.pnts[0].x)
            y.append(self.pnts[0].y)
        else:
            x.append(self.pnts[-1].x)
            y.append(self.pnts[-1].y)
        return x, y
    def spline_gradient(self, num=5):
        u"""This function interpolates the gradient of the spline."""
        dx = []
        dy = []
        for i in range(self.npnls):
            ia = i
            xA = self.pnts[ia].x
            d2xA = self.d2r[ia].x
            yA = self.pnts[ia].y
            d2yA = self.d2r[ia].y
            ib = i+1
            if ib == self.npnts:
                ib = 0
            xB = self.pnts[ib].x
            d2xB = self.d2r[ib].x
            yB = self.pnts[ib].y
            d2yB = self.d2r[ib].y
            sP = self.pnls[i].length
            for j in range(num):
                s = j*sP/num
                A = (sP-s)/sP
                B = s/sP
                dx.append((xB-xA)/sP-(3*A**2-1)*sP/6*d2xA+(3*B**2-1)*sP/6*d2xB)
                dy.append((yB-yA)/sP-(3*A**2-1)*sP/6*d2yA+(3*B**2-1)*sP/6*d2yB)
        if self.clsd:
            dx.append(self.dr[0].x)
            dy.append(self.dr[0].y)
        else:
            dx.append(self.dr[-1].x)
            dy.append(self.dr[-1].y)
        return dx, dy
    def spline_curvature(self, num=1):
        u"""This function interpolates the curvature of the spline."""
        d2x = []
        d2y = []
        for i in range(self.npnls):
            ia = i
            d2xA = self.d2r[ia].x
            d2yA = self.d2r[ia].y
            ib = i+1
            if ib == self.npnts:
                ib = 0
            d2xB = self.d2r[ib].x
            d2yB = self.d2r[ib].y
            sP = self.pnls[i].length
            for j in range(num):
                s = j*sP/num
                A = (sP-s)/sP
                B = s/sP
                d2x.append(A*d2xA+B*d2xB)
                d2y.append(A*d2yA+B*d2yB)
        if self.clsd:
            d2x.append(self.d2r[0].x)
            d2y.append(self.d2r[0].y)
        else:
            d2x.append(self.d2r[-1].x)
            d2y.append(self.d2r[-1].y)
        return d2x, d2y
    # def line_intersection_points(self, line):
    #     from pymath.roots import cubic_equation
    #     pnts = []
    #     xP = line.pnt.x
    #     yP = line.pnt.y
    #     dxdl = line.uvec.x
    #     dydl = line.uvec.y
    #     for i in range(self.npnls):
    #         ia = i
    #         xA = self.pnts[ia].x
    #         yA = self.pnts[ia].y
    #         d2xA = self.d2r[ia].x
    #         d2yA = self.d2r[ia].y
    #         ib = i+1
    #         if ib == self.npnts:
    #             ib = 0
    #         xB = self.pnts[ib].x
    #         yB = self.pnts[ib].y
    #         d2xB = self.d2r[ib].x
    #         d2yB = self.d2r[ib].y
    #         sP = self.pnls[i].length
    #         a = (-d2xA*dydl + d2xB*dydl + d2yA*dxdl - d2yB*dxdl)/(6*dxdl*dydl*sP)
    #         b = (d2xA*dydl - d2yA*dxdl)/(2*dxdl*dydl)
    #         c = (-2*d2xA*dydl*sP**2 - d2xB*dydl*sP**2 + 2*d2yA*dxdl*sP**2 + d2yB*dxdl*sP**2 + 6*dxdl*yA - 6*dxdl*yB - 6*dydl*xA + 6*dydl*xB)/(6*dxdl*dydl*sP)
    #         d = (-dxdl*yA + dxdl*yP + dydl*xA - dydl*xP)/(dxdl*dydl)
    #         slst = cubic_equation(a, b, c, d)
    #         for s in slst:
    #             if isinstance(s, float):
    #                 if s >= 0.0 and s <= sP:
    #                     A = (sP-s)/sP
    #                     B = s/sP
    #                     C = sP**2/6*(A**3-A)
    #                     D = sP**2/6*(B**3-B)
    #                     x = A*xA+B*xB+C*d2xA+D*d2xB
    #                     y = A*yA+B*yB+C*d2yA+D*d2yB
    #                     pnts.append(Point2D(x, y))
    #     return pnts
    def scatter(self, ax=None, label=False):
        u"""This function plots the points of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x = []
        y = []
        for i in range(self.npnts):
            x.append(self.pnts[i].x)
            y.append(self.pnts[i].y)
        ax.scatter(x,y)
        if label:
            for i in range(self.npnts):
                ax.text(x[i], y[i], i)
        return ax
    def plot_spline(self, num=1, ax=None, color='blue'):
        u"""This function plots the spline using the interpolated points."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x, y = self.spline_points(num)
        ax.plot(x, y, color=color)
        return ax
    def arc_length(self, num=1):
        u"""This function calculates the arc length of the spline."""
        s = []
        sc = 0.
        for i in range(self.npnls):
            sP = self.pnls[i].length
            for j in range(num):
                sj = j*sP/num
                s.append(sc+sj)
            sc += sP
        s.append(sc)
        return s
    def plot_gradient(self, ax=None, num=5):
        u"""This function plots the gradient of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Gradient')
        s = self.arc_length(num=num)
        dx, dy = self.spline_gradient(num=num)
        ax.plot(s, dx, color='blue')
        ax.plot(s, dy, color='red')
        return ax
    def quiver_tangent(self, ax=None, color='green'):
        u"""This function quiver plots the tangent of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x = [self.pnts[i].x for i in range(self.npnts)]
        y = [self.pnts[i].y for i in range(self.npnts)]
        dx = [self.dr[i].x for i in range(self.npnts)]
        dy = [self.dr[i].y for i in range(self.npnts)]
        ax.quiver(x, y, dx, dy, color=color)
        return ax
    def plot_curvature(self, ax=None, num=1):
        u"""This function plots the curvature of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Curvature')
        s = self.arc_length(num=num)
        d2x, d2y = self.spline_curvature(num=num)
        ax.plot(s, d2x, color='blue')
        ax.plot(s, d2y, color='red')
        return ax
    def plot_inverse_radius(self, ax=None, num=5):
        u"""This function plots the inverse radius of curvature of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Inverse Radius of Curvature')
        s = self.arc_length(num=num)
        d2x, d2y = self.spline_curvature(num=num)
        dx, dy = self.spline_gradient(num=num)
        nums = len(s)
        k = [(dx[i]*d2y[i]-dy[i]*d2x[i])/(dx[i]**2+dy[i]**2)**1.5 for i in range(nums)]
        ax.plot(s, k, color='blue')
        return ax
    def quiver_normal(self, ax=None, color='red'):
        u"""This function quiver plots the normal of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x = [self.pnts[i].x for i in range(self.npnts)]
        y = [self.pnts[i].y for i in range(self.npnts)]
        d2x = [self.d2r[i].x for i in range(self.npnts)]
        d2y = [self.d2r[i].y for i in range(self.npnts)]
        ax.quiver(x, y, d2x, d2y, color=color)
        return ax
    def print_gradient(self):
        u"""This function prints the gradient of the spline."""
        outstr = '\nGradient\nID\tdxds\tdyds\tTangent'
        print(outstr)
        frmstr = '{:d}\t{:g}\t{:g}'
        for i in range(self.npnts):
            outstr = frmstr.format(i, round(self.dr[i].x, 6), round(self.dr[i].y, 6))
            print(outstr)
    def print_curvature(self):
        u"""This function prints the curvature of the spline."""
        outstr = '\nCurvature\nID\td2xds2\td2yds2\tRadius'
        print(outstr)
        frmstr = '{:d}\t{:g}\t{:g}\t{:g}'
        for i in range(self.npnts):
            outstr = frmstr.format(i, round(self.d2r[i].x, 6), round(self.d2r[i].x, 6), round(self.R[i], 6))
            print(outstr)
