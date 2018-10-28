from .line import Line
from .vector import Vector
from numpy.matlib import zeros
from numpy.linalg import solve
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D

class CubicSpline(object):
    u"""This class stores a 3D parametric cubic spline."""
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
            self.pnls.append(Line(self.pnts[inda], self.pnts[indb]))
        if self.clsd:
            self.d2r = self.calc_d2r_closed()
            self.dr = self.calc_dr_closed()
        else:
            self.d2r = self.calc_d2r_open(tanA=self.tanA, tanB=self.tanB)
            self.dr = self.calc_dr_open()
#        self.th = self.calc_th()
        self.R = self.calc_R()
    def calc_d2r_open(self, tanA=None, tanB=None):
        u"""This function calculates the curvature of an open ended spline."""
        pnl_dx = [pnl.vec.x/pnl.length for pnl in self.pnls]
        pnl_dy = [pnl.vec.y/pnl.length for pnl in self.pnls]
        pnl_dz = [pnl.vec.z/pnl.length for pnl in self.pnls]
        del_dx = [0.]*self.npnts
        del_dy = [0.]*self.npnts
        del_dz = [0.]*self.npnts
        if tanA != None:
            utA = tanA.to_unit()
            dxA = utA.x
            dyA = utA.y
            dzA = utA.z
            del_dx[0] = pnl_dx[0]-dxA
            del_dy[0] = pnl_dy[0]-dyA
            del_dz[0] = pnl_dz[0]-dzA
        for i in range(1, self.npnts-1):
            del_dx[i] = pnl_dx[i]-pnl_dx[i-1]
            del_dy[i] = pnl_dy[i]-pnl_dy[i-1]
            del_dz[i] = pnl_dz[i]-pnl_dz[i-1]
        if tanB != None:
            utB = tanB.to_unit()
            dxB = utB.x
            dyB = utB.y
            dzB = utB.z
            del_dx[-1] = dxB-pnl_dx[-1]
            del_dy[-1] = dyB-pnl_dy[-1]
            del_dz[-1] = dzB-pnl_dz[-1]
        a = [0.]*self.npnts
        b = [1.]*self.npnts
        c = [0.]*self.npnts
        rx = [0.]*self.npnts
        ry = [0.]*self.npnts
        rz = [0.]*self.npnts
        if tanA != None:
            sB = self.pnls[0].length
            b[0] = sB/3
            c[0] = sB/6
            rx[0] = del_dx[0]
            ry[0] = del_dy[0]
            rz[0] = del_dz[0]
        for i in range(1, self.npnts-1):
            sA = self.pnls[i-1].length
            sB = self.pnls[i].length
            a[i] = sA/6
            b[i] = (sA+sB)/3
            c[i] = sB/6
            rx[i] = del_dx[i]
            ry[i] = del_dy[i]
            rz[i] = del_dz[i]
        if tanB != None:
            sA = self.pnls[-1].length
            a[-1] = sA/6
            b[-1] = sA/3
            rx[-1] = del_dx[-1]
            ry[-1] = del_dy[-1]
            rz[-1] = del_dz[-1]
        Γ = [0.]*self.npnts
        d2x = [0.]*self.npnts
        d2y = [0.]*self.npnts
        d2z = [0.]*self.npnts
        β = b[0]
        d2x[0] = rx[0]/β
        d2y[0] = ry[0]/β
        d2z[0] = rz[0]/β
        for i in range(1, self.npnts):
            Γ[i] = c[i-1]/β
            β = b[i]-a[i]*Γ[i]
            d2x[i] = (rx[i]-a[i]*d2x[i-1])/β
            d2y[i] = (ry[i]-a[i]*d2y[i-1])/β
            d2z[i] = (rz[i]-a[i]*d2z[i-1])/β
        for i in range(self.npnts-2, -1, -1):
            d2x[i] -= Γ[i+1]*d2x[i+1]
            d2y[i] -= Γ[i+1]*d2y[i+1]
            d2z[i] -= Γ[i+1]*d2z[i+1]
        d2r = [Vector(d2x[i], d2y[i], d2z[i]) for i in range(self.npnts)]
        return d2r
    def calc_d2r_closed(self):
        u"""This function calculates the curvature of a closed spline."""
        n = self.npnts
        inda = [i-1 for i in range(n)]
        indb = [i for i in range(n)]
        inda[0] = n-1
        pnl_dx = [pnl.vec.x/pnl.length for pnl in self.pnls]
        pnl_dy = [pnl.vec.y/pnl.length for pnl in self.pnls]
        pnl_dz = [pnl.vec.z/pnl.length for pnl in self.pnls]
        del_dx = [0.]*n
        del_dy = [0.]*n
        del_dz = [0.]*n
        for i in range(n):
            del_dx[i] = pnl_dx[indb[i]]-pnl_dx[inda[i]]
            del_dy[i] = pnl_dy[indb[i]]-pnl_dy[inda[i]]
            del_dz[i] = pnl_dz[indb[i]]-pnl_dz[inda[i]]
        A = zeros((n, n))
        B = zeros((n, 3))
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
                A[i, i+1] = sB/6
            B[i, 0] = del_dx[i]
            B[i, 1] = del_dy[i]
            B[i, 2] = del_dz[i]
        X = solve(A, B)
        d2x = [X[i, 0] for i in range(n)]
        d2y = [X[i, 1] for i in range(n)]
        d2z = [X[i, 2] for i in range(n)]
        d2r = [Vector(d2x[i], d2y[i], d2z[i]) for i in range(self.npnts)]
        return d2r
    def calc_dr_open(self):
        u"""This function calculates the gradient of an open ended spline."""
        dx = []
        dy = []
        dz = []
        for i in range(self.npnls):
            xA = self.pnts[i].x
            xB = self.pnts[i+1].x
            d2xA = self.d2r[i].x
            d2xB = self.d2r[i+1].x
            yA = self.pnts[i].y
            yB = self.pnts[i+1].y
            d2yA = self.d2r[i].y
            d2yB = self.d2r[i+1].y
            zA = self.pnts[i].z
            zB = self.pnts[i+1].z
            d2zA = self.d2r[i].z
            d2zB = self.d2r[i+1].z
            sP = self.pnls[i].length
            dxA = (xB-xA)/sP-sP/3*d2xA-sP/6*d2xB
            dyA = (yB-yA)/sP-sP/3*d2yA-sP/6*d2yB
            dzA = (zB-zA)/sP-sP/3*d2zA-sP/6*d2zB
            dx.append(dxA)
            dy.append(dyA)
            dz.append(dzA)
        dxB = (xB-xA)/sP+sP/6*d2xA+sP/3*d2xB
        dyB = (yB-yA)/sP+sP/6*d2yA+sP/3*d2yB
        dzB = (zB-zA)/sP+sP/6*d2zA+sP/3*d2zB
        dx.append(dxB)
        dy.append(dyB)
        dz.append(dzB)
        dr = [Vector(dx[i], dy[i], dz[i]) for i in range(self.npnts)]
        return dr
    def calc_dr_closed(self):
        u"""This function calculates the gradient of a closed spline."""
        n = self.npnts
        inda = [i for i in range(n)]
        indb = [i+1 for i in range(n)]
        indb[-1] = 0
        dx = []
        dy = []
        dz = []
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
            zA = self.pnts[ia].z
            zB = self.pnts[ib].z
            d2zA = self.d2r[ia].z
            d2zB = self.d2r[ib].z
            sP = self.pnls[i].length
            dxA = (xB-xA)/sP-sP/3*d2xA-sP/6*d2xB
            dyA = (yB-yA)/sP-sP/3*d2yA-sP/6*d2yB
            dzA = (zB-zA)/sP-sP/3*d2zA-sP/6*d2zB
            dx.append(dxA)
            dy.append(dyA)
            dz.append(dzA)
        dr = [Vector(dx[i], dy[i], dz[i]) for i in range(self.npnts)]
        return dr
    def calc_R(self):
        u"""This function calculates the radius of curvature of the spline."""
        R = []
        for i in range(self.npnts):
            dri = self.dr[i]
            d2ri = self.d2r[i]
            k = (dri**d2ri).return_magnitude()/(dri.return_magnitude())**3
            if k == 0.: 
                R.append(float('inf'))
            else:
                R.append(1/k)
        return R
    def spline_points(self, num=5):
        u"""This function interpolates the spline with a number of points."""
        x = []
        y = []
        z = []
        for i in range(self.npnls):
            ia = i
            xA = self.pnts[ia].x
            d2xA = self.d2r[ia].x
            yA = self.pnts[ia].y
            d2yA = self.d2r[ia].y
            zA = self.pnts[ia].z
            d2zA = self.d2r[ia].z
            ib = i+1
            if ib == self.npnts:
                ib = 0
            xB = self.pnts[ib].x
            d2xB = self.d2r[ib].x
            yB = self.pnts[ib].y
            d2yB = self.d2r[ib].y
            zB = self.pnts[ib].z
            d2zB = self.d2r[ib].z
            sP = self.pnls[i].length
            for j in range(num):
                s = j*sP/num
                A = (sP-s)/sP
                B = s/sP
                C = (A**3-A)*sP**2/6
                D = (B**3-B)*sP**2/6
                x.append(A*xA+B*xB+C*d2xA+D*d2xB)
                y.append(A*yA+B*yB+C*d2yA+D*d2yB)
                z.append(A*zA+B*zB+C*d2zA+D*d2zB)
        if self.clsd:
            x.append(self.pnts[0].x)
            y.append(self.pnts[0].y)
            z.append(self.pnts[0].z)
        else:
            x.append(self.pnts[-1].x)
            y.append(self.pnts[-1].y)
            z.append(self.pnts[-1].z)
        return x, y, z
    def spline_gradient(self, num=5):
        u"""This function interpolates the gradient of the spline."""
        dx = []
        dy = []
        dz = []
        for i in range(self.npnls):
            ia = i
            xA = self.pnts[ia].x
            d2xA = self.d2r[ia].x
            yA = self.pnts[ia].y
            d2yA = self.d2r[ia].y
            zA = self.pnts[ia].z
            d2zA = self.d2r[ia].z
            ib = i+1
            if ib == self.npnts:
                ib = 0
            xB = self.pnts[ib].x
            d2xB = self.d2r[ib].x
            yB = self.pnts[ib].y
            d2yB = self.d2r[ib].y
            zB = self.pnts[ib].z
            d2zB = self.d2r[ib].z
            sP = self.pnls[i].length
            for j in range(num):
                s = j*sP/num
                A = (sP-s)/sP
                B = s/sP
                dx.append((xB-xA)/sP-(3*A**2-1)*sP/6*d2xA+(3*B**2-1)*sP/6*d2xB)
                dy.append((yB-yA)/sP-(3*A**2-1)*sP/6*d2yA+(3*B**2-1)*sP/6*d2yB)
                dz.append((zB-zA)/sP-(3*A**2-1)*sP/6*d2zA+(3*B**2-1)*sP/6*d2zB)
        if self.clsd:
            dx.append(self.dr[0].x)
            dy.append(self.dr[0].y)
            dz.append(self.dr[0].z)
        else:
            dx.append(self.dr[-1].x)
            dy.append(self.dr[-1].y)
            dz.append(self.dr[-1].z)
        return dx, dy, dz
    def spline_curvature(self, num=1):
        u"""This function interpolates the curvature of the spline."""
        d2x = []
        d2y = []
        d2z = []
        for i in range(self.npnls):
            ia = i
            d2xA = self.d2r[ia].x
            d2yA = self.d2r[ia].y
            d2zA = self.d2r[ia].z
            ib = i+1
            if ib == self.npnts:
                ib = 0
            d2xB = self.d2r[ib].x
            d2yB = self.d2r[ib].y
            d2zB = self.d2r[ib].z
            sP = self.pnls[i].length
            for j in range(num):
                s = j*sP/num
                A = (sP-s)/sP
                B = s/sP
                d2x.append(A*d2xA+B*d2xB)
                d2y.append(A*d2yA+B*d2yB)
                d2z.append(A*d2zA+B*d2zB)
        if self.clsd:
            d2x.append(self.d2r[0].x)
            d2y.append(self.d2r[0].y)
            d2z.append(self.d2r[0].z)
        else:
            d2x.append(self.d2r[-1].x)
            d2y.append(self.d2r[-1].y)
            d2z.append(self.d2r[-1].z)
        return d2x, d2y, d2z
    def scatter(self, ax=None, label=False):
        u"""This function plots the points of the spline."""
        if ax == None:
            fig = figure()
            ax = Axes3D(fig)
            ax.grid(True)
        x = []
        y = []
        z = []
        for i in range(self.npnts):
            x.append(self.pnts[i].x)
            y.append(self.pnts[i].y)
            z.append(self.pnts[i].z)
        ax.scatter(x, y, z)
        if label:
            for i in range(self.npnts):
                ax.text(x[i], y[i], z[i], i)
        return ax
    def plot_spline(self, num=5, ax=None, color='blue'):
        u"""This function plots the spline using the interpolated points."""
        if ax == None:
            fig = figure()
            ax = Axes3D(fig)
            ax.grid(True)
        x, y, z = self.spline_points(num)
        ax.plot(x, y, z, color=color)
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
        dx, dy, dz = self.spline_gradient(num=num)
        ax.plot(s, dx, color='blue')
        ax.plot(s, dy, color='red')
        ax.plot(s, dz, color='green')
        return ax
    def quiver_tangent(self, ax=None, length=1., color='green'):
        u"""This function quiver plots the tangent of the spline."""
        if ax == None:
            fig = figure()
            ax = Axes3D(fig)
            ax.grid(True)
        x = [self.pnts[i].x for i in range(self.npnts)]
        y = [self.pnts[i].y for i in range(self.npnts)]
        z = [self.pnts[i].z for i in range(self.npnts)]
        dx = [self.dr[i].x for i in range(self.npnts)]
        dy = [self.dr[i].y for i in range(self.npnts)]
        dz = [self.dr[i].z for i in range(self.npnts)]
        ax.quiver(x, y, z, dx, dy, dz, length=length, color=color)
        return ax
    def plot_curvature(self, ax=None, num=5):
        u"""This function plots the curvature of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Curvature')
        s = self.arc_length(num=num)
        d2x, d2y, d2z = self.spline_curvature(num=num)
        ax.plot(s, d2x, color='blue')
        ax.plot(s, d2y, color='red')
        ax.plot(s, d2z, color='green')
        return ax
    def plot_inverse_radius(self, ax=None, num=5):
        u"""This function plots the inverse radius of curvature of the spline."""
        if ax == None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Inverse Radius of Curvature')
        s = self.arc_length(num=num)
        d2x, d2y, d2z = self.spline_curvature(num=num)
        dx, dy, dz = self.spline_gradient(num=num)
        nums = len(s)
        k = []
        for i in range(nums):
            dr = Vector(dx[i], dy[i], dz[i])
            d2r = Vector(d2x[i], d2y[i], d2z[i])
            k.append((dr**d2r).return_magnitude()/(dr.return_magnitude())**3)
        ax.plot(s, k, color='blue')
        return ax
    def quiver_normal(self, ax=None, length=1., color='red'):
        u"""This function quiver plots the normal of the spline."""
        if ax == None:
            fig = figure()
            ax = Axes3D(fig)
            ax.grid(True)
        x = [self.pnts[i].x for i in range(self.npnts)]
        y = [self.pnts[i].y for i in range(self.npnts)]
        z = [self.pnts[i].z for i in range(self.npnts)]
        d2x = [self.d2r[i].x for i in range(self.npnts)]
        d2y = [self.d2r[i].y for i in range(self.npnts)]
        d2z = [self.d2r[i].z for i in range(self.npnts)]
        ax.quiver(x, y, z, d2x, d2y, d2z, length=length, color=color)
        return ax
    def print_gradient(self):
        u"""This function prints the gradient of the spline."""
        outstr = '\nGradient\nID\tdxds\tdyds\tdzds'
        print(outstr)
        frmstr = '{:d}\t{:g}\t{:g}\t{:g}'
        for i in range(self.npnts):
            outstr = frmstr.format(i, round(self.dr[i].x, 6), round(self.dr[i].y, 6), round(self.dr[i].z, 6))
            print(outstr)
    def print_curvature(self):
        u"""This function prints the curvature of the spline."""
        outstr = '\nCurvature\nID\td2xds2\td2yds2\td2zds2\tRadius of Curvature'
        print(outstr)
        frmstr = '{:d}\t{:g}\t{:g}\t{:g}\t{:g}'
        for i in range(self.npnts):
            outstr = frmstr.format(i, round(self.d2r[i].x, 6), round(self.d2r[i].y, 6), round(self.d2r[i].z, 6), round(self.R[i], 6))
            print(outstr)
