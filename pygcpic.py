import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time
import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import convert as c

epsilon0 = 8.854e-12
e = 1.602e-19
mp = 1.67e-27
me = 9.11e-31
kb = 1.38e-23

class Particle:
    def __init__(self, m, q, p2c, T, B=np.zeros(3), E=np.zeros(3), grid=None):
        self.r = np.empty(7)
        self.q = q
        self.m = m
        self.T = T
        self.p2c = p2c
        self.vth = np.sqrt(2.0*kb*self.T/self.m)
        #6D mode: q = x,y,z,vx,vy,vz,t
        #GC mode: q = X,Y,Z,vpar,mu,_,t
        self.mode = 0
        #6D mode: 0
        #GC mode: 1
        self.E = E
        self.B = B
        #Electric field at particle position
        if grid != None: self.initialize_6D(grid)
    #end def __init__

    def __repr__(self):
        return f"Particle({self.m}, {self.q}, {self.p2c}, {self.T})"
    #end def

    def get_speed(self):
        return np.sqrt(self.r[3]**2 + self.r[4]**2 + self.r[5]**2)
    #end def get_speed

    def initialize_6D(self, grid):
        self.r[0] = np.random.uniform(0.0, grid.length)
        self.r[1:3] = 0.0
        self.r[3:6] = np.random.normal(0.0, self.vth , 3)
        self.r[6] = 0.0
    #end def initialize_6D

    def interpolate_electric_field_dirichlet(self, grid):
        ind = int(np.floor(self.r[0]/grid.dx))
        w_l = self.r[0]%grid.dx/grid.dx
        w_r = 1.0 - w_l
        self.E[0] = grid.E[ind]*w_l + grid.E[ind+1]*w_r
    #end def interpolate_electric_field

    def push_6D(self,dt):
        """Boris-Buneman integrator."""
        constant = 0.5*dt*self.q/self.m

        self.r[3] += constant*self.E[0]
        self.r[4] += 0.
        self.r[5] += 0.

        tx = constant*self.B[0]
        ty = constant*self.B[1]
        tz = constant*self.B[2]

        t2 = tx*tx + ty*ty + tz*tz

        sx = 2.*tx/(1. + t2)
        sy = 2.*ty/(1. + t2)
        sz = 2.*tz/(1. + t2)

        vfx = self.r[3] + self.r[4]*tz - self.r[5]*ty
        vfy = self.r[4] + self.r[5]*tx - self.r[3]*tz
        vfz = self.r[5] + self.r[3]*ty - self.r[4]*tx

        self.r[3] += vfy*sz - vfz*sy
        self.r[4] += vfz*sx - vfx*sz
        self.r[5] += vfx*sy - vfy*sx

        self.r[3] += constant*self.E[0]
        self.r[4] += 0.
        self.r[5] += 0.

        self.r[0] += self.r[3]*dt
        self.r[1] += self.r[4]*dt
        self.r[2] += self.r[5]*dt

        self.r[6] += dt
    #end push_6D

    def transform_6D_to_GC(self):
        x = self.r[0:3]
        v = self.r[3:6]
        B2 = self.B[0]**2 + self.B[1]**2 + self.B[2]**2
        b = self.B/np.sqrt(B2)

        vpar_mag = v.dot(b)
        vpar = vpar_mag*b
        wc = abs(self.q)*np.sqrt(B2)/self.m
        rho = vpar_mag/wc
        vperp = v - vpar
        vperp_mag = np.sqrt(vperp[0]**2 + vperp[1]**2 + vperp[2]**2)
        vperp_hat = vperp/vperp_mag
        mu = 0.5*self.m*vperp_mag**2/np.sqrt(B2)
        rl_mag = vperp_mag/wc
        rl_hat = -np.sign(self.q)*np.cross(vperp_hat,b)
        rl = rl_mag*rl_hat

        self.r[0:3] = x - rl
        self.r[3] = vpar_mag
        self.r[4] = mu
        self.mode == 1
    #end def transform_6D_to_GC

    def transform_GC_to_6D(self):
        X = self.r[0:3]
        vpar_mag = self.r[3]
        mu = self.r[4]
        B2 = self.B[0]**2 + self.B[1]**2 + self.B[2]**2
        b = self.B/np.sqrt(B2)

        vperp_mag = np.sqrt(2.0*mu*np.sqrt(B2)/self.m)
        wc = abs(self.q)*np.sqrt(B2)/self.m
        rl_mag = vperp_mag/wc
        a = np.random.uniform(0.0, 1.0, 3)
        aperp = a - a.dot(b)
        aperp_mag = np.sqrt(aperp[0]**2 + aperp[1]**2 + aperp[2]**2)
        bperp_hat = aperp/aperp_mag
        rl = rl_mag*bperp_hat
        x = X + rl
        vperp_hat = np.cross(b, bperp_hat)
        v = vpar_mag*b + vperp_mag*vperp_hat

        self.r[0:3] = x
        self.r[3:6] = v
        self.r[6] = self.r[6]
        self.mode == 0
    #end def transform_GC_to_6D

    def push_GC(self,dt):
        #Assuming time-independence of rdot
        r0 = self.r
        k1 = dt*self.eom_GC(r0)
        k2 = dt*self.eom_GC(r0 + k1/2.)
        k3 = dt*self.eom_GC(r0 + k2/2.)
        k4 = dt*self.eom_GC(r0 + k3)
        self.r += (k1 + 2.*k2 + 2.*k3 + k4)/6.
        self.r[6] += dt
    #end def push_GC

    def eom_GC(self,r):
        B2 = self.B[0]**2 + self.B[1]**2 + self.B[2]**2

        b0 = self.B[0]/np.sqrt(B2)
        b1 = self.B[1]/np.sqrt(B2)
        b2 = self.B[2]/np.sqrt(B2)

        wc = abs(self.q)*np.sqrt(B2)/self.m
        rho = r[3]/wc

        rdot = np.empty(7)

        rdot[0] = (self.E[1]*self.B[2] - self.E[2]*self.B[1])/B2 + r[3]*b0
        rdot[1] = (self.E[2]*self.B[0] - self.E[0]*self.B[2])/B2 + r[3]*b1
        rdot[2] = (self.E[0]*self.B[1] - self.E[1]*self.B[0])/B2 + r[3]*b2
        rdot[3] = (self.E[0]*r[0] + self.E[1]*r[1] + self.E[2]*r[2])/np.sqrt(B2)/rho
        rdot[4] = 0.
        rdot[5] = 0.
        rdot[6] = 0.

        return rdot
    #end def eom_GC

    def apply_BCs_periodic(self, grid):
        self.r[0] = self.r[0]%(grid.length + grid.dx)
    #end def apply_BCs

    def apply_BCs_dirichlet(self, grid):
        if self.r[0] <= 0.0 or self.r[0] >= grid.length:
            self.r[0] = np.random.uniform(0.0, grid.length)
            self.r[1:3] = 0.0
            self.r[3:6] = np.random.normal(0.0, self.vth, 3)
        #end if
    #end def apply_BCs_dirichlet
#end class Particle

class Grid:
    def __init__(self, ng, length, Te):
        self.ng = ng
        self.length = length
        self.domain = np.linspace(0.0, length, ng)
        self.dx = self.domain[1] - self.domain[0]
        print(self.dx*self.ng)
        print(length)
        self.rho = np.zeros(ng)
        self.phi = np.zeros(ng)
        self.E = np.zeros(ng)
        self.n = np.zeros(ng)
        self.n0 = 0.0
        self.rho0 = 0.0
        self.Te = Te
        self.fill_laplacian_dirichlet()
    #end def __init__

    def __repr__(self):
        return f"Grid({self.ng}, {self.length}, {self.Te})"
    #end def __repr__

    def __len__(self):
        return int(self.ng)
    #end def __len__

    def weight_particles_to_grid(self, particles):
        self.rho[:] = 0.0
        self.n[:] = 0.0

        for particle in particles:
            index_l = int(np.floor(particle.r[0]/self.dx))
            index_r = (index_l + 1)
            w_r = (particle.r[0]%self.dx)/self.dx
            w_l = 1.0 - w_r

            self.rho[index_l] += particle.q*particle.p2c/self.dx*w_l
            self.rho[index_r] += particle.q*particle.p2c/self.dx*w_r
            self.n[index_l] += particle.p2c/self.dx*w_l
            self.n[index_r] += particle.p2c/self.dx*w_r
        #end for

        self.n0 = 5e16
        self.rho0 = e*5e16

        self.ne = self.n0*np.exp(e*self.phi/kb/self.Te)

    #end def weight_particles_to_grid

    def differentiate_phi_to_E_dirichlet(self):
        for i in range(1,self.ng-1):
            self.E[i] = -(self.phi[i + 1] - self.phi[i - 1])/self.dx/2.
        #end for
        self.E[0]  = -(self.phi[1]  - self.phi[0])/self.dx
        self.E[-1] = -(self.phi[-1] - self.phi[-2])/self.dx
    #end def differentiate_phi_to_E

    def fill_laplacian_dirichlet(self):
        ng = self.ng

        self.A = np.zeros((ng,ng))

        for i in range(1,ng-1):
            self.A[i,i-1] = 1.0
            self.A[i,i]   = -2.0
            self.A[i,i+1] = 1.0
        #end for

        self.A[0, 0]  = 1.
        self.A[-1,-1] = 1.
    #end def fill_laplacian_dirichlet

    def solve_for_phi_dirichlet_boltzmann(self):
        residual = 1.0
        tolerance = 1e-6
        iter_max = 100
        iter = 0

        phi = np.ones(self.ng)
        D = np.zeros((self.ng, self.ng))

        dx2 = self.dx*self.dx
        c0 = 0.9*self.rho[self.ng//2]/epsilon0
        c1 = e/(kb*self.Te)
        c2 = self.rho/epsilon0

        while (residual > tolerance) and (iter < iter_max):
            phi[0] = 0.
            phi[-1] = 0.

            F = np.dot(self.A,phi) - dx2*c0*np.exp(c1*phi) + dx2*c2
            F[0] = phi[0]
            F[-1] = phi[-1]

            D = spp.diags(-dx2*c0*c1*np.exp(c1*phi))

            J = self.A + D
            J = spp.csc_matrix(J)
            dphi = sppla.spsolve(J, F)

            phi = phi - dphi
            residual = la.norm(dphi)
            iter += 1

        #end while
        self.phi = phi - np.min(phi)
        print(residual)
    #end def solve_for_phi_dirichlet
#end class Grid

def main():
    density = 5e16
    N = 50000
    T = 400
    ng = 150
    dt_small = 3e-9
    dt_large = 1e-5
    Ti = 1.0*11600.
    Te = 10.0*11600.
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 0.01
    p2c = L*density/N
    B0 = np.array([0.0, 0.0, 0.0])
    E0 = np.array([0.0, 0.0, 0.0])

    fig1 = plt.figure(1)
    plt.ion()
    fig2 = plt.figure(2)
    plt.ion()

    grid = Grid(ng, L, Te)
    particles = [Particle(1.0*mp, e, p2c, Ti, B=B0, E=E0, grid=grid) for i in range(N)]

    file = open("particle_output.txt","w+")
    positions_x = np.zeros(N)
    positions_y = np.zeros(N)
    positions_z = np.zeros(N)
    test_particle = np.zeros((3, T))
    velocities = np.zeros(N)
    colors = np.zeros(N)

    time = 0.
    for time_index in range(T):
        print(time_index)
        time += dt_small
        grid.weight_particles_to_grid(particles)
        grid.solve_for_phi_dirichlet_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()
        for index,particle in enumerate(particles):
            positions_x[index] = particle.r[0]
            velocities[index] = particle.r[3]
            particle.interpolate_electric_field_dirichlet(grid)
            particle.push_6D(dt_small)
            particle.apply_BCs_dirichlet(grid)
        #end for
        print(np.max(positions_x), grid.length)
        plt.figure(1)
        plt.clf()
        plt.scatter(positions_x, velocities, s=1.0)
        plt.draw()
        plt.pause(0.001)
        plt.savefig('ps_'+str(time_index))

        plt.figure(2)
        plt.clf()
        #plt.plot(grid.domain, grid.n)
        #plt.plot(grid.domain, grid.ne)
        plt.plot(grid.domain, grid.phi)
        print(f"max phi: {np.max(grid.phi)}")
        #plt.plot(grid.domain, grid.rho/np.max(grid.rho))
        plt.draw()
        plt.pause(0.001)
        plt.savefig('phi_'+str(time_index))
    #end for
    plt.show()

    #c.convert('.','ps',0,T,1,'out.gif')
    #c.convert('.','phi',0,T,1,'out.gif')

#end def main

if __name__ == '__main__':
    main()
