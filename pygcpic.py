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
    def __init__(self, m, q, p2c, T, B=[0.,0.,0.], grid=None):
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
        self.E = 0.0
        self.B = B
        #Electric field at particle position
        if grid != None: self.initialize_6D(grid)
    #end def __init__

    def __repr__(self):
        return f"Particle({self.m}, {self.q}, {self.p2c}, {self.T})" #.format(**locals())
    #end def

    def get_speed(self):
        return np.sqrt(self.r[3]**2 + self.r[4]**2 + self.r[5]**2)
    #end def get_speed

    def initialize_6D(self, grid):
        self.r[0:3] = np.random.uniform(0.0, grid.length)
        self.r[1:3] = 0.0
        self.r[3:6] = np.random.normal(0.0, self.vth , 3)
        self.r[6] = 0.0
    #end def initialize_6D

    def interpolate_electric_field(self, grid):
        index = int(np.floor(self.r[0]/grid.dx))
        w_r = (self.r[0] % grid.dx) / grid.dx
        w_l = 1.0 - w_r
        self.E = w_l*grid.E[index] + w_r*grid.E[(index+1)]
    #end def interpolate_electric_field

    def push_6D(self,dt):
        constant = 0.5*dt*self.q/self.m

        self.r[3] += constant*self.E
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

        self.r[3] += constant*self.E
        self.r[4] += 0.
        self.r[5] += 0.

        self.r[0] += self.r[3]*dt
        self.r[1] += self.r[4]*dt
        self.r[2] += self.r[5]*dt

        self.r[6] += dt
    #end push_6D

    def apply_BCs_periodic(self, grid):
        self.r[0] = self.r[0]%(grid.length + grid.dx)
    #end def apply_BCs

    def apply_BCs_dirichlet(self, grid):
        if self.r[0] <= 0.0 or self.r[0] > grid.length:
            self.r[0] = np.random.uniform(grid.length/4.0, 3.0*grid.length/4.0)
            self.r[1:3] = 0.0
            self.r[3:6] = np.random.normal(0.0, self.vth / np.sqrt(2), 3)
        #end if
    #end def apply_BCs_dirichlet

#end class Particle

class Grid:
    def __init__(self, ng, length, Te):
        self.ng = ng
        self.length = length
        self.domain = np.linspace(0.0, length, ng)
        self.dx = self.domain[1] - self.domain[0]
        self.rho = np.zeros(ng)
        self.phi = np.zeros(ng)
        self.E = np.zeros(ng)
        self.n = np.zeros(ng)
        self.n0 = 0.0
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
            index_r = (index_l + 1)%self.ng
            w_r = (particle.r[0]%self.dx)/self.dx
            w_l = 1.0 - w_r
            rho_i = particle.q*particle.p2c/self.dx
            n_i   = particle.p2c/self.dx
            rho_l = rho_i*w_l
            rho_r = rho_i*w_r
            n_l = n_i*w_l
            n_r = n_i*w_r
            self.rho[int(index_l)] += rho_l
            self.rho[int(index_r)] += rho_r
            self.n[int(index_l)] += n_l
            self.n[int(index_r)] += n_r
        #end for

        if(self.n0==0.0):
            self.n0 = self.n[self.ng//2]
        #end if

    #end def weight_particles_to_grid

    def differentiate_phi_to_E_periodic(self):
        idx_2 = 0.5/self.dx
        for i in range(self.ng):
            ind_l = i - 1
            ind_r = (i + 1)%self.ng
            self.E[i] = -(self.phi[ind_r] - self.phi[ind_l])*idx_2
        #end for
    #end def differentiate_phi_to_E

    def differentiate_phi_to_E_dirichlet(self):
        idx_2 = 0.5/self.dx
        for i in range(1,self.ng-1):
            ind_l = i - 1
            ind_r = i + 1
            self.E[i] = -(self.phi[ind_r] - self.phi[ind_l])*idx_2
        #end for
        self.E[0]  = -(self.phi[1]  - self.phi[0])/self.dx
        self.E[-1] = -(self.phi[-1] - self.phi[-2])/self.dx
    #end def differentiate_phi_to_E

    def fill_laplacian_periodic(self):
        ng = self.ng
        self.A = sp.diag(np.ones(ng-1),-1) + sp.diag(-2.*np.ones(ng),0) + sp.diag(np.ones(ng-1),1)
        self.A[0, 0]  = -2.
        self.A[0, 1]  =  1.
        self.A[0,-1]  =  1.

        self.A[-1,-1] = -2.
        self.A[-1,-2] =  1.
        self.A[-1, 0] =  1.
        A = spp.csc_matrix(self.A)
    #end def fill_laplacian_periodic

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

        A = spp.csc_matrix(self.A)
    #end def fill_laplacian_dirichlet

    def solve_for_phi_periodic(self):
        phi = np.zeros(self.ng)
        dx2 = self.dx*self.dx
        c0 = -np.average(self.rho) / epsilon0
        c2 = self.rho/epsilon0

        self.phi = sppla.spsolve(self.A, -dx2*c0 - dx2*c2)
    #end def solve_for_phi_periodic

    def solve_for_phi_dirichlet_boltzmann(self):
        residual = 1.0
        tolerance = 1e-3
        iter_max = 10
        iter = 0

        phi = self.phi
        dx2 = self.dx*self.dx
        c0 = self.rho[self.ng//2]/epsilon0
        c1 = e/kb/self.Te
        c2 = self.rho/epsilon0
        D = np.zeros((self.ng, self.ng))

        while (residual > tolerance) & (iter < iter_max):
            F = self.A.dot(phi) - dx2*c0*np.exp(c1*(phi-phi[self.ng//2])) + dx2*c2
            F[0] = phi[0]
            F[-1] = phi[-1]

            np.fill_diagonal(D, -dx2*c0*c1*np.exp(c1*(phi-phi[self.ng//2]) ))
            D[0,0] = -dx2*c0*c1
            D[-1,-1] = -dx2*c0*c1

            J = spp.csc_matrix(self.A + D)
            dphi = sppla.inv(J).dot(F)

            phi = phi - dphi
            residual = la.norm(dphi)
            iter += 1
        #end while
        self.phi = phi - np.min(phi)
        print(residual)
    #end def solve_for_phi_dirichlet
#end class Grid

def main():
    density = 1e10
    N = 10000
    T = 200
    ng = 200
    dt_small = 1e-6
    dt_large = 1e-5
    Ti = 0.01*11600.
    Te = 1.0*11600.
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 100.0*LD
    p2c = L*density/N
    B = [1.0, 0.0, 0.0]

    #fig1 = plt.figure(1)
    #plt.ion()
    fig2 = plt.figure(2)
    plt.ion()

    grid = Grid(ng, L, Te)
    particles = [Particle(mp, e, p2c, Ti, B=B, grid=grid) for i in range(N)]

    file = open("particle_output.txt","w+")
    positions_x = np.zeros(N)
    positions_y = np.zeros(N)
    positions_z = np.zeros(N)
    test_particle = np.zeros((3, T))
    velocities = np.zeros(N)
    colors = np.zeros(N)

    time = 0.0
    for time_index in range(T):
        print(time_index)
        time += dt_small
        grid.weight_particles_to_grid(particles)
        grid.solve_for_phi_dirichlet_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()

        for particle_index, particle in enumerate(particles):

            positions_x[particle_index] = particle.r[0]
            positions_y[particle_index] = particle.r[1]
            positions_z[particle_index] = particle.r[2]

            if particle_index%100==0:
                print(f'{particle.r[0]}, {particle.r[1]}, {particle.r[2]}', file=file)
            #end if

            velocities[particle_index] = particle.r[3]/particle.vth
            colors[particle_index] = particle.get_speed()

            particle.interpolate_electric_field(grid)

            #if (particle.r[0] <= 20*LD or particle.r[0] >= 80*LD) and particle.r[6]<=time:
            #    particle.push_6D(dt_small)
            #end if
            #if (particle.r[0] > 20*LD and particle.r[0] < 80*LD) and particle.r[6]<=time:
            #    particle.push_6D(dt_large)
            #end if

            particle.push_6D(dt_small)

            particle.apply_BCs_dirichlet(grid)
        #end for

        #plt.figure(1)
        #plt.clf()
        #plt.scatter(positions_y, positions_z, s=1.0, c=colors)
        #plt.draw()
        #plt.pause(0.001)
        #plt.savefig('ps_'+str(time_index))

        plt.figure(2)
        plt.clf()
        plt.plot(grid.domain, grid.phi)
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
