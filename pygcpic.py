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

class Particles:
    def __init__(self, N, m, q, p2c, T, B0, E0, grid):
        self.r = np.empty((N,7))
        self.q = q
        self.m = m
        self.T = T
        self.p2c = p2c
        self.vth = np.sqrt(2.0*kb*self.T/self.m)
        self.mode = 0
        self.E = E0
        self.B = B0
        self.color = 0
    #end def __init__
#end class Particles

class Particle:
    def __init__(self, m, q, p2c, T, B0=np.zeros(3), E0=np.zeros(3), grid=None):
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
        self.E = E0
        self.B = B0
        #Electric field at particle position
        self.active = 1
        self.color = 0
        if grid != None: self._initialize_6D(grid)
    #end def __init__

    def __repr__(self):
        return f"Particle({self.m}, {self.q}, {self.p2c}, {self.T})"
    #end def

    def get_speed(self):
        return np.sqrt(self.r[3]**2 + self.r[4]**2 + self.r[5]**2)
    #end def get_speed

    def get_kinetic_energy(self):
        return 0.5*self.m*self.get_speed()**2
    #end def

    def _initialize_6D(self, grid):
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
        self.r[4] += constant*self.E[1]
        self.r[5] += constant*self.E[2]

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
        self.r[4] += constant*self.E[1]
        self.r[5] += constant*self.E[2]

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
        self.mode = 1
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
        self.mode = 0
    #end def transform_GC_to_6D

    def push_GC(self,dt):
        #Assuming time-independence of rdot
        r0 = self.r
        k1 = dt*self._eom_GC(r0)
        k2 = dt*self._eom_GC(r0 + k1/2.)
        k3 = dt*self._eom_GC(r0 + k2/2.)
        k4 = dt*self._eom_GC(r0 + k3)
        self.r += (k1 + 2.*k2 + 2.*k3 + k4)/6.
        self.r[6] += dt
    #end def push_GC

    def _eom_GC(self,r):
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
            self.active = 0
            output_r = self.r
            return output_r
        else:
            return None
        #end if
    #end def apply_BCs_dirichlet

    def reactivate(self, distribution, grid, time):
        self.r = next(distribution)
        self.r[6] = time
        grid.added_particles += self.p2c
        self.active = 1
    #end def reactivate
#end class Particle

def thompson_inverse_cdf(SBV, r):
    return -SBV*np.sqrt(r)/(np.sqrt(r) - 1.)
#end def thompson_inverse_cdf

def thompson_distribution(SBV):
    while True:
        r = np.random.uniform(0., 1.)
        yield -SBV*np.sqrt(r)/(np.sqrt(r) - 1.)
    #end while
#end def thompson_distribution

def thompson_distribution_isotropic_6D(SBV, mass, grid, mfp):
    while True:
        r = np.empty(7)
        wall = np.random.choice([1, -1])
        if wall == -1:
            r[0] = grid.length - mfp
        elif wall == 1:
            r[0] = mfp
        #end if
        r[1:3] = 0.0
        rand = np.random.uniform(0., 1.)
        alpha, beta, gamma = np.random.uniform(0., np.pi/2., 3)
        energy = -SBV*np.sqrt(rand)/(np.sqrt(rand) - 1.)*e*np.exp(-rand) #Don't forget to convert...
        velocity = np.sqrt(2.0 * energy / mass)
        u = np.array([np.cos(alpha), np.cos(beta), np.cos(gamma)])
        u[0] = wall*abs(u[0])
        u /= np.linalg.norm(u)
        v = u*velocity
        r[3:6] = v
        yield r
    #end while
#end def thompson_distribution_isotropic_6D

def source_distribution_6D(grid, Ti, mass):
    while True:
        vth = np.sqrt(2.0*kb*Ti/mass)
        r = np.empty(7)
        r[0] = np.random.uniform(0.0, grid.length)
        r[1:3] = 0.
        r[3:6] = np.random.normal(0.0, vth, 3)
        yield r
    #end while
#end def source_distribution_6D

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
        self.rho0 = 0.0
        self.Te = Te
        self.ve = np.sqrt(8./np.pi*kb*self.Te/me)
        self.added_particles = 0
        self._fill_laplacian_dirichlet()
    #end def __init__

    def __repr__(self):
        return f"Grid({self.ng}, {self.length}, {self.Te})"
    #end def __repr__

    def __len__(self):
        return int(self.ng)
    #end def __len__

    def copy(self):
        return Grid(self.ng, self.length, self.Te)
    #end def copy

    def weight_particles_to_grid_boltzmann(self, particles,dt):
        self.rho[:] = 0.0
        self.n[:] = 0.0

        for particle in particles:
            if particle.active == 1:
                index_l = int(np.floor(particle.r[0]/self.dx))
                index_r = (index_l + 1)
                w_r = (particle.r[0]%self.dx)/self.dx
                w_l = 1.0 - w_r

                self.rho[index_l] += particle.q*particle.p2c/self.dx*w_l
                self.rho[index_r] += particle.q*particle.p2c/self.dx*w_r
                self.n[index_l] += particle.p2c/self.dx*w_l
                self.n[index_r] += particle.p2c/self.dx*w_r
            #end if
        #end for

        if self.n0 == 0.: #This is only true for the first timestep.
            #TODO(JTD) Find a less sloppy way to initialize this.
            eta = np.exp(self.phi/self.Te/11600.)
            self.p_old = np.trapz(eta, self.domain)
            self.n0 = 0.9*np.average(self.n)
            self.rho0 = e*self.n0
        else:
            eta = np.exp(self.phi/self.Te/11600.)
            p_new = np.trapz(eta, self.domain)
            q_new = eta[0] + eta[-1]
            #TODO(JTD) this line below is sloppy. bShould grid store p2c?
            #Resolved - each particle increments added_particles by p2c
            r_new = 2.*self.added_particles/dt
            #print(f"p: {p_new} q: {q_new} r: {r_new}")
            fn = np.sqrt(self.ve*q_new*dt/p_new)
            self.n0 = self.n0*( (1.0 - fn)*self.p_old/p_new + fn - fn*fn/4.) + r_new*dt/p_new
            self.rho0 = self.n0*e
            self.p_old = p_new
        #end if
    #end def weight_particles_to_grid

    def differentiate_phi_to_E_dirichlet(self):
        for i in range(1,self.ng-1):
            self.E[i] = -(self.phi[i + 1] - self.phi[i - 1])/self.dx/2.
        #end for
        self.E[0]  = -(self.phi[1]  - self.phi[0])/self.dx
        self.E[-1] = -(self.phi[-1] - self.phi[-2])/self.dx
    #end def differentiate_phi_to_E

    def _fill_laplacian_dirichlet(self):
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
        tolerance = 1e-3
        iter_max = 100
        iter = 0

        phi = self.phi
        D = np.zeros((self.ng, self.ng))

        dx2 = self.dx*self.dx
        c0 = e*self.n0/epsilon0
        c1 = e/kb/self.Te
        c2 = e*self.n/epsilon0

        while (residual > tolerance) and (iter < iter_max):
            F = np.dot(self.A,phi) - dx2*c0*np.exp(c1*(phi)) + dx2*c2
            F[0] = phi[0]
            F[-1] = phi[-1]

            np.fill_diagonal(D, -dx2*c0*c1*np.exp(c1*(phi)))
            D[0,0] = -dx2*c0*c1
            D[-1,-1] = -dx2*c0*c1

            J = self.A + D
            J = spp.csc_matrix(J)
            dphi = sppla.inv(J).dot(F)

            phi = phi - dphi
            residual = la.norm(dphi)
            iter += 1
        #end while
        self.phi = phi - np.min(phi)
    #end def solve_for_phi_dirichlet

    def reset_added_particles(self):
        self.added_particles = 0
    #end def reset_added_particles
#end class Grid

def main():
    density = 5e16
    N = 50000
    T = 200
    ng = 300
    dt_small = 1e-9
    dt_large = 1e-8
    Ti = 0.5*11600.
    Te = 2.0*11600.
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 200*LD
    p2c = L*density/N
    B0 = np.array([0.01, 0.0, 0.0])
    E0 = np.array([0.0, 0.0, 0.0])

    phi_floating = (Te/11600)*0.5*np.log(1.0*mp/2./np.pi/me/(1.+Ti/Te))
    print(f"Floating potential: {phi_floating} V")

    fig1 = plt.figure(1)
    plt.ion()
    fig2 = plt.figure(2)
    plt.ion()
    fig3 = plt.figure(3)
    plt.ion()

    np.random.seed(1)
    grid = Grid(ng, L, Te)
    particles = [Particle(1.0*mp, e, p2c, Ti, B0=B0, E0=E0, grid=grid) for i in range(N)]

    file = open("particle_output.txt","w+")
    positions_x = np.zeros(N)
    positions_y = np.zeros(N)
    positions_z = np.zeros(N)
    velocities = np.zeros(N)
    colors = np.zeros(N)
    E_t = np.zeros(T+1)
    phi_max_t = np.zeros(T+1)

    sputtered_distribution = thompson_distribution_isotropic_6D(6.0, mp, grid, grid.dx)
    source_distribution = source_distribution_6D(grid, Ti, mp)
    sputtering_threshold = 7.0*e

    time = 0.
    for time_index in range(T+1):
        print(f"timestep: {time_index}")
        time += dt_small
        grid.weight_particles_to_grid_boltzmann(particles, dt_small)
        print(f"n0: {grid.n0}\nadded_particles: {grid.added_particles}")
        grid.reset_added_particles()

        grid.solve_for_phi_dirichlet_boltzmann()
        print(f"phi_max: {np.max(grid.phi)}")

        grid.differentiate_phi_to_E_dirichlet()
        E_t[time_index] = np.dot(grid.E, grid.E)
        phi_max_t[time_index] = np.max(grid.phi)

        for index, particle in enumerate(particles):
            if particle.active == 0:
                if particle.get_kinetic_energy() < sputtering_threshold:
                    particle.reactivate(source_distribution, grid, time)
                    particle.color = 0
                else:
                    particle.reactivate(sputtered_distribution, grid, time)
                    particle.color = 1
                #end if
            #end if

            if particle.active == 1:
                #For plotting
                if particle.r[0] > grid.length/2.:
                    positions_x[index] = particle.r[0]
                    velocities[index] = -particle.r[3]
                else:
                    positions_x[index] = grid.length - particle.r[0]
                    velocities[index] = particle.r[3]


                colors[index] = particle.color

                particle.interpolate_electric_field_dirichlet(grid)
                particle.push_6D(dt_small)
                particle.apply_BCs_dirichlet(grid)
            #end if
        #end for

        if time_index%10 ==0:
            plt.figure(1)
            plt.clf()
            plt.scatter(positions_x, velocities, s=1.0, c=colors, cmap='jet')
            plt.axis((grid.length/2., grid.length, -8.0*particles[0].vth, 8.0*particles[0].vth))
            plt.draw()
            plt.pause(0.01)
            plt.savefig('ps_'+str(time_index))

            plt.figure(2)
            plt.clf()
            plt.plot(grid.domain, grid.phi)
            plt.draw()
            plt.pause(0.01)
            plt.savefig('phi_'+str(time_index))

            plt.figure(3)
            plt.clf()
            plt.plot(np.linspace(0.0,time,time_index), phi_max_t[:time_index])
            plt.plot(np.linspace(0.0,time,time_index), phi_floating*np.ones(time_index))
            plt.draw()
            plt.pause(0.01)
            plt.savefig('phi_t_'+str(time_index))
        #end if
    #end for
    plt.show()

    c.convert('.','ps',0,T,10,'out_ps.gif')
    c.convert('.','phi',0,T,10,'out_phi.gif')
    c.convert('.','phi_t',0,T,10,'out_phi_t.gif')

#end def main

if __name__ == '__main__':
    main()
