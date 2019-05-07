import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import convert as c
import itertools
#import generate_ftridyn_input as gen

epsilon0 = 8.854e-12
e = 1.602e-19
mp = 1.67e-27
me = 9.11e-31
kb = 1.38e-23

def particle_from_energy_angle_coordinates(energy, ca, cb, cg, m, Z,
    B=np.array([0.0, 0.0, 0.0]), q=e, p2c=0, T=0., grid=None, x0=0., time=0.):
    """
    This function creates and initializes a Particle object using energy-angle
    coordintes (e.g., those from F-TRIDYN output).

    Args:
        energy (float): particle kinetic energy
        ca (float): directional cosine along x-axis, range 0. to 1.
        cb (float): directional cosine along y-axis, range 0. to 1.
        cg (float): directional cosine along z-axis, range 0. to 1.
        m (float): particle mass in kg
        Z (int): particle atomic number
        B (:obj: 'np.ndarray'), optional: magnetic field (assumed zero)
        q (float), optional: particle charge (assumed 1 fundamental charge)
        p2c (int), optional: assumed zero (i.e., chargeless tracer)
        T (float), optional: species temperature (assumed zero)
        grid (:obj: 'Grid'), optional: grid associated with particle, assumed
            None
        x0 (float), optional: starting position along x-axis (assumed zero)
        time (float), optional: particle's current time (assumed zero)
    """
    T = 0.
    speed = np.sqrt(2.*energy*e/(m*mp))
    u = [ca, cb, cg]
    u /= np.linalg.norm(u)
    v = speed * u
    particle = Particle(m*mp, q, p2c, T, Z, grid=grid)
    particle.set_v(v)
    particle.set_x(x0)
    particle.set_time(time)
    particle.set_B(B)
    return particle
#end def particle_from_energy_angle_coordinates

class Particle:
    """
        Generic particle object. Can work in 6D or GC coordinate systems and
        can transform between the two representations on the fly. Includes
        methods for changing the particle's properties and state, and for
        advancing the particle forward in time in either coordinate system.
    """
    def __init__(self, m, q, p2c, T, Z, B0=np.zeros(3), E0=np.zeros(3),
        grid=None):
        """
        Particle initialization.

        Args:
            m (float): mass in kg
            q (float): charge in C
            p2c (float): number of physical particles represented by this
                particle. Should be > 1 except for tracers when p2c = 0.
            T (float): species temperature in K
            Z (int): species atomic number
            B0 (:obj: 'np.ndarray'): magnetic field vector (assumed zero)
            E0 (:obj: 'np.ndarray'): electric field vector (assumed zero)
            grid (:obj: 'Grid'), optional: grid object associated with this
                particle (assumed None)
        """
        self.r = np.zeros(7)
        self.q = q
        self.Z = Z
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
        if grid != None: self._initialize_6D(grid)
    #end def __init__

    def __repr__(self):
        return f"Particle({self.m}, {self.q}, {self.p2c}, {self.T}, {self.Z})"
    #end def

    def get_Z(self):
        """
        Returns the partice's atomic number.

        Returns:
            Z (int): atomic number
        """
        return int(self.Z)
    #end def get_Z

    def set_B(self, B):
        """
        Sets the particle's 3D magnetic field vector.

        Args:
            B (:obj: 'np.ndarray'): 3D B-field vector
        """
        self.B[:] = B
    #end def set_B

    def set_v(self, v):
        """
        Sets the particle's 3D velocity vector.

        Args:
            v (:obj: 'np.ndarray'): 3D velocity vector
        """
        self.r[3:6] = v
    #end def set_v

    def set_x(self, x):
        """
        Sets the particle's x-coordinate.

        Args:
            x (float): x-coordinate of particle
        """
        self.r[0] = x
    #end def set_x

    def get_x(self):
        """
        Returns the particle's x-coordinate.

        Returns:
            x (float): x coordinate
        """
        return self.r[0]
    #end def get_x

    def get_v_x(self):
        """
        Returns the particle's velocity in the x-directions.

        Returns:
            v_x (float): x velocity
        """
        return self.r[3]
    #end def get_v_x

    def set_time(self, time):
        """
        Sets the particle's current time.
        """
        self.r[6] = time
    #end def set_time

    def is_active(self):
        """
        Returns a boolean that is true if the particle is active and false
        if the particle is inactive.

        Returns:
            is_active (bool): whether the particle is active or not
        """
        return self.active == 1
    #end def is_active

    def get_speed(self):
        """
        Returns the particle's speed.

        Tests:

        >>> particle=Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> particle.r[3]=1.0
        >>> particle.r[4:6]=2.0
        >>> particle.get_speed()
        3.0
        """
        return np.sqrt(self.r[3]**2 + self.r[4]**2 + self.r[5]**2)
    #end def get_speed

    def get_angle_wrt_wall(self, use_degrees=True):
        """
        Returns the particle's angle with respect to the normal of the y-x
        plane in degrees. Default return value is in degrees for F-Tridyn
        input.

        Args:
            use_degrees (bool), optional: Whether to use degrees (as opposed
            to radians) for the return value.

        Returns:
            alpha (float): angle w.r.t. y-x plane wall.

        Tests:

        >>> np.random.seed(1)
        >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> particle.r[3] = np.random.uniform(0.0, 1.0)
        >>> particle.get_angle_wrt_wall(use_degrees=True)
        0.0
        >>> particle.get_angle_wrt_wall(use_degrees=False)
        0.0
        """
        v = self.r[3:6]
        u = v / np.linalg.norm(v)
        ca = abs(u[0])
        alpha = np.arccos(ca)
        if use_degrees:
            return alpha*180./np.pi
        else:
            return alpha
        #end if
    #end def get_angle_wrt_wall

    def get_kinetic_energy(self):
        """
        Returns the particle's kinetic energy.

        Tests:

        >>> particle=Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> particle.r[3] = 1.0
        >>> particle.r[4:6] = 2.0
        >>> particle.get_kinetic_energy()
        4.5
        """
        return 0.5*self.m*self.get_speed()**2
    #end def

    def _initialize_6D(self, grid):
        """
        Given a grid object, initialize the particle on the grid with a
        uniform distribution in space and a normal distribution of speeds
        based on its thermal velocity.

        Args:
            grid (:obj: 'Grid'): the grid with which the particle is
                associated

        Tests:

        >>> np.random.seed(1)
        >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> grid = Grid(100, 1.0, 1.0)
        >>> particle._initialize_6D(grid)
        >>> np.random.seed(1)
        >>> particle.r[0] == np.random.uniform(0.0, grid.get_length())
        True
        >>> particle.r[3] == np.random.normal(0.0, particle.vth, 3)[0]
        True
        """
        self.r[0] = np.random.uniform(0.0, grid.get_length())
        self.r[1:3] = 0.0
        self.r[3:6] = np.random.normal(0.0, self.vth , 3)
        self.r[3] = self.r[3]
        self.r[6] = 0.0
    #end def initialize_6D

    def shift_x_velocity_distribution(self, u):
        """
        Shift's the particle's velocity by the mean velocity u.

        Args:
            u (float): mean of velocity distribution (i.e., flow speed)
        """

        self.r[3] += u
    #end def shift_x_velocity_distribution

    def set_x_direction(self, direction):
        """
        Set the direction of the particle by taking the absolute value of its
        x-velocity and, if necessary, negating it.

        Args:
            direction (:obj: 'str'): 'left' or 'right'
        """
        if direction.lower() == 'left':
            self.r[3] = -abs(self.r[3])
        elif direction.lower() == 'right':
            self.r[3] = abs(self.r[3])
        elif type(direction) == type(''):
            raise ValueError('particle.set_x_direction() received neither right nor left')
        else:
            raise TypeError('particle.set_x_direction(direction) received a non-string type for direction')
        #end if
    #end def set_x_direction

    def interpolate_electric_field_dirichlet(self, grid):
        """
        Interpolates electric field values from grid to particle position
        assuming Dirichlet-Dirichlet boundary conditions.

        Args:
            grid (:obj: 'Grid'): the grid with which the particle is
                associated

        Tests:

        >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> grid = Grid(100, 1.0, 1.0)
        >>> particle._initialize_6D(grid)
        >>> grid.E[:] = 1.0
        >>> particle.interpolate_electric_field_dirichlet(grid)
        >>> particle.E[0]
        1.0
        """
        ind = int(np.floor(self.r[0]/grid.dx))
        w_l = (self.r[0]%grid.dx)/grid.dx
        w_r = 1.0 - w_l
        self.E[0] = grid.E[ind]*w_l + grid.E[ind+1]*w_r
    #end def interpolate_electric_field

    def push_6D(self,dt):
        """
        Boris-Buneman integrator that pushes the particle in 6D cooordinates
        one timeste of magnitude dt.

        Args:
            dt (float): timestep
        """
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
        """
        Transform the particle state vector from 6D to guiding-center
        coordinates. This process results in the loss of one coordinate
        which represents the phase of the particle.
        """
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
        """
        Transform the particle state vector from guiding-center to 6D
        coordinates. This method uses a single random number to generate the
        missing phase information from the GC coordinates.
        """
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
        """
        Push the particle using the guiding-center cooordinates one timestep
        of magnitude dt.

        Args:
            dt (float): timestep
        """
        #Assuming direct time-independence of rdot
        r0 = self.r
        k1 = dt*self._eom_GC(r0)
        k2 = dt*self._eom_GC(r0 + k1/2.)
        k3 = dt*self._eom_GC(r0 + k2/2.)
        k4 = dt*self._eom_GC(r0 + k3)
        self.r += (k1 + 2.*k2 + 2.*k3 + k4)/6.
        self.r[6] += dt
    #end def push_GC

    def _eom_GC(self,r):
        """
        An internal method that calculates the differential of the r-vector
        for the equation of motion given to the RK4 guiding-center solver.

        Args:
            r (:obj: 'np.ndarray'): particle state vector in GC coordinates
        """
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
        rdot[3] = (self.E[0]*r[0] + self.E[1]*r[1] + self.E[2]*r[2])\
            /np.sqrt(B2)/rho
        rdot[4] = 0.
        rdot[5] = 0.
        rdot[6] = 0.

        return rdot
    #end def eom_GC

    def apply_BCs_periodic(self, grid):
        """
        Wrap particle x-coordinate around for periodic BCs.
        """
        self.r[0] = self.r[0]%(grid.get_length())
    #end def apply_BCs

    def apply_BCs_dirichlet(self, grid):
        """
        Set particle to inactive when it's x-coordinate exceeds either wall in a
        dirichlet-dirichlet boundary condition case.
        """
        if (self.r[0] <= 0.0) or (self.r[0] >= grid.get_length()):
            self.active = 0
        #end if
    #end def apply_BCs_dirichlet

    def reactivate(self, distribution, grid, time, p2c, m, q, Z):
        """
        Re-activate an inactive particle. This function pulls an r vector
        composed of x, y, z, v_x, v_y, v_z, t from a given distribution and
        applies it ot the particle before reactivating it. Additionally, the
        mass, charge, and p2c ratio can be reset at the instant of
        reactivation.

        Args:
            distribution (:obj: 'generator'): a generator that returns a
                6-vector that overwrites the particle's coordinates
            grid (:obj: 'Grid'): the grid object with which the particle is
                associated
            time (float): the particle's current time
            p2c (float): the ratio of computational to real particles
            m (float): particle mass
            q (float): particle charge
            Z (float): particle atomic number
        """
        self.r = next(distribution)
        self.p2c = p2c
        self.m = m
        self.q = q
        self.Z = Z
        self.r[6] = time
        self.active = 1
        grid.add_particles(p2c)
    #end def reactivate
#end class Particle

def thompson_distribution_isotropic_6D(SBV, mass, grid, mfp):
    while True:
        r = np.empty(7)
        wall = np.random.choice([1, -1])
        if wall == -1:
            r[0] = grid.get_length() - mfp
        elif wall == 1:
            r[0] = mfp
        #end if
        r[1:3] = 0.0
        rand = np.random.uniform(0., 1.)
        alpha, beta, gamma = np.random.uniform(0., np.pi/2., 3)
        energy = -SBV*np.sqrt(rand)/(np.sqrt(rand) - 1.)*e
        velocity = np.sqrt(2.*energy/mass)
        u = np.array([np.cos(alpha), np.cos(beta), np.cos(gamma)])
        u[0] = wall*abs(u[0])
        u /= np.linalg.norm(u)
        v = u*velocity
        r[3:6] = v
        yield r
    #end while
#end def thompson_distribution_isotropic_6D

def distribution_from_list(new_particle_list, grid, x0):
    index = 0
    while True:
        r = np.zeros(7)
        particle_state = new_particle_list[index]
        energy = particle_state[0]
        u = particle_state[1:4]
        if np.random.choice((True, False)):
            u[0] = abs(u[0])
            r[0] = x0
        else:
            u[0] = -abs(u[0])
            r[0] = grid.get_length() - x0
        #end if
        mass = particle_state[4]
        u /= np.linalg.norm(u)
        speed = np.sqrt(2.*energy*e/(mass*mp))
        v = speed * u
        r[3:6] = v
        yield r
        index = (index+1)%len(new_particle_list)
    #end while
#end def distribution_from_list

def distribution_from_file(filename, mass, grid, mfp):
    file = open(filename, 'r')
    #WRITE(IOUT7,259)FLUCA,LABEL(IREC1,2),ESP,XR(IREC1,2),YR(IREC1,2),ZR(IREC1,2),CXSP,CYSP,CZSP,SBE(LABEL(IREC1,2))
    lines = file.readlines()
    num_lines = len(lines)
    index = 0
    while True:
        line = lines[index%num_lines]
        r = np.empty(7)
        direction = np.random.choice((1, -1))
        if direction == 1:
            r[0] = 0.0 + mfp
        else:
            r[0] = grid.get_length() - mfp
        r[1:3] = 0.
        line = line.split()
        energy = float(line[2])*e #Convert to J
        velocity = np.sqrt(2.*energy/mass)
        u = np.array([float(line[6]), float(line[7]), float(line[8])])
        u[0] = direction*abs(u[0])
        u /= np.linalg.norm(u)
        v = u*velocity
        r[3:6] = v
        print(f'Sputtered Energy: {energy/e}')
        index += 1
        yield r
    #end while
#end def distribution_from_file

def source_distribution_6D(grid, Ti, mass):
    while True:
        vth = np.sqrt(2.0*kb*Ti/mass)
        r = np.empty(7)
        r[0] = np.random.uniform(0.0, grid.get_length())
        r[1:3] = 0.
        r[3:6] = np.random.normal(0.0, vth, 3)
        yield r
    #end while
#end def source_distribution_6D

def flux_source_distribution_6D(grid, Ti, mass, flux):
    while True:
        vth = np.sqrt(2.0*kb*Ti/mass)
        r = np.empty(7)
        r[0] = grid.length - np.random.uniform(0.0, 1.0)*grid.dx
        r[1:3] = 0.
        r[3:6] = np.random.normal(0.0, vth, 3)
        r[3] = r[3] + flux
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
        self.n0 = None
        self.rho0 = None
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

    def get_length(self):
        return self.length
    #end def

    def weight_particles_to_grid_boltzmann(self, particles,dt):
        self.rho[:] = 0.0
        self.n[:] = 0.0

        for particle_index, particle in enumerate(particles):
            if particle.is_active():
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

        if self.n0 == None: #This is only true for the first timestep.
            eta = np.exp(self.phi/self.Te/11600.)
            self.p_old = np.trapz(eta, self.domain)
            self.n0 = 0.9*np.average(self.n)
            self.rho0 = e*self.n0
        else:
            eta = np.exp(self.phi/self.Te/11600.)
            p_new = np.trapz(eta, self.domain)
            q_new = eta[0] + eta[-1]
            r_new = 2.*self.added_particles/dt
            #print(f"p: {p_new} q: {q_new} r: {r_new}")
            fn = np.sqrt(self.ve*q_new*dt/p_new)
            self.n0 = self.n0*( (1.0 - fn)*self.p_old/p_new + fn - fn*fn/4.) + \
                r_new*dt/p_new
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

    def _fill_laplacian_dirichlet_neumann(self):
        ng = self.ng

        self.A = np.zeros((ng, ng))

        for i in range(1,ng-1):
            self.A[i,i-1] = 1.0
            self.A[i,i]   = -2.0
            self.A[i,i+1] = 1.0
        #end for

        self.A[0,0] = 1.

        self.A[-1,-1] = 3.
        self.A[-1,-2] = -4.
        self.A[-1,-3] = 1.
    #end def

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

    def solve_for_phi_dirichlet_neumann_boltzmann(self):
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
            F[-1] = 0.

            np.fill_diagonal(D, -dx2*c0*c1*np.exp(c1*(phi)))
            D[0,0] = -dx2*c0*c1
            D[-1,-1] = 0.

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

    def add_particles(self, particles):
        self.added_particles += particles
    #end def add_particles
#end class Grid

def pic_bca():
    #Imports and constants
    import generate_ftridyn_input as gen
    density = 5e16
    N = 100000
    timesteps = 1000
    ng = 300
    dt = 1e-9
    Ti = 100.*11600
    Te = 200.*11600
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 200.*LD
    p2c = density*L/N
    B0 = np.array([1.0, 0.0, 0.0])
    E0 = np.array([0.0, 0.0, 0.0])
    number_histories = 100
    num_energies = 5
    num_angles = 5
    mfp = 5.*LD

    #Skip every 10th plot
    skip=1

    #Calculate floating potential
    phi_floating = (Te/11600.)*0.5*np.log(1.*mp/2./np.pi/me/(1.+Ti/Te))
    print(f"Floating potential: {phi_floating} V")

    #Initialize objects, generators, and counters
    grid = Grid(ng, L, Te)
    particles = [Particle(1.*mp, e, p2c, Ti, Z=1, B0=B0, E0=E0, grid=grid) \
        for _ in range(N - N//10)]

    #impurities = [Particle(10.81*mp, e, p2c, Ti, Z=5, B0=B0, E0=E0, grid=grid) \
    #    for _ in range(N//10)]

    #particles += impurities
    tridyn_interface = gen.tridyn_interface('H', 'B')
    tridyn_interface_B = gen.tridyn_interface('B', 'B')
    source_distribution = source_distribution_6D(grid, Ti, mp)#, -3.*particles[0].vth)
    impurity_distribution = source_distribution_6D(grid, Ti, 10.81*mp)#, -3.*impurities[0].vth)
    num_deposited = 0
    num_sputtered = 0

    #Construct energy and angle range and empty iead array
    angle_range = np.linspace(0.0, 90.0, num_angles)
    energy_range = np.linspace(0.1, 4.*phi_floating, num_energies)
    iead_average = np.zeros((len(energy_range), len(angle_range)))

    #Initialize figures
    fig1 = plt.figure(1)
    plt.ion()
    fig2 = plt.figure(2)
    plt.ion()
    fig3 = plt.figure(3)
    plt.ion()
    fig4 = plt.figure(4)

    #Start of time loop
    time = 0.
    composition_B = 0.
    for time_index in range(timesteps+1):
        #Clear iead collection arrays
        energies_H = []
        angles_H = []
        energies_B = []
        angles_B = []

        #Clear plotting arrays
        positions = np.zeros(N)
        velocities = np.zeros(N)
        colors = np.zeros(N)

        #Advance time, weight particles, solve for phi, differentiate phi to find E
        time += dt
        grid.weight_particles_to_grid_boltzmann(particles, dt)
        grid.reset_added_particles()
        grid.solve_for_phi_dirichlet_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()

        print(f"timestep: {time_index}")
        print(f"n0: {grid.n0}\nadded_particles: {grid.added_particles}")
        print(f"phi_max: {np.max(grid.phi)}")
        print(f"number deposited: {num_deposited}")
        print(f"number sputtered: {num_sputtered}")

        #Begin particle loop
        for particle_index, particle in enumerate(particles):
            #If particle is active, push particles and store positions, velocities
            if particle.is_active():
                #Store particle coordinates for plotting
                positions[particle_index] = particle.get_x()
                velocities[particle_index] = particle.get_v_x()
                colors[particle_index] = particle.get_Z()

                #Interpolate E, push in time, and apply BCs
                particle.interpolate_electric_field_dirichlet(grid)
                particle.push_6D(dt)
                particle.apply_BCs_dirichlet(grid)

                if particle.get_Z() == 5: composition_B += 1./len(particles)

                #If particle is deactivated at wall, store in iead colleciton arrays
                if not particle.is_active():
                    if particle.get_Z() == 1:
                        energies_H.append(particle.get_kinetic_energy()/e)
                        angles_H.append(particle.get_angle_wrt_wall())
                    #end if
                    if particle.get_Z() == 5:
                        energies_B.append(particle.get_kinetic_energy()/e)
                        angles_B.append(particle.get_angle_wrt_wall())
                    #end if
            #If particle is not active, reinitialize as either source H or impurity B
            else:
                if np.random.choice((True, True), p=(0.90, 0.10)):
                    particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1.*e, 1)
                else:
                    particle.reactivate(impurity_distribution, grid, time, p2c, 10.81*mp, 1.*e, 5)
            #end if
        #end for particle_index, particle

        print(f"Percent Boron: {composition_B * 100.}")

        #Collect iead arrays into 2D IEAD histogram
        iead_H, energies_H, angles_H = np.histogram2d(energies_H, angles_H, bins=(num_energies,num_angles))
        iead_B, energies_B, angles_B = np.histogram2d(energies_B, angles_B, bins=(num_energies,num_angles))

        #Run F-TRIDYN for the collected IEADs
        new_particle_list_H_s, new_particle_list_H_r = tridyn_interface.run_tridyn_simulations_from_iead(energies_H, angles_H, iead_H, number_histories=number_histories)
        new_particle_list_B_s, new_particle_list_B_r = tridyn_interface_B.run_tridyn_simulations_from_iead(energies_B, angles_B, iead_B, number_histories=number_histories)


        #Concatenate H and B lists from every NHth particle
        new_particle_list = new_particle_list_H_s[::number_histories] +\
            new_particle_list_H_r[::number_histories] +\
            new_particle_list_B_s[::number_histories] +\
            new_particle_list_B_r[::number_histories]


        #Count number of deposited Boron
        num_deposited += np.sum(iead_B) - len(new_particle_list_B_r[::number_histories])

        num_sputtered += len(new_particle_list_B_s[::number_histories]) +\
            len(new_particle_list_H_s[::number_histories])

        #Create empty new particle array for reflected and sputtered particles
        new_particles = [None]*len(new_particle_list)
        composition_B = 0.
        for index, row in enumerate(new_particle_list):
            #Choose left or right wall, flip cos(alpha) appropriately
            if np.random.choice((True, False)):
                x0 = mfp*abs(row[1])
                row[1] = abs(row[1])
            else:
                x0 = grid.get_length() - mfp*abs(row[1])
                row[1] = -abs(row[1])
            #end if
            #Create new particle
            new_particles[index] = particle_from_energy_angle_coordinates(*row, q=e, p2c=p2c, T=Ti, grid=grid, x0=x0,
                time=time, B=B0)
            #Keep track of added charges for Botlzmann solver
            grid.add_particles(p2c)
        #end for

        #Concatenate particle and new particle lists
        particles += new_particles
        N = len(particles)

        #Plotting routine
        if time_index%skip == 0:
            plt.figure(1)
            plt.clf()
            plt.plot(grid.domain, grid.phi)
            plt.draw()
            plt.savefig('pic_bca_phi'+str(time_index))
            plt.pause(0.001)

            plt.figure(2)
            plt.clf()
            plt.scatter(positions[::10], velocities[::10], s=0.5, c=colors[::10]-1., cmap='jet')
            plt.draw()
            plt.savefig('pic_bca_ps'+str(time_index))
            plt.pause(0.001)

            plt.figure(3)
            plt.clf()
            plt.pcolormesh(energies_H, angles_H, iead_H)
            plt.draw()
            plt.pause(0.001)

            plt.figure(4)
            plt.clf()
            if np.size(new_particle_list_H_s) > 0: plt.hist(np.atleast_2d(new_particle_list_H_s)[:,0], bins=10)
            plt.draw()
            plt.pause(0.001)
        #end if
    #end for time_index

    #Create movies from .png plots
    c.convert('.','pic_bca_ps',0,timesteps,1,'out_ps.gif')
    c.convert('.','pic_bca_phi',0,timesteps,1,'out_phi.gif')
#end def pic_bca

def main():
    density = 5e16
    N = 10000
    T = 1000
    ng = 300
    dt_small = 1e-9
    dt_large = 1e-8
    Ti = 10.0*11600.
    Te = 10.0*11600.
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 200*LD
    p2c = L*density/N
    B0 = np.array([0.0, 0.0, 0.0])
    E0 = np.array([0.0, 0.0, 0.0])

    phi_floating = (Te/11600.)*0.5*np.log(1.0*mp/2./np.pi/me/(1.+Ti/Te))
    print(f"Floating potential: {phi_floating} V")

    fig1 = plt.figure(1)
    plt.ion()
    fig2 = plt.figure(2)
    plt.ion()
    fig3 = plt.figure(3)
    plt.ion()

    np.random.seed(1)
    grid = Grid(ng, L, Te)
    particles = [Particle(1.0*mp, e, p2c, Ti, Z=1, B0=B0, E0=E0, grid=grid) for i in range(N)]

    file = open("particle_output.txt","w+")
    positions_x = np.zeros(N)
    positions_y = np.zeros(N)
    positions_z = np.zeros(N)
    velocities = np.zeros(N)
    colors = np.zeros(N)
    E_t = np.zeros(T+1)
    phi_max_t = np.zeros(T+1)

    #sputtered_distribution = thompson_distribution_isotropic_6D(6.0, mp, grid, grid.dx)
    sputtered_distribution = distribution_from_file('ftridyn/sim0010/H_B_0011SPLST.OUT', mp, grid, 1.0*LD)
    #reflected_distribution = itertools.cycle(distribution_from_file('ftridyn/sim0010/H_B_0011RFLST.OUT', 5.*mp, grid, 2.*grid.dx))
    source_distribution = source_distribution_6D(grid, Ti, mp)
    sputtering_threshold = 28.0*e

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
            if particle.is_active():
                positions_x[index] = particle.r[0]
                velocities[index] = particle.r[3]
                colors[index] = particle.color
                particle.interpolate_electric_field_dirichlet(grid)
                particle.push_6D(dt_small)
                particle.apply_BCs_dirichlet(grid)
            else:
                if (particle.get_kinetic_energy() > sputtering_threshold) and index%10==0:
                    particle.reactivate(sputtered_distribution, grid, time, 293/10000, 10.81*mp, 1.*e, 5)
                    particle.color = 1
                else:
                    particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1.*e, 1)
                    particle.color = 0
                #end if

            #end if
        #end for

        if time_index%1 ==0:
            plt.figure(1)
            plt.clf()
            plt.scatter(positions_x, velocities, s=1.0, c=colors, cmap='jet')
            plt.axis((0.0, grid.get_length(), -8.0*particles[0].vth, 8.0*particles[0].vth))
            plt.draw()
            plt.pause(0.01)
            plt.savefig('plots/ps_'+str(time_index))

            plt.figure(2)
            plt.clf()
            plt.plot(grid.domain, grid.phi)
            plt.draw()
            plt.pause(0.01)
            plt.savefig('plots/phi_'+str(time_index))

            plt.figure(3)
            plt.clf()
            plt.plot(np.linspace(0.0,time,time_index), phi_max_t[:time_index])
            plt.plot(np.linspace(0.0,time,time_index), phi_floating*np.ones(time_index))
            plt.draw()
            plt.pause(0.01)
            plt.savefig('plots/phi_t_'+str(time_index))
        #end if
    #end for
    plt.show()

    c.convert('plots','ps',0,T,10,'out_ps.gif')
    c.convert('plots','phi',0,T,10,'out_phi.gif')
    c.convert('plots','phi_t',0,T,10,'out_phi_t.gif')
#end def main

def run_tests():
    import doctest
    doctest.testmod()
#end def run_tests

if __name__ == '__main__':
    run_tests()
    pic_bca()
