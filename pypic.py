from __future__ import print_function
import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import math
import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import numba as nb

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size' : 12})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)

#plotting linewidth
lw = 3.0

#physical constants
epsilon0 = 8.854E-12
e = 1.602E-19
mp = 1.67E-27
me = 9.11E-31
kb = 1.38E-23

@nb.jit('float64[:](float64[:],float64[:],int32,int32,float64)', nopython=True, nogil=True, parallel=True, fastmath=False)
def interpolate_p(F, x, Ng, N, dx):
    """
    Interpolates field values F from grid to positions in x with periodic BCs.

    Args:
        F (:obj:'numpy.ndarray'): Field to interpolate.
        x (:obj:'numpy.ndarray'): Spatial positions to interpolate to. [m]
        Ng (int32): Number of gridpoints of field.
        dx (float64): grid spacing. [m]

    Returns:
        F_interp (:obj:'numpy.ndarray'): Array of interpolated values.
    """
    #Initialize interpolated field array.
    F_interp = np.zeros(N)

    idx = (1./dx)

    #Calculate left and right cell index arrays from particle positions
    index_L = x*idx #% Ng
    index_R = (index_L + 1)%Ng

    #Calculate left and right weight arrays from particle positions
    w_R = (x%dx)*idx
    w_L = 1. - w_R

    #For every particle, find the interpolated field value at its position
    for i in range(N):
        F_interp[i] = F[int(index_L[i])]*w_L[i] + F[int(index_R[i])]*w_R[i]
    #end for

    return F_interp
#end def_interpolate_p

@nb.jit('float64[:](float64[:])', nogil=True, parallel=True, fastmath=False)
def smooth_field_p(F):
    """
    Applies simple binomial smoothing to a field on the grid with perioidic BCs.

    Args:
        F (:obj:'numpy.ndarray'): Field to smooth.

    Returns:
        F_smooth (:obj:'numpy.ndarray'): Smoothed field.
    """
    F_smooth = (np.roll(F,-1) + 2.0 * F + np.roll(F,1)) * 0.25
    return F_smooth
#end def smooth_field_p

@nb.jit('float64[:](float64[:],int32)', nogil=True, parallel=True, fastmath=False,
    nopython=True)
def smooth_field_nopython_p(F,Ng):
    """
    Applies simple binomial smoothing to a field on the grid with perioidic BCs.
    This version compiles with nopython=True.

    Args:
        F (:obj:'numpy.ndarray'): Field to smooth.
        Ng (int32): Number of grid points.

    Returns:
        F_smooth (:obj:'numpy.ndarray'): Smoothed field.
    """
    F_smooth = np.zeros(Ng)

    for i in range(Ng):
        F_smooth[i] = 0.25*(F[i-1] + 2.0*F[i] + F[(i+1)%Ng])
    #end for i

    return F_smooth
#end def smooth_field_p

@nb.jit(nb.types.UniTuple(nb.float64[:],4)(nb.float64[:], nb.int32, nb.int32,
    nb.float64[:]), nopython=True, nogil=True, parallel=True, fastmath=False)
def find_cell_indices_and_weights_p(x,Ng,N,dx):
    """
    Function to find cell indices and weights for a list of particles.

    Args:
        x (:obj:'numpy.ndarray'): List of particle positions. [m]
        Ng (int32): Number of grid points.
        N (int32): Number of particles.
        dx (float64): grid spacing. [m]

    Returns:
        index_L (:obj:'numpy.ndarray'): list of cell indices to left of particle
        index_R (:obj:'numpy.ndarray'): list of cell indices to right
        w_L (:obj:'numpy.ndarray'): list of weights to left node
        w_R (:obj:'numpy.ndarray'): list of weights to right node
    """
    #Calculate left and right cell index arrays from particle positions
    index_L = x/dx
    index_R = (index_L + 1)%Ng

    #Calculate left and right weight arrays from particle positions
    w_R = (x%dx)/dx
    w_L = 1. - w_R

    return index_L, index_R, w_R, w_L
#end def find_cell_indices_and_weights_p

@nb.jit('float64[:](float64[:], float64[:], float64[:], int32, int32, int32, float64)',
nopython=True, nogil=True, parallel=True, fastmath=False)
def weight_current_p(x, q, v, p2c, Ng, N, dx):
    """
    Weights current density values from particles to the grid.

    Args:
        x (:obj:'numpy.ndarray'): Particle position list. [m]
        q (:obj:'numpy.ndarray'): Particle charge list. [C]
        v (:obj:'numpy.ndarray'): Particle velocity list. [m/s]
        p2c (float64): Number of physical particles represented by one computational particle.
        Ng (int32): Number of grid points.
        N (int32): Number of particles.
        dx (float64): Grid spacing. [m]

    Returns:
        j (:obj:'numpy.ndarray'): Current density. [A/m^2]
    """
    #Initialize empty current array
    j = np.zeros(Ng)
    idx = (1./dx)

    #Calculate left and right cell index arrays from particle positions
    index_L = x*idx
    index_R = (index_L + 1)%Ng

    #Calculate left and right weight arrays from particle positions
    w_R = (x%dx)*idx
    w_L = 1. - w_R

    #Calculate prefactor array for current density calculation
    j_i = q*v*p2c*idx

    #Calculate current density contribution on left and right grid nodes for
    #every particle
    j_L = j_i*w_L
    j_R = j_i*w_R

    #For every particle, add its contribution to the grid nodes to the left and
    #right
    for i in range(N):
        j[int(index_L[i])] += j_L[i]
        j[int(index_R[i])] += j_R[i]
    #end for

    return j
#end def weight_current_p

@nb.jit('float64[:](float64[:], float64[:], int32, int32, int32, float64)',
nopython=True, nogil=True, parallel=True, fastmath=False)
def weight_density_p(x, q, p2c, Ng, N, dx):
    """
    Weights charge density to grid.

    Args:
        x (:obj:'numpy.ndarray'): Particle position list. [m]
        q (:obj:'numpy.ndarray'): Particle charge list. [C]
        p2c (float64): Number of physical particles represented by one computational particle.
        Ng (int32): Number of grid points.
        N (int32): Number of particles.
        dx (float64): Grid spacing. [m]

    Returns:
        rho (:obj:'numpy.ndarray'): Charge density. [C/m^3]
    """
    #Initialize empty current density array
    rho = np.zeros(Ng)

    idx = (1./dx)

    #Calculate left and right cell index arrays from particle positions
    index_L = x*idx# % Ng
    index_R = (index_L + 1)%Ng

    #Calculate left and right weight arrays from particle posiitons
    w_R = (x%dx)/dx
    w_L = 1. - w_R

    #Calculate prefactor for charge density calculation
    q_i = q*p2c*idx

    #Calculate charge density contribution to left and right grid nodes for each
    #particle
    q_L = q_i*w_L
    q_R = q_i*w_R

    #For every particle, add its charge density contribution to the left and
    #right grid nodes
    for i in range(N):
        rho[int(index_L[i])] += q_L[i]
        rho[int(index_R[i])] += q_R[i]
    #end for

    return rho
#end def weight_density_p

@nb.jit('float64[:](float64[:], float64, int32)', nopython=True, nogil=True,
    parallel=True, fastmath=False)
def differentiate_p(F, dx, Ng):
    """
    Numerically differentiate a field with perioidic BCs.

    dF     F(x+dx) - F(x-dx)
    __ ~= ___________________
    dx           2 dx

    Args:
        F (:obj:'numpy.ndarray'): Field to differentiate numerically.
        dx (float64): Grid-spacing. [m]
        Ng (int32): Number of grid points.

    Returns:
        dF (:obj:'numpy.ndarray'): Differentiated field.
    """
    #Initialize empty differentiated field array
    dF = np.zeros(Ng)
    idx_2 = (0.5/dx)

    #Loop over all grid points and calculate numerical derivative
    for i in range(Ng):
        ind_L = (i - 1)# % Ng
        ind_R = (i + 1)%Ng
        dF[i] = (F[ind_R] - F[ind_L])*idx_2
    #end for

    return dF
#end def differentiate_p

@nb.jit(nb.types.UniTuple(nb.float64[:],4)(nb.float64[:],nb.float64[:],
    nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.int32,nb.int32,
    nb.int32,nb.float64,nb.float64,nb.float64,nb.float64,nb.int32),
    nopython=True, nogil=True, parallel=True, fastmath=False)
def particle_push_p(x0, v0, q, m, E0, j0, N, Ng, p2c, dx, dt, L, tol, maxiter):
    """
        Implicit particle pusher and field advancer. Uses Picard iteration and a
        Crank-Nicholson Discretization of the Lorentz force equation to push
        particles in time. Fields are weighted and then evolved using a Crank-
        Nicholson type discretization of Ampere's Law. Simple binomial filter is
        applied to damp 2-delta-x waves.

        Args:
            x0 (:obj:'numpy.ndarray'): Initial particle position list. [m]
            v0 (:obj:'numpy.ndarray'): Initial particle velocity list. [m/s]
            q (:obj:'numpy.ndarray'): Particle charge list. [C]
            m (:obj:'numpy.ndarray'): Particle mass list. [kg]
            E0 (:obj:'numpy.ndarray'): Initial electric field on grid. [V/m]
            j0 (:obj:'numpy.ndarray'): Initial current density on grid. [A/m^2]
            N (int32): Number of particles.
            Ng (int32): Number of grid points.
            p2c (float64): real to computatinal particle ratio.
            dx (float64): Grid spacing. [m]
            dt (float64): timestep. [s]
            L (float64): Domain size. [m]
            tol (float64): tolerance for electric field. [V^2/m^2]
            maxiter (int32): maximum number of iterations for Picard loop.

        Returns:
            x1 (:obj:'numpy.ndarray'): Final particle posiiton list. [m]
            v1 (:obj:'numpy.ndarray'): Final particle velocity list. [m/s]
            E1 (:obj:'numpy.ndarray'): Final electric field. [V/m]
            j1 (:obj:'numpy.ndarray'): Final current density. [A/m^2]

    """
    q_m = q/m

    #Initial guess from n timestep levels for E-field and particle positions
    Es = E0
    xs = x0

    #(Re)set iteration count and residual
    r = 1.0
    k = 0

    #Picard loop
    while(r>tol) & (k<maxiter):
        #Find electric field at all particle positions using smoothed E-field
        E_interp = interpolate_p(smooth_field_nopython_p(Es,Ng), xs, Ng, N, dx)

        #Push particles via Crank-Nicholson to n+1 timestep
        x1 = x0 + dt*v0 + dt*dt*(q_m)*E_interp*0.5
        v1 = v0 + dt*(q_m)*E_interp

        #Find half timestep position and velocity by averaging n & n+1 timesteps
        xh = (x0 + x1)*0.5
        vh = (v0 + v1)*0.5

        #Keep particles at half timestep in periodic domain
        xh = xh%L
        #Weight current at half time step from half timestep position & velocity
        jh = weight_current_p(xh, q, vh, p2c, Ng, N, dx)

        #Keep particles at n+1 timestep in periodic domain
        x1 = x1%L
        #Weight current at n+1 time step from n+1 timestep position and velocity
        j1 = weight_current_p(x1, q, v1, p2c, Ng, N, dx)

        #Use 1D Ampere's Law to find E-field at n+1 timestep from current at
        #half timestep
        E1 = E0 + (dt/epsilon0)*(np.sum(jh)/Ng - smooth_field_nopython_p(jh,Ng))
        #Find half timestep E-field from E-field at n+1 & n timesteps
        Eh = (E1 + E0)*0.5

        #Calculate residual from E-field half timestep guess & half timestep
        #E-field
        r = np.linalg.norm(Es-Eh)

        #Set new E-field and position guesses
        Es = Eh
        xs = xh

        k += 1
    #end while
    print("Iterations: ",k)
    print("Residual  : ",r)
    return x1,v1,E1,j1
#end def particle_push_p

def differentiate_t(F, dt):
    """
        Differentiate a field w.r.t. time:

        dF     F(t+dt) - F(t-dt)
        __ ~= ___________________
        dt            2 dt

        And on the final time step:

        dF     F(t) - F(t-dt)
        __ ~= ________________
        dt           dt

        Args:
            F (:obj:'numpy.ndarray'): Values over time to differentiate.
            dt (float64): timestep. [s]

        Returns:
            dF (:obj:'numpy.ndarray'): Differentiated field in time.
    """
    T = len(F)
    dF = np.zeros(T)

    dF[0] = (F[1] - F[0]) / dt
    for i in range(1, T-1):
        ind_L = i - 1
        ind_R = i + 1
        dF[i] = (F[ind_R] - F[ind_L])/dt*0.5
    #end for
    dF[T-1] = (F[T-1] - F[T-2])/dt

    return dF
#end differentiate_t

def laplacian_1D_p(Ng):
    """
        Generate 1D laplacian to solve Poisson's problem.

        Args:
            Ng (int32): Number of grid points.

        Returns:
            A (:obj:'numpy.ndarray'): 1D Laplacian stencil.
    """
    A = sp.diag(np.ones(Ng-1),-1) + sp.diag(-2.*np.ones(Ng),0) + sp.diag(np.ones(Ng-1),1)
    A[0, 0]  = -2.
    A[0, 1]  =  1.
    A[0,-1]  =  1.

    A[-1,-1] = -2.
    A[-1,-2] =  1.
    A[-1, 0] =  1.

    return A
#end def laplacian_1D_p

def solve_poisson_p(dx, Ng, rho, phi0):
    """
    Solves a 1D Poisson Problem using scipy sparse matrix routines.

    Args:
        dx (float64): Grid spacing. [m]
        Ng (int32): Number of grid points.
        rho (:obj:'numpy.ndarray'): Charge density. [C/m^3]
        phi0 (:obj:'numpy.ndarray'): Initial guess of electric potential. [V]

    Returns:
        phi (:obj:'numpy.ndarray'): Solved electric potential. [V]
    """
    phi = phi0
    D = np.zeros((Ng,Ng))
    A = spp.csc_matrix(laplacian_1D_p(Ng))
    dx2 = dx*dx
    c0 = -np.average(rho) / epsilon0
    c2 = rho/epsilon0

    phi = sppla.spsolve(A, -dx2 * c0 - dx2 * c2)

    return phi
#end def solve_poisson_p

def initialize_p(system, N, density, Kp, perturbation, dx, Ng, Te, Ti, L, X):
    """
    Initialize a particle-in-cell simulation.

    Args:
        system (:obj:'str'): system name. Options are 'bump-on-tail', 'two-stream', and 'landau-damping.'
        N (int32): Number of particles.
        density (float64): Number density of species. [1/m^3]
        Kp (float64): Number of wavelengths in domain for perturbation.
        perturbation (float64): Strength of perturbation.
        dx (float64): Grid-spacing. [m]
        Ng (int32): Number of grid points.
        Te (float64): Electron temperature. [eV]
        Ti (float64): Ion temperature. [eV]
        L (float64): Domain size. [m]
        X (:obj:'numpy.ndarray'): Grid positions list. [m]

    Returns:
        m (:obj:'numpy.ndarray'): List of particle masses. [kg]
        q (:obj:'numpy.ndarray'): List of particle charges. [C]
        x0 (:obj:'numpy.ndarray'): Initial particle position list. [m]
        v0 (:obj:'numpy.ndarray'): Initial particle velocity list. [m]
        kBTe (float64): Electron temperature times Boltzmann's constant. [J]
        kBTi (float64): Ion temperature times Boltzmann's constant. [J]
        growth_rate (float64): Theoretical growth rate of instability.
        K (float64): Wavenumber of perturbation. [1/m]
        p2c (float64): Ratio of real to computational particles.
        wp (float64): Plasma frequency (electrons). [rad/s]
        invwp (float64): Inverse plasma frequency (electrons). [s/rad]
        LD (float64): Debye length. [m]
    """
    #Calculate plasma parameters from input parameters
    wp = np.sqrt( e**2 * density / epsilon0 / me)
    invwp = 1./wp
    K = Kp * 2.0 * np.pi / L
    p2c = L * density / N
    kBTe = kb*Te
    kBTi = kb*Ti
    v_thermal = np.sqrt(2.0 * kBTe / me)
    LD = np.sqrt(kBTe * epsilon0 / e / e / density)

    m = np.ones(N) * me
    q = -np.ones(N) * e

    if system=='bump-on-tail':
        beam_proportion = N*1//6
        plasma_proportion = N*5//6
        beam_temperature = 1./20.
        beam_drift = 4.0
        growth_rate = np.sqrt(3.)/2.*wp*(float(beam_proportion)/float(plasma_proportion)/2.)**(1./3.)
        v0 = np.zeros(N)
        v0[0:plasma_proportion] = np.random.normal(0.0, np.sqrt(kBTe/me), plasma_proportion)
        v0[plasma_proportion:] = np.random.normal(beam_drift * np.sqrt(kBTe/me), beam_temperature * np.sqrt(kBTe/me), beam_proportion+1)
    #end if

    if system=='two-stream':
        beam_1_proportion = N*1//2
        beam_2_proportion = N*1//2
        beam_temperature = 1./5.
        beam_drift = 2.0
        growth_rate = np.sqrt(3.)/2.*wp*(float(beam_1_proportion)/float(beam_2_proportion)/2.)**(1./3.)
        v0 = np.zeros(N)
        v0[0:beam_1_proportion] = np.random.normal(-beam_drift * np.sqrt(kBTe/me), beam_temperature * np.sqrt(kBTe/me), beam_1_proportion)
        v0[beam_1_proportion:] = np.random.normal(beam_drift * np.sqrt(kBTe/me), beam_temperature * np.sqrt(kBTe/me), beam_2_proportion)
    #end if

    if system=='landau-damping':
        #Assign velocity initial distribution function
        v0 = np.zeros(N)
        v0 = np.random.normal(0.0, v_thermal / np.sqrt(2) ,N)
        growth_rate = -np.sqrt(np.pi) * wp * (wp/K/v_thermal)**3 * np.exp(-1./(2.0 * K**2 * LD**2) - 3./2.)
    #end if

    #perturbation
    x0 = np.random.uniform(0.0, L, N)
    F = 1.0 + np.cos(K * X)
    F = ( N * perturbation ) * F / np.sum(F)
    j = 0
    for i in range(Ng):
        for k in range(int(F[i])):
            x0[j] = np.random.uniform(X[i], X[i+1])
            j+=1
        #end for k
    #end for i

    return m, q, x0, v0, kBTe, kBTi, growth_rate, K, p2c, wp, invwp, LD
#end def initialize_p

def implicit_pic(T, nplot, system, density, perturbation, Kp, N, Ng, Nv, Vmax, dt, Ti, Te, L, tol, maxiter):
    """
    Main implicit particle-in-cell routine. Produces live plots of all relevant parameters.

    Args:
        T (int32): Number of time steps.
        nplot (int32): Number of time steps between plots.
        system (:obj:'str'): system type. Options are 'bump-on-tail', 'two-stream', and 'landau-damping'.
        density (float64): Number density of simulation. [1/m^3]
        perturbation (float64): Perturbation strength.
        Kp (float64): Wavelengths per domain.
        N (int32): Number of particles.
        Ng (int32): Number of gridpoints.
        Nv (int32): Number of velocity gridpoints for phase-space plots.
        Vmax (float64): Max velocity for phase-space plots in thermal speeds.
        dt (float64): Timestep. [s]
        Ti (float64): Ion temperature. [eV]
        Te (float64): Electron temperature. [eV]
        L (float64): domainsize. [m]
        tol (float64): tolerance on electric field squared error. [V^2/m^2]
        maxiter (int32): Maximum number of Picard iterations.
    """
    #Tracer particle index for summary plot
    tracer = 9999

    #V and X domains (V domain used only for summary plot)
    V = np.linspace(-Vmax, Vmax, Nv)
    dv = V[1] - V[0]
    X = np.linspace(0.0, L, Ng+1)
    dx = L / float(Ng)

    #Initialize system
    m, q, x0, v0, kBTe, kBTi, growth_rate, K, p2c, wp, invwp, LD = initialize_p(system, N, density, Kp, perturbation, dx, Ng, Te, Ti, L, X)

    #Print plasma parameters
    print("wp : ",wp,"[1/s]")
    print("dt : ",dt/invwp," [dt * wp]")
    print("tau: ",invwp,"[s]")
    print("k*LD: ",K*LD)
    print("p2c :", p2c)
    print("gamma: ",growth_rate)

    fig1 = plt.figure(1,figsize=(20,10))

    #Initialize time-tracking arrays.
    KE = []
    EE = []
    TT = []
    j_bias = []
    trajectory_v = []
    trajectory_x = []

    #colormap for scatterplot that approximates density plot
    scattermap = plt.cm.viridis( 1.0 - 2.0 * np.sqrt(v0 * v0) / np.max( np.sqrt( v0*v0 ) ) )
    #scattermap = plt.cm.coolwarm((v0 + 0.5*np.max(v0))/np.max(v0))

    #Initialize field, position, and velocity arrays
    E0 = np.zeros(Ng)
    Eh = np.zeros(Ng)
    E1 = np.zeros(Ng)
    Es = np.zeros(Ng)
    E_interp = np.zeros(N)

    xh = np.zeros(N)
    x1 = np.zeros(N)
    xs = np.zeros(N)

    vh = np.zeros(N)
    v1 = np.zeros(N)

    j0 = np.zeros(Ng)
    jh = np.zeros(Ng)
    j1 = np.zeros(Ng)

    rho0 = np.zeros(Ng)
    phi0 = np.zeros(Ng)

    #Find initial electric field via poisosn solve.
    rho0 = weight_density_p(x0, q, p2c, Ng, N, dx)
    j0 = weight_current_p(x0, q, v0, p2c, Ng, N, dx)
    phi0 = solve_poisson_p(dx, Ng, rho0, phi0)
    phi0 = phi0 - np.max(phi0)
    E0 = -differentiate_p(phi0, dx, Ng)

    #Time loop
    for t in range(T):
        print('t: ',t)

        #particle push and field advance
        x1, v1, E1, j1 = particle_push_p(x0, v0, q, m, E0, j0, N, Ng, p2c, dx, dt, L, tol, maxiter)

        #Set new n timestep from previous n+1 timestep
        E0 = E1
        x0 = x1
        v0 = v1
        j0 = j1

        #Find time-tracked information.
        TT.append(t * dt)
        EE.append(np.sum(epsilon0 * E0*E0 * dx / 2.))
        KE.append(p2c * np.sum(me * v0*v0 / 2.))
        print("Total Energy: ", (np.sum(epsilon0 * E0*E0 * dx / 2.) + p2c * np.sum(me * v0*v0 / 2.)),' J')
        j_bias.append(np.average(j0))
        trajectory_x.append(x0[tracer])
        trajectory_v.append(v0[tracer]/np.sqrt(kBTe/me))

        #Plotting routine
        if (t % nplot == 0):
            fig1 = plt.figure(1)
            plt.clf()
            ax = fig1.subplots(2, 2)
            ax[0,0].hist2d( x0, v0/np.sqrt(kBTe/me), bins=(Ng*4,Nv*4), range=[[0.0, L],[-Vmax, Vmax]], norm=mpl.colors.PowerNorm(0.8))
            #ax[0,0].contourf(X0,V0,np.rot90(d0),7)
            #ax[0,0].scatter(trajectory_x,trajectory_v,color='white',s=1.0)
            ax[0,0].set_ylim([-Vmax, Vmax])
            ax[0,0].set_xticks([0.0, L])
            ax[0,0].set_title('Phase Space Density')
            ax[0,0].set_xlabel('x [m]')
            ax[0,0].set_ylabel('v [thermal]')

            ax[0,1].hist(v0/np.sqrt(kBTe/me),bins=200,orientation='horizontal',density=True,histtype='stepfilled',color='grey')
            ax[0,1].set_ylim([-Vmax, Vmax])
            ax[0,1].set_title('Total Distribution Function')
            ax[0,1].set_xlabel('')
            ax[0,1].set_ylabel('v [thermal]')

            ax[1,1].semilogy(np.array(TT) * wp, EE, linewidth=lw)
            if system == 'landau-damping' and t>2:
                dEE = differentiate_t(EE, dt)
                dEE_shift = dEE[1:]
                dEE_trunc = dEE[:-1]
                dEE_multi = dEE_shift * dEE_trunc
                index_first_peak = 0
                for s in range(len(dEE_multi)):
                    if dEE_multi[s] < 0.0 and dEE_trunc[s]>0.0:
                        index_first_peak = s + 1
                        break
                    #end if
                #end for
                ax[1,1].semilogy(np.array(TT)*wp+TT[index_first_peak]*wp,EE[index_first_peak]*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
                #plt.semilogy(np.array(TT)*wp,np.abs(dEE),linewidth=lw)
            else:
                ax[1,1].semilogy(np.array(TT)*wp,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
            #end if
            ax[1,1].set_yticks([])
            ax[1,1].legend(['E2','Theoretical'])
            ax[1,1].set_title('Total Electrostatic Energy')
            ax[1,1].set_xlabel('t [1/wp]')
            ax[1,1].set_ylabel('E2 [J]')

            ax[1,0].plot(X[:-1],smooth_field_p(0.6 * E0/np.max(E0)), linewidth=lw,color='blue')
            ax[1,0].plot(X[:-1],smooth_field_p(0.6 * j0/np.max(j0)),linewidth=lw,color='black')
            #ax[1,0].plot(X[:-1],smooth_field_p(0.6 * (rho0/np.max(rho0)-1.0)), linewidth=lw, color='orange')
            ax[1,0].set_xticks([0.0, L-dx])
            ax[1,0].set_ylim([-1.0, 1.0])
            ax[1,0].set_xlim([0.0,L-dx])
            ax[1,0].legend(['E','J'])
            ax[1,0].set_xlabel('x [m]')
            ax[1,0].set_ylabel('A.U.')
            plt.draw()
            plt.savefig('plots/summary_'+str(t))
            plt.pause(0.0001)
        #end if
    #end for t

    np.savetxt('plots/E2.txt',EE)
    np.savetxt('plots/J.txt',j0)
    output_file = open('plots/parameters.out','w+')
    print('wp',wp,file=output_file)
    print('Te',Te,file=output_file)
    print('G',growth_rate,file=output_file)
    print('tau',1.0/wp,file=output_file)
    print('p2c',p2c,file=output_file)
    print('dt',dt,file=output_file)
    print('dx',dx,file=output_file)
    print('Ng',Ng,file=output_file)
    print('L',L+dx,file=output_file)

    return EE
#end def implicit_pic

def explicit_pic(T, nplot):
    np.random.seed(1)

    #system = 'bump-on-tail'
    #density = 1e10
    #perturbation = 0.05
    #Kp = 1
    #N = 100000
    #Ng = 40
    #dt = 2e-8
    #dx = 0.1
    #Ti = 0.1 * 11600.
    #Te = 10.0 * 11600.

    #landau-damping best params
    #system = 'landau-damping'
    #density = 1e10 # [1/m3]
    #perturbation = 0.05
    #Kp = 2
    #N = 100000
    #Ng = 100
    #dt = 1E-8 #[s]
    #dx = 0.04     #[m]
    #Ti = 1.0 * 11600. #[K]
    #Te = 1.0 * 11600. #[K]

    L = Ng * dx
    print('L: ',L)
    X = np.linspace(0.0, L, Ng+1)

    m, q, x, v, kBTe, growth_rate, K, p2c, wp, invwp = initialize_p(system, N, density, Kp, perturbation, dx, Ng, Te, L, X)

    print("wp : ",wp,"[1/s]")
    print("dt : ",dt/invwp," [w * tau]")
    print("tau: ",invwp,"[s]")
    print("k  : ",K,"[1/m]")
    print("p2c :", p2c)

    #Initialize figures
    num_figs = 5
    for i in range(num_figs):
        plt.figure(i+1)
        plt.ion()
    #end for i

    KE = []
    EE = []
    TT = []
    j_bias = []

    scattermap = plt.cm.viridis( 1.0 - 2.0 * np.sqrt(v * v) / np.max( np.sqrt( v*v ) ) )

    E = np.zeros(Ng)
    phi = np.zeros(Ng)
    j = np.zeros(Ng)
    rho = weight_density_p(x,q,p2c,Ng,N,dx)
    rho0 = np.average(rho)

    for t in range(T):
        print('t: ', t)

        #PIC loop
        rho = weight_density_p(x, q, p2c, Ng, N, dx)
        j = weight_current_p(x, q, v, p2c, Ng, N, dx)
        phi = solve_poisson_p(dx, Ng, rho, phi)
        phi = phi - np.max(phi)
        E = -differentiate_p(phi, dx, Ng)
        E_interp = np.zeros(N)

        for i in range(N):
            E_interp[i] = interpolate_p(E, x[i], Ng, dx)
        #end for i

        vhalf = v + (q/m) * (dt*0.5) * E_interp
        xout = x + vhalf * dt
        vout = vhalf + (q/m) * (dt*0.5) * E_interp

        x = xout % L
        v = vout

        EE.append(np.sum(epsilon0 * E*E / 2.))
        KE.append(np.sum(me * v*v / 2.)*p2c**2)
        TT.append(dt * t)

        #Plotting routine
        if (t % nplot == 0):
            plt.figure(1)
            plt.clf()
            plt.scatter(x,v/np.sqrt(2.0*kBTe/me),s=0.5,color=scattermap)
            plt.title('Phase Space, Implicit')
            plt.axis([0.0, L, -5., 5.])
            plt.xlabel('$x$ [$m$]')
            plt.ylabel('$v$ [$v_{thermal}$]')
            plt.xticks([0.0, L])
            plt.yticks([-Vmax, -Vmax/2.0, 0.0, Vmax/2.0, Vmax])
            plt.draw()
            plt.savefig('plots/ps_'+str(t))
            #plt.pause(0.0001)

            plt.figure(2)
            plt.clf()
            plt.plot(X[:-1],j,linewidth=lw)
            plt.xticks([0.0, L-dx])
            plt.title('Current, Implicit')
            plt.xlabel('$x$ [$m$]')
            plt.ylabel(r'$J$ [$\frac{A}{m^{2}}$]')
            #plt.draw()
            #plt.pause(0.0001)

            plt.figure(3)
            plt.clf()
            plt.plot(X[:-1],E,linewidth=lw)
            plt.xticks([0.0, L-dx])
            plt.xlabel('$x$ [$m$]')
            plt.ylabel(r'$E$ [$\frac{V}{m}$]')
            plt.title('Electric Field, Implicit')
            #plt.draw()
            plt.savefig('plots/e_'+str(t))
            #plt.pause(0.0001)

            plt.figure(4)
            plt.clf()
            plt.semilogy(np.array(TT)*wp,np.array(KE)+np.array(EE),linewidth=lw)
            plt.xlabel(r't [$\omega_{p}^{-1}$]')
            plt.ylabel('$KE$ [$J$]')
            plt.title('Total Energy, Implicit')
            #plt.draw()
            #plt.pause(0.0001)

            plt.figure(5)
            plt.clf()
            plt.semilogy(np.array(TT)*wp,EE,linewidth=lw)
            if system == 'landau-damping':
                plt.semilogy(np.array(TT)*wp,np.max(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
            else:
                plt.semilogy(np.array(TT)*wp,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
            #end if
            plt.title('$E^{2}$, Implicit')
            plt.ylabel(r'$E^{2}$ [$\frac{V^{2}}{m^{2}}$]')
            plt.xlabel(r'$t$ [$\omega_{p}^{-1}$]')
            plt.legend([r'$E^{2}$','Theoretical'])
            #plt.draw()
            plt.savefig('plots/e2_'+str(t))
            #plt.pause(0.0001)
            #end if
        #end for t
    np.savetxt('plots/E2.txt',EE)
    np.savetxt('plots/J.txt',j)
    output_file = open('plots/parameters.out','w+')
    print('wp',wp,file=output_file)
    print('Te',Te,file=output_file)
    print('G',growth_rate,file=output_file)
    print('tau',1.0/wp,file=output_file)
    print('p2c',p2c,file=output_file)
    print('dt',dt,file=output_file)
    print('dx',dx,file=output_file)
    print('Ng',Ng,file=output_file)
    print('L',L+dx,file=output_file)
    return EE
#end def explicit_pic

def main(T,nplot):
    """
    Runs an implicit particle-in-cell simulation.

    Args:
        T (int32): Number of timesteps.
        nplot (int32): Number of timesteps between plots.
    """
    #system = 'two-stream'
    #density = 1e10
    #perturbation = 0.1
    #Kp = 1
    #N = 1000000
    #Ng = 100
    #dt = 1e-7
    #Ti = 0.1 * 11600.
    #Te = 0.1 * 11600.
    #L = 12.0 * np.sqrt(kb*Te * epsilon0/e/e/density)
    #L = np.sqrt(3.) * np.sqrt( e**2 * density / epsilon0 / me) / 2. / np.sqrt(kb*Te/me) / 2.

    #system = 'bump-on-tail'
    #density = 1e5
    #perturbation = 0.1
    #Kp = 1
    #N = 1000000
    #Ng = 50
    #dt = 1e-5
    #Ti = 0.1 * 11600.
    #Te = 0.1 * 11600.
    #L = 30.0 * np.sqrt(kb*Te * epsilon0/e/e/density)

    #landau-damping best params
    system = 'landau-damping'
    density = 1e5 # [1/m3]
    perturbation = 0.01
    Kp = 1
    N =  20000
    Ng = 40
    dt = 1e-5 #[s]
    Ti = 0.1 * 11600. #[K]
    Te = 100.0 * 11600. #[K]
    L = 22.0 *  np.sqrt(kb*Te * epsilon0 / e / e / density)

    Vmax = 6.0
    Nv = Ng/2
    tol = 1e-15
    maxiter = 100

    EE_i = implicit_pic(T,nplot,system,density,perturbation,Kp,N,Ng,Nv,Vmax,dt,Ti,Te,L,tol,maxiter)
#end def main

if __name__ == '__main__':
    main(10000,20)
#end if
