from __future__ import print_function
import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import numba as nb

from concurrent.futures import ThreadPoolExecutor as PoolExecutor

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

def interpolate_p(F, x, Ng, dx):
	index_L = int( np.floor(x/dx) % Ng )
	index_R = (index_L + 1) % Ng

	w_R = (x % dx) / dx
	w_L = 1. - w_R

	return w_L * F[index_L] + w_R * F[index_R]
#end def interpolate_p

@nb.jit('float64[:](float64[:],float64[:],int32,int32,float64)', nopython=True)
def interpolate_p_2(F, x, Ng, N, dx):
	F_interp = np.zeros(N)

	index_L = np.floor(x/dx) % Ng
	index_R = (index_L + 1) % Ng

	w_R = (x % dx) / dx
	w_L = 1. - w_R

	for i in range(N):
		F_interp[i] = F[int(index_L[i])] * w_L[i] + F[int(index_R[i])] * w_R[i]
	#end for

	return F_interp
#end def_interpolate_p_2

@nb.jit('float64[:](float64[:])')
def smooth_field_p(F):
	F_smooth = (np.roll(F,-1) + 2.0 * F + np.roll(F,1)) / 4.0
	return F_smooth
#end def smooth_field_p

@nb.jit('float64[:](float64[:], float64[:], float64[:], int32, int32, int32, float64)',nopython=True)
def weight_current_p(x, q, v, p2c, Ng, N, dx):
	j = np.zeros(Ng)

	index_L = np.floor(x/dx) % Ng
	index_R = (index_L + 1) % Ng

	w_R = (x % dx) / dx
	w_L = 1. - w_R

	idx = (1./dx)

	j_i = q * v * p2c * idx

	j_L = j_i * w_L
	j_R = j_i * w_R

	for i in range(N):
		j[int(index_L[i])] += j_L[i]
		j[int(index_R[i])] += j_R[i]
	#end for

	return j
#end def weight_current_p

@nb.jit('float64[:](float64[:], float64[:], int32, int32, int32, float64)',nopython=True)
def weight_density_p(x, q, p2c, Ng, N, dx):
	rho = np.zeros(Ng)

	index_L = np.floor(x/dx) % Ng
	index_R = (index_L + 1) % Ng

	w_R = (x % dx) / dx
	w_L = 1. - w_R

	idx = (1./dx)

	for i in range(N):
		ind_L = int(index_L[i])
		ind_R = int(index_R[i])
		rho[ind_L] += q[i] * p2c * w_L[i] * idx
		rho[ind_R] += q[i] * p2c * w_R[i] * idx
	#end for

	return rho
#end def weight_density_p

@nb.jit('float64[:](float64[:], float64, int32)',nopython=True)
def differentiate_p(F, dx, Ng):
	dF = np.zeros(Ng)

	for i in range(Ng):
		ind_L = (i-1) % Ng
		ind_R = (i+1) % Ng
		dF[i] = (F[ind_R] - F[ind_L]) / dx * 0.5
	#end for

	return dF
#end def differentiate_p

@nb.jit(nb.types.UniTuple(nb.float64[:],4)(nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:],nb.int32,nb.int32,nb.int32,nb.float64,nb.float64,nb.float64,nb.float64,nb.int32))
def particle_push_p(x0, v0, q, m, E0, j0, N, Ng, p2c, dx, dt, L, tol, maxiter):
	#Initial guess from n-step levels
	Es = E0
	xs = x0

	r = 1.0
	k = 0

	while(r>tol) & (k<maxiter):

		#Particle Loop to find local E-field
		#for i in range(N):
		#	E_interp[i] = interpolate_p(Eh, xh[i], Ng, dx)
		#end for i

		q_m = q/m

		E_interp = interpolate_p_2(Es, xs, Ng, N, dx)

		x1 = x0 + dt * v0 + dt * dt * (q_m) * E_interp*0.5
		v1 = v0 + dt * (q_m) * E_interp

		xh = (x0 + x1) * 0.5
		vh = (v0 + v1) * 0.5

		xh = xh % (L)
		#jh = weight_current_p(xh, q, vh, p24c, Ng, N, dx)

		x1 = x1 % (L)
		j1 = weight_current_p(x1, q, v1, p2c, Ng, N, dx)
		j1 = smooth_field_p(j1)

		jh = (j1 + j0) * 0.5

		E1 = E0 + (dt/epsilon0) * (np.average(jh) - jh)
		E1 = smooth_field_p(E1)
		Eh = (E1 + E0) * 0.5

		r = np.linalg.norm(Es-Eh)

		Es = Eh
		xs = xh

		k += 1
	#end while
	print("Iterations: ",k)
	print("Residual  : ",r)
	return x1,v1,E1,j1
#end def particle_push_p

def differentiate_t(F, dt):
	T = len(F)
	dF = np.zeros(T)

	dF[0] = (F[1] - F[0]) / dt
	for i in range(1, T-1):
		ind_L = i-1
		ind_R = i+1
		dF[i] = (F[ind_R] - F[ind_L]) / dt * 0.5
	#end for
	dF[T-1] = (F[T-1] - F[T-2]) / dt

	return dF
#end differentiate_t

def laplacian_1D_p(Ng):
	A =sp.diag(np.ones(Ng-1),-1) + sp.diag(-2.*np.ones(Ng),0) + sp.diag(np.ones(Ng-1),1)
	A[0, 0]  = -2.
	A[0, 1]  =  1.
	A[0,-1]  =  1.

	A[-1,-1] = -2.
	A[-1,-2] =  1.
	A[-1, 0] =  1.

	return A
#end def laplacian_1D_p

def solve_poisson_p(dx, Ng, rho, phi0):
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
	wp = np.sqrt( e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * 2.0 * np.pi / L
	p2c = L * density / N
	kBTe = kb*Te
	kBTi = kb*Ti
	v_thermal = np.sqrt(kBTe / me)
	#print(v_thermal / 6.5E6)

	LD = np.sqrt(kBTe * epsilon0 / e / e / density)

	m = np.ones(N) * me
	q = -np.ones(N) * e

	if system=='bump-on-tail':
		beam_proportion = N*2/6
		plasma_proportion = N*4/6
		beam_temperature = 1./20.
		beam_drift = 6.0
		growth_rate = np.sqrt(3.)/2.*wp*(float(beam_proportion)/float(plasma_proportion)/2.)**(1./3.)
		v0 = np.zeros(N)
		v0[0:plasma_proportion] = np.random.normal(0.0, np.sqrt(kBTe/me), plasma_proportion)
		v0[plasma_proportion:] = np.random.normal(beam_drift * np.sqrt(kBTe/me), beam_temperature * np.sqrt(kBTe/me), beam_proportion+1)
	#end if

	if system=='two-stream':
		beam_1_proportion = N*1/2
		beam_2_proportion = N*1/2
		beam_drift = 5.0
		growth_rate = np.sqrt(3.)/2.*wp*(float(beam_1_proportion)/float(beam_2_proportion)/2.)**(1./3.)
		v0 = np.zeros(N)
		v0[0:beam_1_proportion] = np.random.normal(-beam_drift * np.sqrt(kBTe/me), np.sqrt(kBTe/me), beam_1_proportion)
		v0[beam_1_proportion:] = np.random.normal(beam_drift * np.sqrt(kBTe/me), np.sqrt(kBTe/me), beam_2_proportion)
	#end if


	if system=='landau damping':
		#d_landau = -np.sqrt(np.pi) * wp**4 / K**3 / np.sqrt(kBTe/me)**3 * np.exp(- wp**2 / K**2 / np.sqrt(kBTe/me)**2 * np.exp(-3./2.))
		#print(d_landau)
		d_landau = -np.sqrt(np.pi) * wp * (wp/K/v_thermal)**3 * np.exp(-1./(2.0 * K**2 * LD**2) - 3./2.)
		#d_landau = -np.sqrt(np.pi) * wp * (wp/K/v_thermal)**3 * np.exp(-1./(2.0 * K**2 * LD**2) - 3./2.)
		#print(d_landau)
		#le = v_thermal / wp
		#d_landau = -np.sqrt(np.pi / 8.) * wp / (K * le)**3 * np.exp( -1./2.0/(K * le)**2 - 3./2.)
		#print(d_landau)
		#exit()
		#Assign velocity initial distribution function
		v0 = np.zeros(N)
		v0 = np.random.normal(0.0, v_thermal ,N)
		growth_rate = d_landau
	#end if

	#perturbation
	x0 = np.random.uniform(0.0, L, N)
	F = 1.0 + np.cos(K * X)
	F = ( N * perturbation ) * F / np.sum(F)
	j = N/2 - int(N*perturbation / 2)
	for i in range(Ng):
		for k in range(int(F[i])):
			x0[j] = np.random.uniform(X[i], X[i+1])
			j+=1
		#end for k
	#end for i

	x0 = x0 % L
	return m, q, x0, v0, kBTe, kBTi, growth_rate, K, p2c, wp, invwp, LD
#end def initialize_p

def implicit_pic(T, nplot):
	tol = 1e-8
	maxiter = 100
	np.random.seed(2)

	#system = 'two-stream'
	#density = 1e15
	#perturbation = 0.1
	#Kp = 1
	#N = 100000
	#Ng = 100
	#dt = 5e-11
	#Ti = 0.1 * 11600.
	#Te = 0.1 * 11600.
	#L = 40.0 * np.sqrt(kb*Te * epsilon0/e/e/density)

	#system = 'bump-on-tail'
	#density = 1e15
	#perturbation = 0.1
	#Kp = 1
	#N = 100000
	#Ng = 40
	#dt = 1e-11
	#Ti = 0.1 * 11600.
	#Te = 0.1 * 11600.
	#L = 20.0 * np.sqrt(kb*Te * epsilon0/e/e/density)

	#Landau damping best params
	system = 'landau damping'
	density = 1e10 # [1/m3]
	perturbation = 0.05
	Kp = 1
	N = 100000
	Ng = 100
	dt = 1E-8 #[s]
	Ti = 0.1 * 11600. #[K]
	Te = 1.0 * 11600. #[K]
	L = 20.0 *  np.sqrt(kb*Te * epsilon0 / e / e / density)


	dx = L / float(Ng)

	print('L: ',L)
	X = np.linspace(0.0, L, Ng+1)

	m, q, x0, v0, kBTe, kBTi, growth_rate, K, p2c, wp, invwp, LD = initialize_p(system, N, density, Kp, perturbation, dx, Ng, Te, Ti, L, X)

	print("wp : ",wp,"[1/s]")
	print("dt : ",dt/invwp," [dt * wp]")
	print("tau: ",invwp,"[s]")
	print("k*LD: ",K*LD)
	print("p2c :", p2c)
	print("gamma: ",growth_rate)

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

	scattermap = plt.cm.viridis( 1.0 - 2.0 * np.sqrt(v0 * v0) / np.max( np.sqrt( v0*v0 ) ) )
	#scattermap = plt.cm.coolwarm((v0 + 0.5*np.max(v0))/np.max(v0))
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

	rho0 = weight_density_p(x0, q, p2c, Ng, N, dx)
	j0 = weight_current_p(x0, q, v0, p2c, Ng, N, dx)
	phi0 = solve_poisson_p(dx, Ng, rho0, phi0)
	phi0 = phi0 - np.max(phi0)
	E0 = -differentiate_p(phi0, dx, Ng)

	r = 1.0
	k = 0

	for t in range(T):
		print('t: ',t)

		x1, v1, E1, j1 = particle_push_p(x0, v0, q, m, E0, j0, N, Ng, p2c, dx, dt, L, tol, maxiter)

		E0 = E1
		x0 = x1
		v0 = v1
		j0 = j1
		k = 0
		r = 1.0

		EE.append(np.sum(epsilon0 * E0*E0 / 2.))
		KE.append(np.sum(me * v0*v0 / 2.))
		TT.append(t * dt)
		j_bias.append(np.average(j0))

		#Plotting routine
		if (t % nplot == 0):
			fig = plt.figure(1)
			plt.clf()
			plt.scatter(x0,v0/np.sqrt(1.5*kBTe/me),s=2.0,color=scattermap)
			#plt.scatter(x0[0::N/100000],v0[0::N/100000]/np.sqrt(2.0*kBTe/me),s=1.0,color=scattermap[0::N/100000])
			ax = plt.gca()
			#ax.set_facecolor('xkcd:dark blue')
			plt.title('Phase Space, Implicit')
			plt.axis([0.0, L, -10., 10.])
			plt.xlabel('$x$ [$m$]')
			plt.ylabel('$v$ [$v_{thermal}$]')
			plt.xticks([0.0, L])
			plt.yticks([-10.0, -5.0, 0.0, 5.0, 10.0])
			plt.draw()
			plt.savefig('plots/ps_'+str(t))
			plt.pause(0.0001)

			plt.figure(2)
			plt.clf()
			plt.plot(X[:-1],j0,linewidth=lw)
			plt.xticks([0.0, L-dx])
			plt.title('Current, Implicit')
			plt.xlabel('$x$ [$m$]')
			plt.ylabel(r'$J$ [$\frac{A}{m^{2}}$]')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(3)
			plt.clf()
			plt.plot(X[:-1],E0,linewidth=lw)
			plt.xticks([0.0, L-dx])
			plt.xlabel('$x$ [$m$]')
			plt.ylabel(r'$E$ [$\frac{V}{m}$]')
			plt.title('Electric Field, Implicit')
			plt.draw()
			plt.savefig('plots/e_'+str(t))
			plt.pause(0.0001)

			plt.figure(4)
			plt.clf()
			plt.semilogy(np.array(TT)*wp,KE,linewidth=lw)
			plt.xlabel(r't [$\omega_{p}^{-1}$]')
			plt.ylabel('$KE$ [$J$]')
			plt.title('KE, Implicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(5)
			plt.clf()
			plt.semilogy(np.array(TT)*wp,EE,linewidth=lw)
			if system == 'landau damping' and t>2:
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
				plt.semilogy(np.array(TT)*wp+TT[index_first_peak]*wp,EE[index_first_peak]*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
				#plt.semilogy(np.array(TT)*wp,np.abs(dEE),linewidth=lw)
			else:
				plt.semilogy(np.array(TT)*wp,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
			#end if
			plt.title('$E^{2}$, Implicit')
			plt.ylabel(r'$E^{2}$ [$\frac{V^{2}}{m^{2}}$]')
			plt.xlabel(r'$t$ [$\omega_{p}^{-1}$]')
			plt.legend([r'$E^{2}$','Theoretical'])
			plt.draw()
			plt.savefig('plots/e2_'+str(t))
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

	#Landau damping best params
	system = 'landau damping'
	density = 1e10 # [1/m3]
	perturbation = 0.05
	Kp = 2
	N = 20000
	Ng = 50
	dt = 1E-8 #[s]
	dx = 0.04	 #[m]
	Ti = 1.0 * 11600. #[K]
	Te = 1.0 * 11600. #[K]

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
			plt.axis([0.0, L, -10., 10.])
			plt.xlabel('$x$ [$m$]')
			plt.ylabel('$v$ [$v_{thermal}$]')
			plt.xticks([0.0, L])
			plt.yticks([-10.0, -5.0, 0.0, 5.0, 10.0])
			plt.draw()
			plt.savefig('plots/ps_'+str(t))
			plt.pause(0.0001)

			plt.figure(2)
			plt.clf()
			plt.plot(X[:-1],j,linewidth=lw)
			plt.xticks([0.0, L-dx])
			plt.title('Current, Implicit')
			plt.xlabel('$x$ [$m$]')
			plt.ylabel(r'$J$ [$\frac{A}{m^{2}}$]')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(3)
			plt.clf()
			plt.plot(X[:-1],E,linewidth=lw)
			plt.xticks([0.0, L-dx])
			plt.xlabel('$x$ [$m$]')
			plt.ylabel(r'$E$ [$\frac{V}{m}$]')
			plt.title('Electric Field, Implicit')
			plt.draw()
			plt.savefig('plots/e_'+str(t))
			plt.pause(0.0001)

			plt.figure(4)
			plt.clf()
			plt.semilogy(np.array(TT)*wp,np.array(KE)+np.array(EE),linewidth=lw)
			plt.xlabel(r't [$\omega_{p}^{-1}$]')
			plt.ylabel('$KE$ [$J$]')
			plt.title('Total Energy, Implicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(5)
			plt.clf()
			plt.semilogy(np.array(TT)*wp,EE,linewidth=lw)
			if system == 'landau damping':
				plt.semilogy(np.array(TT)*wp,np.max(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
			else:
				plt.semilogy(np.array(TT)*wp,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
			#end if
			plt.title('$E^{2}$, Implicit')
			plt.ylabel(r'$E^{2}$ [$\frac{V^{2}}{m^{2}}$]')
			plt.xlabel(r'$t$ [$\omega_{p}^{-1}$]')
			plt.legend([r'$E^{2}$','Theoretical'])
			plt.draw()
			plt.savefig('plots/e2_'+str(t))
			plt.pause(0.0001)
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

if __name__ == '__main__':
	#EE_e = explicit_pic(200,10)
	EE_i = implicit_pic(1000,10)
	plt.figure(7)
	#plt.plot(EE_e)
	plt.plot(EE_i)
	plt.savefig('test')
	plt.show()
