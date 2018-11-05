from __future__ import print_function
import numpy as np
import scipy.linalg as la
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import gc as gc

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size' : 12})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)

lw = 3.0

np.random.seed(1)

#enable garbage collection
gc.enable()

#physical constants
epsilon0 = 8.854E-12
e = 1.602E-19
mp = 1.67E-27
me = 9.11E-31
kb = 1.38E-23

def interpolateField(F,x,dx):
	index = int(np.floor(x/dx))
	wR = x % dx / dx
	wL = 1.0 - wR
	return wL * F[index] + wR * F[index+1]
#end interpolateFieldac

def interpolateFieldPeriodic(F,x,Ng,dx):
	index = int(np.floor(x/dx)) % (Ng+1)

	wR = (x % dx) / dx
	wL = 1. - wR

	return wL * F[index] + wR * F[index+1]
#end interpolateField

def weightCurrents(x,q,v,p2c,Ng,N,dx):
	j = np.zeros(Ng)
	for i in range(N):
		index = int(np.floor(x[i]/dx))

		wR = (x[i] % dx) / dx
		wL = 1. - wR

		j[index] += (1./dx) * q[i] * p2c * v[i] * wL
		j[index+1] += (1./dx) * q[i] * p2c * v[i] * wR
	#end for
	return j
#end def weightCurrents

def weightCurrentsPeriodic(x,q,v,p2c,Ng,N,dx):
	j = np.zeros(Ng+1)

	index = np.floor(x/dx) % (Ng+1)
	wR = (x % dx) / dx
	wL = 1. - wR

	idx = (1./dx)

	for i in range(N):
		ind = int(index[i])
		j[ind] += q[i] * v[i] * p2c * wL[i] * idx
		j[ind+1] += q[i] * v[i] * p2c * wR[i] * idx
	#end for

	j[0] = j[-1] + j[0]
	j[-1] = j[0]
	return j
#end def weightCurrents


def weightDensities(x,q,p2c,Ng,N,dx):
	rho = np.zeros(Ng)

	index = np.floor(x/dx)

	wR = (x % dx) / dx
	wL = 1. - wR

	idx = (1./dx)

	for i in range(N):
		rho[int(index[i])] += q[i] * p2c * wL[i] * idx
		rho[int(index[i])+1] += q[i] * p2c * wR[i] * idx
	#end for
	return rho
#end def weightDensities

def weightDensitiesPeriodic(x,q,p2c,Ng,N,dx):
	rho = np.zeros(Ng+1)

	index = np.floor(x/dx) % (Ng+1)

	wR = (x % dx) / dx
	wL = 1. - wR

	idx = (1./dx)

	for i in range(N):
		ind = int(index[i])
		rho[ind] += q[i] * p2c * wL[i] * idx
		rho[ind+1] += q[i] * p2c * wR[i] * idx
	#end for
	rho[-1] = rho[0] + rho[-1]
	rho[0] = rho[-1]
	return rho
#end def weightDensities

def laplacian1DPeriodic(Ng):
	A =sp.diag(np.ones(Ng-1+1),-1) + sp.diag(-2.*np.ones(Ng+1),0) + sp.diag(np.ones(Ng-1+1),1)

	A[0,0]  = -2.
	A[0,1]  = 1.
	A[0,-1] = 1.

	A[-1,-1] = -2.
	A[-1,-2] = 1.
	A[-1,0]  = 1.

	return A
#end def laplacian1D

def laplacian1D(Ng):
	A =sp.diag(np.ones(Ng-1),-1) + sp.diag(-2.*np.ones(Ng),0) + sp.diag(np.ones(Ng-1),1)
	A[0,0] = 1.
	A[0,1] = 0.
	A[0,2] = 0.

	A[-1,-1] = -2.
	A[-1,-2] = 1.
	A[-1,-3] = 1.
	return A
#end def laplacian1D

def solvePoisson(dx,Ng,rho,kBT,tol,maxiter,phi0):
	phi = phi0
	D = np.zeros((Ng,Ng))
	A = laplacian1D(Ng)

	resid = 1.
	k = 0

	dx2 = dx*dx
	c0 = rho[Ng/2] / epsilon0
	c1 = e / kBT
	c2 = rho/epsilon0

	while (resid > tol) & (k <= maxiter):
		F = A.dot(phi) - dx2 * c0 * np.exp(c1 * (phi-phi[Ng/2]) ) + dx2 * c2
		F[0] = phi[0]
		F[-1] = phi[-1]

		np.fill_diagonal(D, -dx2 * c0 * c1 * np.exp( c1 * (phi-phi[Ng/2]) ))
		D[0,0] = -dx2 * c0 * c1
		D[-1,-1] = -dx2 * c0 * c1

		J = spp.csc_matrix(A + D)
		dphi = sppla.inv(J).dot(F)

		phi = phi - dphi
		resid = la.norm(dphi)
		k+=1
	#end while

	return phi
#end def solvePoisson

def solvePoissonPeriodic(dx,Ng,rho,kBT,tol,maxiter,phi0):
	phi = phi0
	D = np.zeros((Ng+1,Ng+1))
	A = laplacian1DPeriodic(Ng)

	resid = 1.
	k = 0

	dx2 = dx*dx
	c0 = rho[(Ng+1)/2] / epsilon0
	c1 = e / kBT
	c2 = rho/epsilon0

	while (resid > tol) & (k <= maxiter):
		F = A.dot(phi) - dx2 * c0 * np.exp(c1 * (phi-phi[(Ng+1)/2]) ) + dx2 * c2

		np.fill_diagonal(D, -dx2 * c0 * c1 * np.exp( c1 * (phi-phi[(Ng+1)/2]) ))

		J = spp.csc_matrix(A + D)
		dphi = sppla.inv(J).dot(F)

		phi = phi - dphi
		resid = la.norm(dphi)
		k+=1
	#end while

	return phi
#end def solvePoisson

def solvePoissonPeriodicElectronsNeutralized(dx,Ng,rho,kBT,tol,maxiter,phi0):
	phi = phi0
	D = np.zeros((Ng+1,Ng+1))
	A = spp.csc_matrix(laplacian1DPeriodic(Ng))

	dx2 = dx*dx
	c0 = -np.average(rho) / epsilon0
	c2 = rho / epsilon0

	phi = sppla.spsolve(A,-dx2 * c0 - dx2 * c2)

	return phi
#end def solvePoisson

def differentiateField(F,dx,Ng):
	dF = np.zeros(Ng)

	for i in range(1,Ng-1):
		dF[i] = -(F[i+1] - F[i-1]) / dx / 2.
	#end for

	dF[0] = -(F[1] - F[0]) / dx
	dF[-1] = -(F[-1] - F[-2]) / dx

	return dF
#end def differentiateField

def differentiateFieldPeriodic(F,dx,Ng):
	dF = np.zeros(Ng+1)

	for i in range(1,Ng+1-1):
		dF[i] = -(F[i+1] - F[i-1]) / dx * 0.5
	#end for

	dF[-1] = -(F[0] - F[-2]) / dx * 0.5
	dF[0] = -(F[1] - F[-1]) / dx * 0.5

	return dF
#end def differentiateField

def pushParticlesExplicit(x,v,q,m,N,Ng,dt,dx,E):
	E_interp = np.zeros(N)

	for i in range(N):
		E_interp[i] = interpolateFieldPeriodic(E,x[i],Ng,dx)
	#end for

	vhalf = v + (q/m) * (dt*0.5) * E_interp
	xout = x + vhalf * dt
	vout = vhalf + (q/m) * (dt*0.5) * E_interp
	return xout,vout
	#end def PushParticlesExplicit

def pushParticlesImplicit(x0,xh,v,q,m,N,Ng,dt,dx,Eh):
	E_interp = np.zeros(N)

	for i in range(N):
		E_interp[i] = interpolateFieldPeriodic(Eh,xh[i],Ng,dx)
	#end for

	xout = x0 + dt * v + dt * dt * (q/m) * E_interp*0.5
	vout = v + dt * (q/m) * E_interp
	return xout,vout

def applyBoundaryConditions(x,v,m,N,L,dx,kBT):
	for i in range(N):
		if x[i] > L:
			x[i] = np.random.uniform(1.*dx,L-1.*dx)
			v[i] = np.random.normal(0.0, np.sqrt(kBT/mp), 1)
		elif x[i] <= 0.:
			x[i] = np.random.uniform(1.*dx,L-1.*dx)
			v[i] = np.random.normal(0.0, np.sqrt(kBT/mp), 1)
		#end if
	return x,v
	#end def applyBoundaryConditions

def applyBoundaryConditionsPeriodic(x,v,m,N,L,dx,kBT):
	xout = x % (L+dx)
	vout = v
	return xout,vout
	#end def applyBoundaryConditions

def initialize(system,N,density,Kp,perturbation,dx,Ng,Te,L,X):
	wp = np.sqrt(e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * np.pi / (L+dx)
	p2c = (L+dx) * density / N
	kBTe = kb*Te #[J]
	vthermal = np.sqrt(2.0 * kBTe / me)
	LD = 7430.0 * np.sqrt(kBTe/e/density)
	print('debye length * kp: ',LD*K)

	m = np.ones(N) * me
	q =  -np.ones(N) * e

	if system=='bump-on-tail':
		beam_proportion = N*2/6
		plasma_proportion = N*4/6
		beam_temperature = 1./20.
		beam_drift = 5.0

		growth_rate = np.sqrt(3.)/2.*wp*(float(beam_proportion)/float(plasma_proportion)/2.)**(1./3.)

		#Assign velocity initial distribution function
		v0 = np.zeros(N)
		v0[0:plasma_proportion] = np.random.normal(0.0, np.sqrt(kBTe/me), plasma_proportion)
		v0[plasma_proportion:] = np.random.normal(beam_drift * np.sqrt(kBTe/me), beam_temperature * np.sqrt(kBTe/me), beam_proportion+1)
	#end if

	if system=='landau damping':
		#d_landau = -np.sqrt(np.pi) * wp**4 / K**3 / np.sqrt(kBTe/me)**3 * np.exp(- wp**2 / K**2 / np.sqrt(kBTe/me)**2 * np.exp(-3./2.))

		d_landau = -np.sqrt(np.pi) * wp * (wp / K / vthermal)**3 * np.exp( -wp**2/K**2/vthermal**2)*np.exp(-3./2.)

		#Assign velocity initial distribution function
		v0 = np.zeros(N)
		v0 = np.random.normal(0.0, np.sqrt(kBTe/me),N)
		growth_rate = d_landau
	#end if

	if system=='two-stream':
		beam_1_proportion = N/2
		beam_2_proportion = N - beam_1_proportion
		beam_1_temperature = 1./40.
		beam_2_temperature = 1./40.
		beam_1_drift = 1.0
		beam_2_drift = 0.0

		v0 = np.zeros(N)
		v0[0:beam_1_proportion] = np.random.normal(beam_1_drift * np.sqrt(kBTe/me), beam_1_temperature * np.sqrt(kBTe/me), beam_1_proportion)
		v0[beam_1_proportion:] = np.random.normal(0.0, beam_2_temperature * np.sqrt(kBTe/mp), beam_2_proportion)
		m[beam_1_proportion:] = mp

		growth_rate = wp*(me/mp)**(1./3.)

	#end if

	#Apply perturbation by resampling particles from uniform distribution
	x0 = np.random.uniform(0., L+dx, N)

	F = -np.cos(Kp * np.pi * X / (L+dx)) + 1.0
	F = (N * perturbation) * F / np.sum(F)

	j = N/2 - int(N * perturbation / 2)
	for i in range(Ng):
		for k in range(int(F[i])):
			x0[j] = np.random.uniform(X[i],X[i+1])
			j += 1
		#end for
	#end for

	x0 = x0 % (L+dx)

	#Two-stream instability hot
	#v0[0:N/2] = np.random.normal(2.*np.sqrt(kBTe/me), np.sqrt(kBTe/me), N/2)
	#v0[N/2:N] = np.random.normal(-2.*np.sqrt(kBTe/me), np.sqrt(kBTe/me), N/2)

	return m,q,x0,v0,kBTe,growth_rate
#end def initialize

def main_i(T,nplot):
	tol = 1E-6
	maxiter = 20

	#Bump on tail best parameters
	#system = 'bump-on-tail'
	#density = 1e10 # [1/m3]
	#perturbation = 0.01
	#Kp = 1
	#N = 100000
	#Ng = 40

	#dt = 1E-8 #[s]
	#dx = 0.1	 #[m]

	#Ti = 0.1 * 11600. #[K]
	#Te = 2.0 * 11600. #[K]

	#Landau damping best params
	system = 'landau damping'
	density = 1e10 # [1/m3]
	perturbation = 0.05
	Kp = 2
	N = 40000
	Ng = 100
	dt = 1E-8 #[s]
	dx = 0.04	 #[m]
	Ti = 0.1 * 11600. #[K]
	Te = 10.0 * 11600. #[K]

	#Two-stream
	#system = 'two-stream'
	#density = 1e12 # [1/m3]
	#perturbation = 0.01
	#Kp = 2
	#N = 40000
	#Ng = 40

	#dt = 5E-9 #[s]
	#dx = 0.02	 #[m]

	#Ti = 0.1 * 11600. #[K]
	#Te = 2.0 * 11600. #[K]

	L = dx * (Ng-1)
	X = np.linspace(0.0,L+dx,Ng+1)

	wp = np.sqrt(e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * np.pi / (L+dx)
	p2c = (L+dx) * density / N

	m,q,x0,v0,kBTe,growth_rate = initialize(system,N,density,Kp,perturbation,dx,Ng,Te,L,X)

	print("wp : ",wp,"[1/s]")
	print("dt : ",dt/invwp," [w * tau]")
	print("tau: ",invwp,"[s]")
	print("k  : ",K,"[1/m]")
	print("p2c :", p2c)

	scattermap = plt.cm.viridis(1.0 - 2.0*np.sqrt(v0*v0)/np.max(np.sqrt(v0*v0)))
	#scattermap = plt.cm.viridis(v0/np.max(np.abs(v0)))

	E0 = np.zeros(Ng+1)
	Eh = np.zeros(Ng+1)
	E1 = np.zeros(Ng+1)
	Es = np.zeros(Ng+1)
	E_interp = np.zeros(N)

	xh = np.zeros(N)
	x1 = np.zeros(N)
	xs = np.zeros(N)

	vh = np.zeros(N)
	v1 = np.zeros(N)

	j0 = np.zeros(Ng+1)
	jh = np.zeros(Ng+1)
	j1 = np.zeros(Ng+1)

	rho0 = np.zeros(Ng+1)
	phi0 = np.zeros(Ng+1)

	#Guess E0 from Poisson Solve
	rho0 = weightDensitiesPeriodic(x0,q,p2c,Ng,N,dx)
	phi0 = solvePoissonPeriodicElectronsNeutralized(dx,Ng,rho0,1.0,1E-4,10,phi0)
	phi0 = phi0 - np.max(phi0)
	E0 = differentiateFieldPeriodic(phi0,dx,Ng)
	j0 = weightCurrentsPeriodic(x0,q,v0,p2c,Ng,N,dx)

	r = 1.0
	k = 0

	plt.figure(1)
	plt.ion()
	plt.figure(2)
	plt.ion()
	plt.figure(3)
	plt.ion()
	plt.figure(4)
	plt.ion()
	plt.figure(5)
	plt.ion()

	KE = []
	EE = []
	TT = []
	jbias = []

	for t in range(T+1):
		print('t: ',t)
		#Initial guess from n-step levels
		Es = E0
		xs = x0

		#Picard loop to find x1, v1, E1
		while (r>tol) & (k<maxiter):

			for i in range(N):
				E_interp[i] = interpolateFieldPeriodic(Eh,xh[i],Ng,dx)
			#end for

			x1 = x0 +  dt * v0 + dt * dt * (q/m) * E_interp*0.5
			v1 = v0 + dt * (q/m) * E_interp

			xh = (x0 + x1) * 0.5
			vh = (v0 + v1) * 0.5

			xh = xh % (L+dx)
			jh = weightCurrentsPeriodic(xh,q,vh,p2c,Ng,N,dx)

			x1 = x1 % (L+dx)
			j1 = weightCurrentsPeriodic(x1, q, v1, p2c, Ng, N, dx)

			E1 = E0 + (dt/epsilon0) * (np.average(jh) - jh)
			Eh = (E1 + E0) * 0.5

			r = np.linalg.norm(Es-Eh)

			Es = Eh
			xs = xh

			k += 1
		#end while
		print("Iterations: ",k)
		print("r: ",r)

		#Replace n-step values with n+1-step values
		E0 = E1
		x0 = x1
		v0 = v1
		j0 = j1
		k = 0
		r = 1.0

		#Update time-tracked values
		EE.append(np.sum(epsilon0 * E0*E0 / 2.))
		KE.append(np.sum(me * v0*v0 / 2.))
		TT.append(t * dt)
		jbias.append(np.average(j0))

		#Plotting routine
		if (t % nplot == 0):
			plt.figure(1)
			plt.clf()
			plt.scatter(x0,v0/np.sqrt(kBTe/me),s=0.5,color=scattermap)
			plt.title('Phase Space, Implicit')
			plt.axis([0.0, L+dx, -10., 10.])
			plt.xlabel('$x$ [$m$]')
			plt.ylabel('$v$ [$v_{thermal}$]')
			plt.xticks([0.0, L+dx])
			plt.yticks([-10.0, -5.0, 0.0, 5.0, 10.0])
			plt.draw()
			plt.savefig('plots/ps_'+str(t))
			plt.pause(0.0001)

			plt.figure(2)
			plt.clf()
			plt.plot(X,j0,linewidth=lw)
			plt.xticks([0.0, L+dx])
			plt.title('Current, Implicit')
			plt.xlabel('$x$ [$m$]')
			plt.ylabel(r'$J$ [$\frac{A}{m^{2}}$]')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(3)
			plt.clf()
			plt.plot(X,E0,linewidth=lw)
			plt.xticks([0.0, L+dx])
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
			if system == 'landau damping':
				plt.semilogy(np.array(TT)*wp,np.max(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
			else:
				plt.semilogy(np.array(TT)*wp,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
			plt.title('$E^{2}$, Implicit')
			plt.ylabel(r'$E^{2}$ [$\frac{V^{2}}{m^{2}}$]')
			plt.xlabel(r'$t$ [$\omega_{p}^{-1}$]')
			plt.legend([r'$E^{2}$','Theoretical'])
			plt.draw()
			plt.savefig('plots/e2_'+str(t))
			plt.pause(0.0001)
		#end if
	#end for
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
#end main_i

def main(T,nplot):
	tol = 1E-8
	maxiter = 20

	#Bump on tail best parameters
	#system = 'bump-on-tail'
	#density = 1e10 # [1/m3]
	#perturbation = 0.01
	#Kp = 1
	#N = 20000
	#Ng = 40

	#dt = 1E-8 #[s]
	#dx = 0.1	 #[m]
	#Ti = 0.1 * 11600. #[K]
	#Te = 2.0 * 11600. #[K]

	#Landau damping best params
	system = 'landau damping'
	density = 1e10 # [1/m3]
	perturbation = 0.05
	Kp = 2
	N = 100000
	Ng = 200
	dt = 1E-9 #[s]
	dx = 0.02	 #[m]
	Ti = 0.1 * 11600. #[K]
	Te = 10.0 * 11600. #[K]

	#Two-stream
	#system = 'two-stream'
	#density = 1e12 # [1/m3]
	#perturbation = 0.01
	#Kp = 2
	#N = 40000
	#Ng = 40
	#dt = 5E-9 #[s]
	#dx = 0.02	 #[m]
	#Ti = 0.1 * 11600. #[K]
	#Te = 2.0 * 11600. #[K]

	L = dx * (Ng-1)
	X = np.linspace(0.0,L+dx,Ng+1)

	wp = np.sqrt(e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * np.pi / (L+dx)
	p2c = (L+dx) * density / N

	m,q,x0,v0,kBTe,growth_rate = initialize(system,N,density,Kp,perturbation,dx,Ng,Te,L,X)

	x = x0
	v = v0

	print("wp : ",wp,"[1/s]")
	print("dt : ",dt/invwp," [w * tau]")
	print("tau: ",invwp,"[s]")
	print("k  : ",K,"[1/m]")
	print("p2c :", p2c)

	scattermap = plt.cm.viridis(1.0 - 2.0*np.sqrt(v0*v0)/np.max(np.sqrt(v0*v0)))
	#scattermap = plt.cm.viridis(v0/np.max(np.abs(v0)))

	E = np.zeros(Ng+1)
	phi = np.zeros(Ng+1)
	j = np.zeros(Ng+1)
	rho = weightDensitiesPeriodic(x,q,p2c,Ng,N,dx)
	rho0 = np.average(rho)

	plt.figure(1)
	plt.ion()
	plt.figure(2)
	plt.ion()
	plt.figure(3)
	plt.ion()
	plt.figure(4)
	plt.ion()
	plt.figure(5)
	plt.ion()

	EE = []
	KE = []
	TT  = []

	rho = weightDensitiesPeriodic(x,q,p2c,Ng,N,dx)
	phi = solvePoissonPeriodicElectronsNeutralized(dx,Ng,rho,kBTe,1E-3,20,phi)
	phi = phi - np.max(phi)
	E = differentiateFieldPeriodic(phi,dx,Ng)

	for t in range(T):
		print('t: ',t)

		#Update time-tracked values
		EE.append(np.sum(epsilon0 * E*E / 2.))
		KE.append(np.sum(me * v*v / 2.))
		TT.append(dt * t)

		#Plotting routine
		if (t % nplot == 0):
			plt.figure(1)
			plt.clf()
			plt.scatter(x0,v0/np.sqrt(kBTe/me),s=0.5,color=scattermap)
			plt.title('Phase Space, Explicit')
			plt.axis([0.0, L+dx, -10., 10.])
			plt.xlabel('$x$ [$m$]')
			plt.ylabel('$v$ [$v_{thermal}$]')
			plt.xticks([0.0, L+dx])
			plt.yticks([-10.0, -5.0, 0.0, 5.0, 10.0])
			plt.draw()
			plt.savefig('plots/ps_'+str(t))
			plt.pause(0.0001)

			plt.figure(2)
			plt.clf()
			plt.plot(X,j,linewidth=lw)
			plt.xticks([0.0, L+dx])
			plt.title('Current, Explicit')
			plt.xlabel('$x$ [$m$]')
			plt.ylabel(r'$J$ [$\frac{A}{m^{2}}$]')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(3)
			plt.clf()
			plt.plot(X,E,linewidth=lw)
			plt.xticks([0.0, L+dx])
			plt.xlabel('$x$ [$m$]')
			plt.ylabel(r'$E$ [$\frac{V}{m}$]')
			plt.title('Electric Field, Explicit')
			plt.draw()
			plt.savefig('plots/e_'+str(t))
			plt.pause(0.0001)

			plt.figure(4)
			plt.clf()
			plt.semilogy(np.array(TT)*wp,KE,linewidth=lw)
			plt.xlabel(r't [$\omega_{p}^{-1}$]')
			plt.ylabel('$KE$ [$J$]')
			plt.title('KE, Explicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(5)
			plt.clf()
			plt.semilogy(np.array(TT)*wp,EE,linewidth=lw)
			if system == 'landau damping':
				plt.semilogy(np.array(TT)*wp,np.max(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
			else:
				plt.semilogy(np.array(TT)*wp,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT),linewidth=lw)
			plt.title('$E^{2}$, Explicit')
			plt.ylabel(r'$E^{2}$ [$\frac{V^{2}}{m^{2}}$]')
			plt.xlabel(r'$t$ [$\omega_{p}^{-1}$]')
			plt.legend([r'$E^{2}$',r'$Theoretical$'],loc='lower left')
			plt.draw()
			plt.savefig('plots/e2_'+str(t))
			plt.pause(0.0001)
		#end if

		#PIC Loop
		rho = weightDensitiesPeriodic(x,q,p2c,Ng,N,dx)
		phi = solvePoissonPeriodicElectronsNeutralized(dx,Ng,rho,kBTe,1E-3,20,phi)
		phi = phi - np.max(phi)
		E = differentiateFieldPeriodic(phi,dx,Ng)
		x,v = pushParticlesExplicit(x,v,q,m,N,Ng,dt,dx,E)
		x,v = applyBoundaryConditionsPeriodic(x,v,m,N,L,dx,kBTe)
	#end for
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
	plt.figure(1)
	plt.savefig('plots/PS_I_m.png')
	plt.figure(5)
	plt.savefig('plots/E2_I_m.png')
#end main

if __name__ == '__main__':
	main()
