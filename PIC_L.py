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

from numba import jit

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
#end interpolateField

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
		d_landau = -np.sqrt(np.pi) * wp**4 / K**3 / np.sqrt(kBTe/me)**3 * np.exp(- wp**2 / K**2 / np.sqrt(kBTe/me)**2 * np.exp(-3./2.))

		#Assign velocity initial distribution function
		v0 = np.zeros(N)
		v0 = np.random.normal(0.0, np.sqrt(kBTe/me),N)
	#end if

	if system=='two-stream':
		pass
	#end if

	m = np.ones(N) * me
	q =  -np.ones(N) * e

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
	tol = 1E-8
	maxiter = 6

	density = 1E11 # [1/m3]
	perturbation = 0.02
	Kp = 2
	N = 40000
	Ng = 20

	dt = 1E-8 #[s]
	dx = 0.05	 #[m]

	Ti = 0.1 * 11600. #[K]
	Te = 4.0 * 11600. #[K]

	L = dx * (Ng-1)
	X = np.linspace(0.0,L+dx,Ng+1)

	wp = np.sqrt(e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * np.pi / (L+dx)
	p2c = (L+dx) * density / N

	m,q,x0,v0,kBTe,growth_rate = initialize('bump-on-tail',N,density,Kp,perturbation,dx,Ng,Te,L,X)

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
	plt.figure(6)
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
			plt.scatter(x0,v0,s=0.5,color=scattermap)
			plt.title('Phase Space, Implicit')
			plt.axis([0.0, L, -np.sqrt(kBTe/me) * 10., np.sqrt(kBTe/me) * 10.])
			plt.draw()
			plt.savefig('plots/ps_'+str(t))
			plt.pause(0.0001)

			plt.figure(2)
			plt.clf()
			plt.plot(X,j0)
			plt.title('Current, Implicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(3)
			plt.clf()
			plt.plot(X,E0)
			plt.title('Electric Field, Implicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(4)
			plt.clf()
			plt.semilogy(TT,KE)
			plt.title('KE, Implicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(5)
			plt.clf()
			plt.semilogy(TT,EE)
			plt.semilogy(TT,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT))
			plt.title('E^2, Implicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(6)
			plt.clf()
			plt.plot(jbias)
			plt.draw()
			plt.pause(0.0001)
		#end if
	#end for
	np.savetxt('E2_I_m.txt',EE)
	plt.figure(1)
	plt.savefig('PS_I_m.png')
	plt.figure(5)
	plt.savefig('E2_I_m.png')
#end main_i

def main():
	tol = 1E-8
	maxiter = 50

	T = 10000
	density = 1E11
	perturbation = 0.02
	Kp = 2
	N = 80000
	Ng = 40
	nplot = int(N/10)

	dt = 5E-9
	dx = 0.08

	L = dx * (Ng-1)
	X = np.linspace(0.0,L+dx,Ng+1)

	wp = np.sqrt(e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * np.pi / (L+dx)

	print("wp : ",wp,"[1/s]")
	print("dt : ",dt/invwp," [tau]")
	print("tau: ",invwp,"[s]")
	print("k  : ",K,"[1/m]")

	p2c = (L+dx) * density / N
	print("p2c :", p2c)

	m = np.ones(N) * me
	q =  -np.ones(N) * e

	Ti = 0.1 * 11600. #[K]
	Te = 2.0 * 11600. #[K]

	kBTi = kb*Ti #[J]
	kBTe = kb*Te #[J]

	x0 = np.random.uniform(0., L+dx, N)

	F = -np.cos(Kp * np.pi * X / (L+dx)) + 1.0
	F = (N * perturbation) * F / np.sum(F)

	j = N/2 - N * int(perturbation / 2)
	for i in range(Ng):
		for k in range(int(F[i])):
			x0[j] = np.random.uniform(X[i],X[i+1])
			j += 1
		#end for
	#end for

	x0 = x0 % (L+dx)
	#v = np.random.normal(0.0, np.sqrt(kBTe/me),N)
	v0 = np.zeros(N)
	# v[0:N/2] = np.random.normal(4.*np.sqrt(kBTe/me), np.sqrt(kBTe/me), N/2)
	# v[N/2:N] = np.random.normal(-4.*np.sqrt(kBTe/me), np.sqrt(kBTe/me), N/2)

	#v0[0:N/2] = -1.*np.sqrt(kBTe/me)
	#v0[N/2:N] = 1.*np.sqrt(kBTe/me)

	v0[0:N*5/10] = np.random.normal(0.0, np.sqrt(kBTe/me), N)
	v0[N*5/10:] = np.random.normal(2.0 * np.sqrt(kBTe/me), 0.1 * np.sqrt(kBTe/me), N)
	#Landau Damping case works perfectly now
	#scattermap = plt.cm.viridis(np.sqrt(v0*v0)/np.max(np.sqrt(v0*v0)))

	x = x0
	v = v0

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
	plt.figure(6)
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

		#Plotting Routine
		if t % nplot ==0:
			scattermap = plt.cm.viridis(np.sqrt(rho0*rho0)/np.max(np.sqrt(rho0*rho0)))

			plt.figure(1)
			plt.clf()
			plt.scatter(x,v,s=0.2,color=scattermap)
			plt.title('Phase Space, Explicit')
			plt.axis([0.0, L+dx, -np.sqrt(kBTe/me) * 8., np.sqrt(kBTe/me) * 8.])
			plt.draw()
			plt.pause(0.0001)

			plt.figure(2)
			plt.clf()
			plt.plot(X,weightCurrentsPeriodic(x,q,v,p2c,Ng,N,dx))
			plt.title('Current, Explicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(3)
			plt.clf()
			plt.plot(X,E)
			plt.title('Electric Field, Explicit')
			plt.draw()

			plt.pause(0.0001)

			plt.figure(4)
			plt.clf()
			plt.plot(X,-rho/e)
			plt.title('Density, Explicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(5)
			plt.clf()
			plt.semilogy(TT,KE)
			plt.title('KE, Explicit')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(6)
			plt.clf()
			plt.semilogy(TT,EE)
			plt.title('E^2, Explicit')
			plt.draw()
			plt.pause(0.0001)
		#end if

		#PIC Loop
		rho = weightDensitiesPeriodic(x,q,p2c,Ng,N,dx)
		phi = solvePoissonPeriodicElectronsNeutralized(dx,Ng,rho,kBTe,1E-3,20,phi)
		phi = phi - np.max(phi)
		E = differentiateFieldPeriodic(phi,dx,Ng)
		x,v = pushParticlesExplicit(x,v,q,m,N,Ng,dt,dx,E)
		x,v = applyBoundaryConditionsPeriodic(x,v,m,N,L,dx,kBTi)
	#end for
	np.savetxt('E2_E_s.txt',EE)
	plt.figure(1)
	plt.savefig('PS_E_s.png')
	plt.figure(6)
	plt.savefig('E2_E_s.png')

#end main

if __name__ == '__main__':
	main()
