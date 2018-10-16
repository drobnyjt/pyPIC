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

#from numba import jit

#enable garbage collection
gc.enable()

#physical constants
epsilon0 = 8.854E-12
e = 1.602E-19
mp = 1.67E-27
me = 9.11E-31
kb = 1.38E-23

def interpolateField(F,x,Ng,dx):
	index = int(np.floor(x/dx))

	wR = (x % dx) / dx
	wL = 1. - wR

	return wL * F[index] + wR * F[index+1]
#end interpolateField

def weightCurrents(x,q,v,p2c,Ng,N,dx,dt,active):
	j = np.zeros(Ng)

	index = np.floor(x/dx)
	wR = (x % dx) / dx
	wL = 1. - wR

	idx = (1./dx)

	for i in range(N):
		if active[i] == 1:
			ind = int(index[i])
			j[ind]   += q[i] * v[i] * p2c * wL[i] * idx
			j[ind+1] += q[i] * v[i] * p2c * wR[i] * idx
		elif active[i] == -1:
			pass
			#j[0] += q[i] * v[i] * p2c * idx
			j[0] += -q[i] * p2c / dt
		elif active[i] == 0:
			pass
			#j[-1] +=  q[i] * v[i] * p2c * idx
			j[-1] += q[i] * p2c / dt

		#j[0] = 0.0
		#j[-1] = 0.0
		#end if
	#end for

	return j
#end def weightCurrents

def weightDensities(x,q,p2c,Ng,N,dx,active):
	rho = np.zeros(Ng)

	index = np.floor(x/dx) % (Ng)

	wR = (x % dx) / dx
	wL = 1. - wR

	idx = (1./dx)

	for i in range(N):
		if active[i] == 1:
			ind = int(index[i])
			rho[ind] += q[i] * p2c * wL[i] * idx
			rho[ind+1] += q[i] * p2c * wR[i] * idx
		#end if
	#end for
	return rho
#end def weightDensities

def laplacian1DPeriodic(Ng):
	A =sp.diag(np.ones(Ng-1),-1) + sp.diag(-2.*np.ones(Ng),0) + sp.diag(np.ones(Ng-1),1)

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
	D = np.zeros((Ng,Ng))
	A = laplacian1DPeriodic(Ng)

	resid = 1.
	k = 0

	dx2 = dx*dx
	c0 = rho[(Ng)/2] / epsilon0
	c1 = e / kBT
	c2 = rho/epsilon0

	while (resid > tol) & (k <= maxiter):
		F = A.dot(phi) - dx2 * c0 * np.exp(c1 * (phi-phi[(Ng)/2]) ) + dx2 * c2

		np.fill_diagonal(D, -dx2 * c0 * c1 * np.exp( c1 * (phi-phi[(Ng)/2]) ))

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
	D = np.zeros((Ng,Ng))
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
		dF[i] = -(F[i+1] - F[i-1]) / dx * 0.5
	#end for

	dF[-1] = -(F[-1] - F[-2]) / dx
	dF[0]  = -(F[1] - F[0])   / dx

	return dF
#end def differentiateField

def integrateField(F,dx,Ng):
	IF = np.zeros(Ng)

	for i in range(0,Ng):
		IF[i] = -np.sum(F[:i+1]) * dx
	#end for

	return IF
#end def integrateField

def initialize(system,N,density,Kp,perturbation,dx,Ng,Te,Ti,L,X):
	wp = np.sqrt(e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * np.pi / (L)
	p2c = (L) * density / N
	kBTe = kb*Te #[J]
	kBTi = kb*Ti

	m = np.zeros(N)
	q = np.zeros(N)
	species = np.zeros(N)

	m[:N/2] =  np.ones(N/2) * me
	q[:N/2] = -np.ones(N/2) * e

	m[N/2:] = 1.0 * np.ones(N/2) * mp
	q[N/2:] = np.ones(N/2) * e

	species[:N/2] = 1
	species[N/2:] = 2

	#m = np.ones(N) * mp
	#q = np.ones(N) * e

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

	if system=='beam':
		vthermal = np.sqrt(kBTe/me)
		growth_rate = -np.sqrt(np.pi) * wp**4 / K**3 / np.sqrt(kBTe/me)**3 * np.exp(- wp**2 / K**2 / np.sqrt(kBTe/me)**2 * np.exp(-3./2.))
		print('Growth rate: ',growth_rate)
		x0 = np.random.uniform(0.0,L)

		u0 = np.zeros(N)
		v0 = np.zeros(N)
		w0 = np.zeros(N)

		u0[:N/2] = np.random.normal(0.0,np.sqrt(kBTe/m[:N/2]))
		u0[N/2:] = np.random.normal(0.0,np.sqrt(kBTi/m[N/2:]))

		v0[:N/2] = np.random.normal(0.0,np.sqrt(kBTe/m[:N/2]))
		v0[N/2:] = np.random.normal(0.0,np.sqrt(kBTi/m[N/2:]))

		w0[:N/2] = np.random.normal(0.0,np.sqrt(kBTe/m[:N/2]))
		w0[N/2:] = np.random.normal(0.0,np.sqrt(kBTi/m[N/2:]))
	#end if

	#Apply perturbation by resampling particles from uniform distribution
	x0 = np.random.uniform(0., L, N)

	F = -np.cos(Kp * np.pi * X / (L)) + 1.0
	F = (N * perturbation) * F / np.sum(F)

	j = N/2 - int(N * perturbation / 2)
	for i in range(Ng):
		for k in range(int(F[i])):
			x0[j] = np.random.uniform(X[i],X[i+1])
			j += 1
		#end for
	#end for

	#Two-stream instability hot
	#v0[0:N/2] = np.random.normal(2.*np.sqrt(kBTe/me), np.sqrt(kBTe/me), N/2)
	#v0[N/2:N] = np.random.normal(-2.*np.sqrt(kBTe/me), np.sqrt(kBTe/me), N/2)

	return m,q,x0,u0,v0,w0,species,kBTe,kBTi,growth_rate
#end def initialize

def main_i(T,nplot):
	tol = 1E-8
	maxiter = 100

	density = 1E19 # [1/m3]
	perturbation = 0.0
	Kp = 1.0
	N = 100000
	Ng = 51

	dt = 1E-12 #[s]
	dx = 0.00001	 #[m]

	Ti = 10.0 * 11600. #[K]
	Te = 10.0 * 11600. #[K]
	gamma = 0.001

	L = dx * (Ng-1)
	X = np.linspace(0.0,L,Ng)

	wp = np.sqrt(e**2 * density / epsilon0 / me)
	invwp = 1./wp
	K = Kp * np.pi / (L)
	p2c = (L) * density / N

	m,q,x0,u0,v0,w0,species,kBTe,kBTi,growth_rate = initialize('beam',N,density,Kp,perturbation,dx,Ng,Te,Ti,L,X)
	active = np.ones(N)

	print("wp : ",wp,"[1/s]")
	print("dt : ",dt/invwp," [w * tau]")
	print("tau: ",invwp,"[s]")
	print("k  : ",K,"[1/m]")
	print("p2c :", p2c)

	scattermap = plt.cm.viridis(1.0 - 2.0 * np.sqrt((u0*u0*m*0.5))/np.max(np.sqrt(u0*u0*m*0.5)))
	#scattermap = plt.cm.viridis()
	#scattermap = plt.cm.viridis(v0/np.max(np.abs(v0)))

	E0 = np.zeros(Ng)
	Eh = np.zeros(Ng)
	E1 = np.zeros(Ng)
	Es = np.zeros(Ng)
	E_interp = np.zeros(N)

	xh = np.zeros(N)
	x1 = np.zeros(N)
	xs = np.zeros(N)

	uh = np.zeros(N)
	u1 = np.zeros(N)

	vh = np.zeros(N)
	v1 = np.zeros(N)

	wh = np.zeros(N)
	w1 = np.zeros(N)

	j0 = np.zeros(Ng)
	jh = np.zeros(Ng)
	j1 = np.zeros(Ng)

	rho0 = np.zeros(Ng)
	phi0 = np.zeros(Ng)
	phih = np.zeros(Ng)
	phis = np.zeros(Ng)
	phi1 = np.zeros(Ng)

	#Guess E0 from Poisson Solve
	rho0 = weightDensities(x0,q,p2c,Ng,N,dx,active)
	#phi0 = solvePoissonPeriodicElectronsNeutralized(dx,Ng,rho0,1.0,1E-4,10,phi0)
	phi0 = phi0 - np.max(phi0)
	E0 = differentiateField(phi0,dx,Ng)
	j0 = weightCurrents(x0,q,u0,p2c,Ng,N,dx,dt,active)

	r = 1.0
	k = 0

	plt.figure(1)
	plt.ion()
	plt.figure(2)
	plt.ion()
	plt.figure(3)
	plt.ion()
	#plt.figure(4)
	#plt.ion()
	plt.figure(5)
	plt.ion()
	#plt.figure(6)
	#plt.ion()
	#plt.figure(7)
	#plt.ion()
	plt.figure(8)
	plt.ion()

	KE = []
	EE = []
	TT = []
	jbias = []
	vionout = []

	for t in range(T+1):
		print('t: ',t)

		#Thermostat
		for i in range(N):
			if active[i]==1 and np.random.uniform(0.0, 1.0) < gamma:
				u0[i] = np.random.normal(0.0,np.sqrt(kBTi/m[i]))
				v0[i] = np.random.normal(0.0,np.sqrt(kBTi/m[i]))
				w0[i] = np.random.normal(0.0,np.sqrt(kBTi/m[i]))
			#end if
		#end for

		#Particle Reinitialization
		for i in range(N):
			if active[i]!=1 and species[i]==2:
				#Reinitialize lost ions
				x0[i] = np.random.uniform(0.0,L)
				u0[i] = np.random.normal(0.0,np.sqrt(kBTi/m[i]))
				v0[i] = np.random.normal(0.0,np.sqrt(kBTi/m[i]))
				w0[i] = np.random.normal(0.0,np.sqrt(kBTi/m[i]))
				active[i] = 1
			#end if

			if active[i]!=1 and species[i] ==1:
			  #Reinitialize lost electrons
				x0[i] = np.random.uniform(0.0,L)
				u0[i] = np.random.normal(0.0,np.sqrt(kBTe/m[i]))
				v0[i] = np.random.normal(0.0,np.sqrt(kBTe/m[i]))
				w0[i] = np.random.normal(0.0,np.sqrt(kBTe/m[i]))
				active[i] = 1
			#end if
		#end for

		#Initial guess from n-step levels
		Es = E0
		xs = x0
		phis = integrateField(Es,dx,Ng)

		#Picard loop to find x1, v1, E1
		while (r>tol) & (k<maxiter):
			x1 = x1 * 0.0
			u1 = u1 * 0.0
			v1 = v1 * 0.0
			w1 = w1 * 0.0

			xh = xh * 0.0
			uh = uh * 0.0
			vh = vh * 0.0
			wh = wh * 0.0

			#Precalculate interpolated field quantities
			for i in range(N):
				if active[i]==1:
					E_interp[i] = interpolateField(Es,xs[i],Ng,dx)
				#end if
			#end for

			#Particle pusher
			for i in range(N):
				if active[i] == 1:
					x1[i] = x0[i] +  dt * u0[i] + dt * dt * (q[i]/m[i]) * E_interp[i]*0.5

					u1[i] = u0[i] + dt * (q[i]/m[i]) * E_interp[i]
					v1[i] = v0[i]
					w1[i] = w0[i]

					xh[i] = (x0[i] + x1[i]) * 0.5

					uh[i] = (u0[i] + u1[i]) * 0.5
					vh[i] = (v0[i] + v1[i]) * 0.5
					wh[i] = (w0[i] + w1[i]) * 0.5
				#end if
			#end for

			#Remove particles that leave domain
			for i in range(N):
				if active[i]==1 and (x0[i]>=L or xh[i]>=L or x1[i]>=L):
					active[i] = 0
					if i<N/2 and t > 2000:
						vionout.append(u0[i])
				#endif
				if active[i]==1 and (x0[i]<=0.0 or xh[i]<=0.0 or x1[i]<=0.0):
					active[i] = -1
					if i<N/2 and t > 2000:
						vionout.append(-u0[i])
				#end if
			#end for

			#Update field quantities in time
			#xh = xh % (L+dx)
			jh = weightCurrents(xh, q, uh, p2c, Ng, N, dx, dt, active)

			#x1 = x1 % (L+dx)
			j1 = weightCurrents(x1, q, u1, p2c, Ng, N, dx, dt, active)

			E1 = E0 + (dt/epsilon0) * (np.average(jh) - jh)
			phi1 = integrateField(E1,dx,Ng)
			phi1 = phi1 - np.max(phi1)

			Eh = (E1 + E0) * 0.5
			phih = integrateField(Eh,dx,Ng)
			phih = phih - np.max(phih)

			r = np.linalg.norm(Es-Eh)

			Es = Eh
			xs = xh
			phis = phih

			k += 1
			#gc.collect()
		#end while
		print("Iterations: ",k)
		print("r: ",r)

		#Replace n-step values with n+1-step values
		E0 = E1
		x0 = x1
		u0 = u1
		v0 = v1
		w0 = w1
		j0 = j1
		k = 0
		r = 1.0

		#Update time-tracked values
		EE.append(np.sum(epsilon0 * E0*E0 / 2.))
		KE.append(np.sum(me * u0*u0 / 2.))
		TT.append(t * dt)
		jbias.append(np.average(j0))

		#Plotting routine
		if (t % nplot == 0):
			plt.figure(1)
			plt.clf()
			plt.scatter(x0[N/2:],np.sign(u0[N/2:])*u0[N/2:]*u0[N/2:]*0.5*m[N/2:]/e,s=2.0,color=scattermap[N/2:])
			plt.title('Ion Phase Space')
			plt.axis([0.0, L, -100.0, 100.0])
			plt.xlabel('x [m]')
			plt.xticks(np.linspace(0.0,L,5))
			plt.yticks(np.linspace(-100.0,100.0,6))
			plt.ylabel('v [thermal]')
			plt.draw()
			plt.savefig('plots/ps_i_'+str(t))
			plt.pause(0.0001)

			plt.figure(2)
			plt.clf()
			plt.plot(X,j0,linewidth=lw)
			plt.title('J')
			plt.xlabel('x [m]')
			plt.xticks(np.linspace(0.0,L,5))
			plt.ylabel('J [A/m2]')
			plt.draw()
			plt.pause(0.0001)

			plt.figure(3)
			plt.clf()
			plt.plot(X,E0,linewidth=lw)
			plt.xlabel('x [m]')
			plt.ylabel('E [V/m]')
			#plt.title('Electric Field, Implicit')
			plt.draw()
			plt.savefig('plots/e_'+str(t))
			plt.pause(0.0001)

			#plt.figure(4)
			#plt.clf()
			#plt.semilogy(TT,KE)
			#plt.title('KE, Implicit')
			#plt.draw()
			#plt.pause(0.0001)

			plt.figure(5)
			plt.clf()
			plt.semilogy(TT,EE/np.max(EE),linewidth=lw)
			#plt.semilogy(TT,np.min(EE)*np.exp(np.ones(np.size(TT))*growth_rate * TT))
			plt.title('E2, Implicit')
			plt.xlabel('t [s]')
			plt.ylabel('E2 [A.U.]')
			plt.draw()
			plt.pause(0.0001)

			#plt.figure(6)
			#plt.clf()
			#plt.plot(jbias)
			#plt.draw()
			#plt.pause(0.0001)

			#plt.figure(7)
			#plt.clf()
			#plt.hist(vionout)
			#plt.draw()
			#plt.pause(0.0001)

			plt.figure(8)
			plt.clf()
			plt.scatter(x0[:N/2],np.sign(u0[:N/2])*u0[:N/2]*u0[:N/2]*0.5*m[:N/2]/e,s=0.5,color=scattermap[:N/2])
			plt.title('Electron Phase Space')
			plt.axis([0.0, L, -100.0, 100.0])
			plt.yticks(np.linspace(-100.0,100.0,6))
			plt.xlabel('x [m]')
			plt.xticks(np.linspace(0.0,L,5))
			plt.ylabel('v [thermal]')
			plt.draw()
			plt.savefig('plots/ps_e_'+str(t))
			plt.pause(0.0001)
		#end if
		gc.collect()
	#end for
	plt.figure(1)
	plt.savefig('PS_I_final.png')
	plt.figure(2)
	plt.savefig('E_final.png')
	#plt.figure(3)
	#plt.savefig('E.png')
	plt.figure(5)
	plt.savefig('E2_final.png')
	#plt.figure(6)
	#plt.savefig('JB_final.png')
	plt.figure(8)
	plt.savefig('PS_E_final.png')

	np.savetxt('vionout.txt',vionout)
	np.savetxt('E0.txt',E0)
	np.savetxt('jb.txt',jbias)
#end main_i

if __name__ == '__main__':
	main_i()
