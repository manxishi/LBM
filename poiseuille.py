# Manxi (Maggie) Shi
# Nov 1 2022
# Lattice Boltzmann Method: Poiseuille Flow

import matplotlib.pyplot as plt
import numpy as np
import math

def main():
	nx                     = 400    # resolution x-dir
	ny                     = 100    # resolution y-dir
	rho0                   = 1    	# average density
	tau                    = 0.6    # collision timescale
	nu		       = 1/30	# kinematic viscosity
	nt 		       = 30000	# number of time steps
	dpdx 		       = 1e-6	# presure difference
	plot 		       = True 	# plot in real time

	# Lattice speeds / weights
	n 	= 9
	idxs 	= np.arange(n)
	cxs 	= np.array([0,1,0,-1,0,1,-1,-1,1])
	cys 	= np.array([0,0,1,0,-1,1,1,-1,-1])
	weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
	
	# Cylinder boundary
	X, Y = np.meshgrid(range(nx), range(ny))
	cylinder = (X - nx/4)**2 + (Y - ny/4)**2 < (ny/16)**2 # true for anything inside cylinder, false outside

	# Initial Conditions
	f = np.ones((ny,nx,n))
	# ny elements, within these elements nx elements, nested lists have 9 elements, one value for each direction
	# np.random.seed(42)
	# f += 0.01*np.random.randn(ny,nx,n) # adds a bit of randomness to f
	# f[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/nx*4))  # initial velocity condition, how the fluid comes in, only in direction 3 (positive x)
	
	rho = np.sum(f,2) # sum over all 9 directions
	for i in idxs:
		f[:,:,i] *= rho0 / rho # normalizing
	
	for it in range(nt):
		# Stream
		for i, cx, cy in zip(idxs, cxs, cys):
			f[:,:,i] = np.roll(f[:,:,i], cx, axis=1) # rolling along x 1 grid
			f[:,:,i] = np.roll(f[:,:,i], cy, axis=0) # rolling along y 1 grid

		# Calculate fluid variables
		rho = np.sum(f,2)
		if (it<15000):
			ux  = (np.sum(f*cxs,2) + dpdx/2) / rho # particple distribution function * x velocities, summed over 9 directions
		else:
			ux = np.sum(f*cxs,2) / rho
		uy  = np.sum(f*cys,2) / rho

		rho0 = rho
		ux0 = ux
		uy0 = uy

		# Calculate feq
		feq = np.zeros(f.shape)
		for i, cx, cy, w in zip(idxs, cxs, cys, weights):
			feq[1:-1,:,i] = rho[1:-1,:] * w * (1 + 3*(cx*ux[1:-1,:]+cy*uy[1:-1,:])  + 9*(cx*ux[1:-1,:]+cy*uy[1:-1,:])**2/2 - 3*(ux[1:-1,:]**2+uy[1:-1,:]**2)/2)


		####### top and bottom boundary condition
		####### long expansion but reduced for delta=1
		#		feq
		rhow1 = rho[1,:] # 400
		rhow2 = rho[-2,:] # 400
		tbFeq = [1,1,1,1,1,1,1,1,1]
		bbFeq = [1,1,1,1,1,1,1,1,1]
		for i, cx, cy, w in zip(idxs,cxs,cys,weights):
			tbFeq[i] = w*rhow1 # top bound feq, 9 lists of 400
			bbFeq[i] = w*rhow2 # bottom bound feq
		tbFeq = np.swapaxes(tbFeq,0,1)
		bbFeq = np.swapaxes(bbFeq,0,1)
		#		Fneq
		tbFneq = f[1,:,:] - feq[1,:,:]
		bbFneq = f[-2,:,:] - feq[-2,:,:]
		#		f = feq + Fneq
		f[0,:,:] = tbFeq + (1.0-1/tau)*tbFneq # 400 lists of 9
		f[-1,:,:] = bbFeq + (1.0-1/tau)*bbFneq

		# Collision
		f[1:-1,:,:] += (1.0/tau) * (feq[1:-1,:,:] - f[1:-1,:,:])

		# Add force term
		source = np.zeros(f.shape)
		for i, cx, cy, w in zip(idxs,cxs,cys,weights):
			source[:,:,i] = (1.0 - 0.5/tau) * w * (3.0 * (cx - ux) + 9.0 * (cx*ux + cy*uy)*cx)*dpdx
		if (it<15000):
			f[:,:,:] += source[:,:,:]
		
		rho1 = np.sum(f,2) # x y grid of densities at every lattice point
		ux1  = np.sum(f*cxs,2) / rho # particple distribution function * x velocities, summed over 9 directions
		uy1  = np.sum(f*cys,2) / rho

		# Calculate force, with ux uy before and after collision with cylinder
		fx = rho1*ux1 - rho0*ux0
		fy = rho1*uy1 - rho0*uy0
		fx = np.ma.array(fx, mask=cylinder) # 100x400, one fx for every lattice point
		fy = np.ma.array(fy, mask=cylinder) # 100x400
		netfx = fx.sum()
		netfy = fy.sum()

		with open('forces.csv','a') as file:
			file.write(str(it) + " " + str(netfx) + " " + str(netfy) + "\n")


		if (it%100==0):
			print(it)

		if (plot and (it % 100) == 0) or (it == nt-1):
			plt.cla()
			plt.xlabel("ux")
			plt.ylabel("y")

			# Analytical
			y = np.arange(ny)
			mu = nu*np.mean(rho,axis=1)*2.8
			u_true = -(1/(2*mu)*dpdx)*(y**2 - ny*y)
			plt.plot(u_true,y, color="Red")
			
			velprof = np.mean(ux,axis=1)
			plt.plot(velprof,y, color="Blue")

			err = np.zeros([nx,1])
			for i in range(1,nx):
				err[i] = sum(ux[:,i] - u_true)

			plt.pause(0.001)
			rms = (1/nx * sum(err**2))**0.5
			if it>9800:
				print(rms)
			if (rms<0.001):
				break
		
	plt.show()
	return 0

if __name__== "__main__":
  main()
