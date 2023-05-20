# Manxi (Maggie) Shi
# 2022 2023
# Lattice Boltzmann Method: Flow around a Cylinder

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def main():
	# Simulation parameters
	nx                     = 400    		# x lattice grid
	ny                     = 100    		# y lattice grid
	rho0                   = 1      		# average density
	Re 		       = 60			# Reynolds number
	tau 		       = 27/Re + 1/2		# relaxation time
	nt                     = 6000   		# number of timesteps
	plot 		       = True 			# plot in real time
	dpdx 		       = 1e-5			# pressure difference
	wall_boundary          = True           	# True if top and bottom boundaries are solid walls
	
	# Lattice speeds and weights
	n 	= 9
	idxs 	= np.arange(n)
	cxs 	= np.array([0,1,0,-1,0,1,-1,-1,1])
	cys 	= np.array([0,0,1,0,-1,1,1,-1,-1])
	weights = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])

	# Initial conditions
	f = np.ones((ny,nx,n))
	np.random.seed(42)
	X, Y = np.meshgrid(range(nx), range(ny))
	f += 0.01*np.random.randn(ny,nx,n)
	f[:,:,1] += 2 * (1+0.2*np.cos(2*np.pi*X/nx*4))

	rho = np.sum(f,2)
	for i in idxs:
		f[:,:,i] *= rho0 / rho
	
	# Cylinder boundary
	X, Y = np.meshgrid(range(nx), range(ny))
	cylinder = (X - nx/4)**2 + (Y - ny/2)**2 < (ny/8)**2

	# Set up plots
	c = mpl.colors.LinearSegmentedColormap.from_list("", ["black","green","white","deeppink","black"])
	plt.imshow(np.array([[0],[0]]),cmap=c)
	plt.colorbar(orientation="horizontal")

	# Simulation main loop
	for it in range(nt):
		# Stream
		for i, cx, cy in zip(idxs, cxs, cys):
			f[:,:,i] = np.roll(f[:,:,i], cx, axis=1) # rolling along x 1 grid
			f[:,:,i] = np.roll(f[:,:,i], cy, axis=0) # rolling along y 1 grid

		# Calculate macroscopic fluid variables
		rho = np.sum(f,2)
		ux  = np.sum(f*cxs,2) / rho
		uy  = np.sum(f*cys,2) / rho

		if wall_boundary: # solid walls at the top and bottom boundary
			# Calculate feq for non-walls
			feq = np.zeros(f.shape)
			for i, cx, cy, w in zip(idxs, cxs, cys, weights):
				feq[1:-1,:,i] = rho[1:-1,:] * w * (1 + 3*(cx*ux[1:-1,:]+cy*uy[1:-1,:])  + 9*(cx*ux[1:-1,:]+cy*uy[1:-1,:])**2/2 - 3*(ux[1:-1,:]**2+uy[1:-1,:]**2)/2)

			####### Top and bottom extrapolation boundary condition
			####### Long expansion but reduced for delta=1
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
			f[:,:,:] += source[:,:,:]

		else: # no walls at the top and bottom
			# Calculate feq everywhere
			feq = np.zeros(f.shape)
			for i, cx, cy, w in zip(idxs, cxs, cys, weights):
				feq[:,:,i] = rho * w * (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)
		
			# Collision
			f[:,:,:] += (1.0/tau) * (feq[:,:,:] - f[:,:,:])
		
		# Set reflective boundaries at cylinder walls
		bndryF = f[cylinder,:]
		bndryF = bndryF[:,[0,2,1,4,3,7,8,5,6]]
		f[cylinder,:] = bndryF
		

		if (plot and (it % 100) == 0) or (it == nt-1):
			plt.cla()
			ux[cylinder] = 0
			uy[cylinder] = 0
			
			# plt.subplot(3,1,1)
			vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
			vorticity[cylinder] = np.nan # rotation in a fluid
			vorticity = np.ma.array(vorticity, mask=cylinder)
			plt.imshow(vorticity, cmap=c)
			
			# x velocity
			# uxgraph = np.ma.array(ux, mask=cylinder)
			# plt.subplot(3,1,2)
			# plt.imshow(uxgraph, cmap=c)

			# y velocity
			# uygraph = np.ma.array(uy, mask=cylinder)
			# plt.subplot(3,1,3)
			# plt.imshow(uygraph, cmap=c)
			
			plt.pause(0.001)

	# save image to file and show
	filename = 'cylw' + str(round(Re)) + '.png'
	plt.savefig(filename, dpi=240)
	plt.show()

	return 0

if __name__== "__main__":
  main()
