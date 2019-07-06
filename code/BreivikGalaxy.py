import numpy as np
import pandas as pd
import scipy.special as ss
import scipy.stats
import os
import math
import multiprocessing 

#This is copied from Katie's GxRealizationThinDisk.py, but here we've created a Class
#This will draw the binaries from the Galaxy realization
class BreivikGalaxy(object):

	def __init__(self, *args,**kwargs):
		self.verbose = False
		self.GalaxyFile = '../input/Breivik/dat_ThinDisk_12_0_12_0.h5' #for Katie's model
		self.GalaxyFileLogPrefix = '../input/Breivik/fixedPopLogCm_'

		self.n_bin = 100000
		self.n_cores = 4
		self.popID = '0012'
		self.seed = None

		#set in getKernel
		self.fixedPop = None
		self.sampleKernel = None

		self.datMinMax = dict()

	def setMinMax(self):
		def paramMinMax(dat):
			if (not isinstance(min(dat), str)):
				datMin = min(dat)-0.0001
				datMax = max(dat)+0.0001

				return [datMin, datMax]
			else:
				return [None, None]

		for x in list(self.fixedPop.keys()):
			self.datMinMax[x] = paramMinMax(self.fixedPop[x])


	def setKernel(self):
		print("getting Breivik kernel")

		def paramTransform(dat):
			datMin = min(dat)-0.0001
			datMax = max(dat)+0.0001
			datZeroed = dat-datMin

			datTransformed = datZeroed/(datMax-datMin)
			return datTransformed


		# CONSTANTS
		##############################################################################

		mTotDisk = 2.15*10**10

		##############################################################################
		# STELLAR TYPES - KW
		#
		#   0 - deeply or fully convective low mass MS star
		#   1 - Main Sequence star
		#   2 - Hertzsprung Gap
		#   3 - First Giant Branch
		#   4 - Core Helium Burning
		#   5 - First Asymptotic Giant Branch
		#   6 - Second Asymptotic Giant Branch
		#   7 - Main Sequence Naked Helium star
		#   8 - Hertzsprung Gap Naked Helium star
		#   9 - Giant Branch Naked Helium star
		#  10 - Helium White Dwarf
		#  11 - Carbon/Oxygen White Dwarf
		#  12 - Oxygen/Neon White Dwarf
		#  13 - Neutron Star
		#  14 - Black Hole
		#  15 - Massless Supernova
		##############################################################################


		# LOAD THE FIXED POPULATION DATA
		##############################################################################
		############## UNITS ################
		#                                   #
		# mass: Msun, porb: year, sep: rsun #
		#                                   #
		#####################################

				
		self.fixedPop = pd.read_hdf(self.GalaxyFile, key='bcm')
		FixedPopLog = np.loadtxt(self.GalaxyFileLogPrefix+self.popID+'.dat', delimiter = ',')
				
		# COMPUTE THE NUMBER AT PRESENT DAY NORMALIZED BY TOTAL MASS OF THE GX COMPONENT
		##############################################################################
		mTotFixed = sum(FixedPopLog[:,2])
		nPop = int(len(self.fixedPop)*mTotDisk/mTotFixed)
		print('The number of binaries in the Gx for: '+str(self.popID)+' is: '+str(nPop))
			
		# TRANSFORM THE FIXED POP DATA TO HAVE LIMITS [0,1] &
		# COMPUTE THE BINWIDTH TO USE FOR THE KDE SAMPLE; SEE KNUTH_BIN_WIDTH IN ASTROPY
		##############################################################################

		# UNITS: 
		# MASS [MSUN], ORBITAL PERIOD [LOG10(YEARS)], LUMINOSITIES [LSUN], RADII [RSUN]
		#self.fixedPop['m1'] = self.fixedPop['mass_1'] #or maybe some more efficient way
		###
		#print (self.fixedPop['m1'])

		#some changes to the names here because some were in the log but not names as such by Katie!
		self.fixedPop['m1'] = self.fixedPop['mass_1']
		#print (self.fixedPop['m1'])
		self.fixedPop['m2'] = self.fixedPop['mass_2']
		self.fixedPop['logL1'] = self.fixedPop['lumin_1']
		self.fixedPop['logL2'] = self.fixedPop['lumin_2']
		self.fixedPop['logr1'] = self.fixedPop['rad_1']
		self.fixedPop['logr2'] = self.fixedPop['rad_2']
		self.fixedPop['logp'] = np.log10(self.fixedPop['porb'])
		self.setMinMax()

		m1Trans = ss.logit(paramTransform(self.fixedPop['m1']))				
		m2Trans = ss.logit(paramTransform(self.fixedPop['m2']))				
		porbTrans = ss.logit(paramTransform(self.fixedPop['logp']))				
		Lum1Trans = ss.logit(paramTransform(self.fixedPop['logL1']))
		Lum2Trans = ss.logit(paramTransform(self.fixedPop['logL2']))
					
		# The eccentricity is already transformed, but only fit KDE to ecc if ecc!=0.0
		eIndex, = np.where(self.fixedPop['ecc']>1e-2)
		if len(eIndex) > 50:

			eccTrans = self.fixedPop['ecc']
			for jj in eccTrans.keys():
				if eccTrans[jj] > 0.999:
					eccTrans[jj] = 0.999
				elif eccTrans[jj] < 1e-4:
					eccTrans[jj] = 1e-4
			eccTrans = ss.logit(eccTrans)


		rad1Trans = ss.logit(paramTransform(self.fixedPop['logr1']))
		rad2Trans = ss.logit(paramTransform(self.fixedPop['logr2']))
				
		# GENERATE THE DATA LIST DEPENDING ON THE TYPE OF COMPACT BINARY TYPE
		##############################################################################
		type1Save = 1
		if type1Save < 14 and len(eIndex)>50:
			print('both bright stars and eccentric')
			datList = np.array((m1Trans, m2Trans, porbTrans, eccTrans, rad1Trans, rad2Trans, Lum1Trans, Lum2Trans))
		elif type1Save < 14 and len(eIndex)<50:
			print('both bright stars and circular')
			datList = np.array((m1Trans, m2Trans, porbTrans, rad1Trans, rad2Trans, Lum1Trans, Lum2Trans))
						
		# GENERATE THE KDE FOR THE DATA LIST
		##############################################################################
		if (self.verbose):
			print(datList)

		self.sampleKernel = scipy.stats.gaussian_kde(datList)#, bw_method=popBw)


	def GxSample(self, nSample, x=None, output=None, saveFile=False):
		def untransform(dat, datSet):
			datMin = min(datSet)
			datMax = max(datSet)
			
			datUntransformed = dat*(datMax-datMin)
			datUnzeroed = datUntransformed+datMin
			
			return datUnzeroed  
		
	  
		##### UNITS EXPECTED FOR POPULATION ###################
		# mass: Msun, orbital period: days, Tobs: seconds     #
		#######################################################
		# seed the random generator
		########################################
		np.random.seed()

		# solar coordinates in the galaxy: in parsecs from 
		# (Chaper 8 of Galactic Structure and stellar Pops book) Yoshii (2013)
		############################################################################
		x_sun = 8000.0
		y_sun = 0.0
		z_sun = 25
		
		
		# open the file to save the gx data
		#################################################
		binDat = [] 
		
		dataSample = self.sampleKernel.resample(nSample)
			
		# check to see if the population has BHs or is ecc
		###########################################################
		   
		m1T = ss.expit(dataSample[0,:])
		m2T = ss.expit(dataSample[1,:])
		porbT = ss.expit(dataSample[2,:])
		eccT = ss.expit(dataSample[3,:])
		rad1T = ss.expit(dataSample[4,:])
		rad2T = ss.expit(dataSample[5,:])
		Lum1T = ss.expit(dataSample[6,:])
		Lum2T = ss.expit(dataSample[7,:])
			
		m1 = untransform(m1T, self.fixedPop['m1'])
		m2 = untransform(m2T, self.fixedPop['m2'])
		logp = untransform(porbT, self.fixedPop['logp'])
		ecc = eccT             
		ii=0
		for e in ecc:
			if e < 0:
				ecc[ii] = abs(e)
			elif e > 1:
				ecc[ii] = 1-(ecc-1)
			ii+=1
		rad1 = 10**(untransform(rad1T, self.fixedPop['logr1']))
		rad2 = 10**(untransform(rad2T, self.fixedPop['logr2']))
		Lum1 = 10**(untransform(Lum1T, self.fixedPop['logL1']))
		Lum2 = 10**(untransform(Lum2T, self.fixedPop['logL2']))
		
		# COMPUTE THE POSITION AND ORIENTATION OF EACH BINARY
		##############################################################################
		# First assign the position relative to the galactic center
		# Assign the radial position of the binary
		a_0_r = np.random.uniform(0, 1.0, len(m1))
		r = -2.5*np.log(1-a_0_r)
		# Assign the azimuthal position of the star
		phi = np.random.uniform(0, 2*np.pi, len(m1))
		# Assign the z position of the star with r as a parameter
		a_0_zr = np.random.uniform(0, 1.0, len(m1))
		z = 1/1.42*np.arctanh(a_0_zr/(0.704)-1)
		# convert to cartesian and parsecs
		xGX = r*np.cos(phi)*1000.0
		yGX = r*np.sin(phi)*1000.0
		zGX = z*1000.0
		# compute the distance to Earth/LISA/us in kiloparsecs
		dist = ((xGX-x_sun)**2+(yGX-y_sun)**2+(zGX-z_sun)**2)**(1/2.0)
		dist_kpc = dist/1000.0
			
		inc = np.arccos(2.*np.random.uniform(0,1,len(m1)) - 1.)
		OMEGA = np.random.uniform(0,2*math.pi,len(m1))
		omega = np.random.uniform(0,2*math.pi,len(m1))

		binDat = np.vstack((m1, m2, logp, ecc, rad1, rad2, Lum1, Lum2, xGX, yGX, zGX, dist_kpc, inc, OMEGA, omega)).T		

		# radTotAU = (rad1+rad2)/rsun_in_au
		# radAng = radTotAU/dist
		# binEclipseIndex, = np.where(radAng>inc*4.8e-6)
	 
		if (saveFile):
			gxFile = 'gxRealization_'+str(x)+'_'+str(self.popID)+'.dat'
			np.savetxt(gxFile, binDat, delimiter = ',')     
		
		if (output == None):
			return pd.DataFrame(binDat, columns=['m1', 'm2', 'logp', 'ecc', 'r1', 'r2', 'L1', 'L2', 'xGX', 'yGX', 'zGX', 'dist_kpc', 'inc', 'OMEGA', 'omega'])
		else:	
			output.put(np.shape(binDat)) 




	def LSSTsim(self):

		self.setKernel()		
				
		# CALL THE MONTE CARLO GALAXY SAMPLE CODE
		##############################################################################
		print('nSample: '+str(self.n_bin))
		output = multiprocessing.Queue()
		nSample = int(self.n_bin/float(self.n_cores))
		processes = [multiprocessing.Process(target = self.GxSample, \
						args = (nSample, x, output, True)) \
						for x in range(self.n_cores)]
		for p in processes:
			p.start()

		for p in processes:
			p.join()

		nSRC = [output.get() for p in processes]
				
		print('The number of sources in pop '+self.popID+' is ', nSRC)

		gxDatTot = []
		for kk in range(self.n_cores):
			if os.path.getsize('gxRealization_'+str(kk)+'_'+str(self.popID)+'.dat') > 0:
				gxReal = np.loadtxt('gxRealization_'+str(kk)+'_'+str(self.popID)+'.dat', delimiter = ',')
			else:
				gxReal = []
			if len(gxReal)>0:
				gxDatTot.append(gxReal)
			os.remove('gxRealization_'+str(kk)+'_'+str(self.popID)+'.dat')
		gxDatSave = np.vstack(gxDatTot)

		return gxDatSave
