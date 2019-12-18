import numpy as np
import pandas as pd
from astropy import units
from dust_extinction.parameter_averages import F04

from SingleStar import SingleStar, getSingleStars

def getd2D(rPlummer, Nvals = 1):

	X1 = np.random.random(Nvals)
	X2 = np.random.random(Nvals)
	X3 = np.random.random(Nvals)
	zeta = (X1**(-2/3.) - 1.)**(-0.5)
	r = zeta*np.ones(Nvals)*rPlummer
	#this is a 3D r; we need to make this 2D
	z = 2.*r*X2 - r
	x = (r**2. - z**2.)**0.5*np.cos(X3*2.*np.pi)
	y = (r**2. - z**2.)**0.5*np.sin(X3*2.*np.pi)
	d2D = (x**2. + y**2.)**0.5
	
	return d2D, x, y, z

def gauss2D(A, x1, mu1, s1, x2, mu2, s2):
	#http://mathworld.wolfram.com/GaussianFunction.html
	return A/(2.*np.pi*s1*s2)*np.exp(-((x1 - mu1)**2./(2.*s1**2.) + (x2 - mu2)**2./(2.*s2**2.)))

class crowding(object):
	def __init__(self, *args,**kwargs):

		#required inputs for cluster crowding
		self.clusterRPlummer = None #pc 
		self.clusterAge = None #Myr
		self.clusterFeH = None
		self.clusterDist = None #kpc
		self.clusterAV = None
		self.clusterNstars = None 
		
		#for SED
		self.filterFilesRoot = '../input/filters/'

		#required for galaxy crowding
		self.Galaxy = None

		#optional inputs
		self.xBinary = None
		self.yBinary = None
		self.random_seed = None

		#best not to change these
		self.seeing = 0.5
		self.pixel = 0.2
		self.dLim = 3.

		self. RV = 3.1
		self.filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']
		self.wavelength = {
			'u_': (324. + 395.)/2.,
			'g_': (405. + 552.)/2.,
			'r_': (552. + 691.)/2.,
			'i_': (691. + 818.)/2.,
			'z_': (818. + 921.)/2.,
			'y_': (922. + 997. )/2.
		} 

		#outputs
		self.clusterSingles = None
		self.galaxySingles = None
		self.backgroundFlux = {}
		self.backgroundMag = {}
		self.xgrid = None
		self.ygrid = None
		self.fluxgrid = None
		self.nCrowd = 0
		self.nCrowdGalaxy = 0
		self.nCrowdCluster = 0
		self.AV = 0

	def generateClusterSingles(self):

		#draw random positions and sort by distance
		nCrowd = 0

		#draw from a Plummer star clusters distribution
		if (self.xBinary is None):
			d2D, self.xBinary, self.yBinary, self.zBinary = getd2D(self.clusterRPlummer)

		d2d, x, y, z = getd2D(self.clusterRPlummer, int(np.round(self.clusterNstars)) )
		dpc = ((self.xBinary - x)**2. + (self.yBinary - y)**2.)**0.5
		dAng = np.array(np.arctan2(dpc, self.clusterDist*1000.))*180./np.pi*3600.

		#take only those within the limits
		use = np.where(dAng < self.dLim*self.seeing)
		xSingles = x[use]
		ySingles = y[use]
		zSingles = z[use]

		#sort by distance from the center
		d = np.array((xSingles**2. + ySingles**2. + zSingles**2.)**0.5)
		srt = np.argsort(d)
		xSinglesAng = np.arctan2(xSingles[srt] - self.xBinary, self.clusterDist*1000.)*180./np.pi*3600.
		ySinglesAng = np.arctan2(ySingles[srt] - self.yBinary, self.clusterDist*1000.)*180./np.pi*3600.

		self.nCrowdCluster = len(xSinglesAng)

		self.nCrowd += self.nCrowdCluster

		if (self.nCrowdCluster > 0):
			#generate single stars with COSMIC (actually wide binaries)
			sampler = getSingleStars(self.clusterAge, self.clusterFeH, self.nCrowdCluster)
			sampler.random_seed = self.random_seed
			sampler.Initial_Single_Sample()
			sampler.EvolveSingles()

			#sort by mass (a crude way to get mass segregation)
			singles = sampler.SinglesEvolved.sort_values(by='mass_1', ascending=False)
			singles['xAng'] = xSinglesAng
			singles['yAng'] = ySinglesAng

			singles['M_H'] = np.ones(self.nCrowdCluster)*self.clusterFeH
			singles['dist'] = np.ones(self.nCrowdCluster)*self.clusterDist #kpc
			singles['AV'] = np.ones(self.nCrowdCluster)*self.clusterAV
			self.clusterSingles = self.getSinglesFlux(singles)

	def generateGalaxySingles(self):

		self.nCrowdGalaxy = int(np.random.poisson(self.Galaxy.starsPerResEl*self.dLim**2.))

		self.nCrowd += self.nCrowdGalaxy

		if (self.nCrowdGalaxy > 0):

			singles = pd.dataFrame()

			#take a uniform distribution 
			singles['xAng'] = np.random.random(size = self.nCrowdGalaxy)*self.dLim*self.seeing - self.dLim*self.seeing/2.
			singles['yAng'] = np.random.random(size = self.nCrowdGalaxy)*self.dLim*self.seeing - self.dLim*self.seeing/2.

			crowd = self.Galaxy.model.sample(self.nCrowdGalaxy)
			singles['M_H'] = crowd['[M/H]']
			singles['dist'] = 10.**crowd['logDist'] #kpc
			singles['AV'] = crowd['AV']
			singles['mass_1'] = crowd['Mact']
			singles['logg'] = crowd['logg']
			singles['rad_1'] = self.getRad(crowd['logg'], crowd['Mact'])
			singles['lumin_1'] = 10.**crowd['logL']
			singles['teff_1'] = 10.**crowd['logTe']

			self.galaxySingles = self.getSinglesFlux(singles)	

	def getSinglesFlux(self, singles):
		#get the fluxes
		flux = {}
		for f in self.filters:
			flux[f] = []
		
		for index, star in singles.iterrows():
			#sample a random star
			s = SingleStar()
			s.filterFilesRoot = self.filterFilesRoot
			s.M_H = star['M_H']
			s.dist = star['dist']
			s.m = star['mass_1']
			s.L = star['lumin_1']
			if ('teff_1' in star): s.T = star['teff_1']
			if ('logg' in star): s.logg = star['logg']
			if ('rad_1' in star): s.R = star['rad_1']
			if ('AV' in star): s.AV = star['AV']
			s.initialize()
			#print(s.m, s.Fv, s.appMagMean)
			for f in self.filters:
				flux[f].append(s.Fv[f])
				
		for f in self.filters:
			singles['flux_'+f] = flux[f]

		return singles 

	def integrateFlux(self):

		self.fluxgrid = {}

		def sumFlux(singles):
			for index, star in singles.iterrows():
				for f in self.filters:
					amp = star['flux_'+f]
					self.fluxgrid[f][i,j] += gauss2D(amp, x, star['xAng'], self.seeing, y, star['yAng'], self.seeing)
					if ('AV' in star): 
						self.AV += star['AV']

		#integrate up the flux in all the pixels within the seeing area
		extent = int(np.ceil(self.seeing/2./self.pixel))
		xvals = np.linspace(-extent*self.pixel, extent*self.pixel, 2*extent+1)
		yvals = np.linspace(-extent*self.pixel, extent*self.pixel, 2*extent+1)
		zvals = np.zeros((len(xvals), len(yvals)))
		self.xgrid, self.ygrid = np.meshgrid(xvals, yvals)

		for f in self.filters:
			self.fluxgrid[f] = zvals

		for i,x in enumerate(xvals):
			for j,y in enumerate(yvals):
				if (self.clusterSingles is not None): sumFlux(self.clusterSingles)
				if (self.galaxySingles is not None): sumFlux(self.galaxySingles)


		dx = xvals[1] - xvals[0]
		dy = yvals[1] - yvals[0]
		dA = dx*dy
		ext = F04(Rv=self.RV)
		if (self.clusterAV is not None):
			self.AV = self.clusterAV
		else:
			#for the AV when it's just the field here, let's just take a mean
			self.AV /= self.nCrowd

		for f in self.filters:
			Ared = ext(self.wavelength[f]*units.nm)*self.AV

			self.backgroundFlux[f] = np.sum(self.fluxgrid[f]*dA)
			self.backgroundMag[f] = -2.5*np.log10(self.backgroundFlux[f]) + Ared



	def getCrowding(self):
		print("getting crowding ... ")
		self.nCrowd = 0

		if (self.random_seed is not None):
			np.random.seed(seed = self.random_seed)

		for f in self.filters:
			self.backgroundFlux[f] = 0.
			self.backgroundMag[f] = 999.

		if (self.clusterRPlummer is not None):
			self.generateClusterSingles()

		if (self.Galaxy is not None):
			self.generateGalaxySingles()

		if (self.nCrowd > 0):
			self.integrateFlux()	

		print('crowding Ncluster, Ngalaxy', self.nCrowdCluster, self.nCrowdGalaxy)
		print('crowding mag', self.backgroundMag)			
