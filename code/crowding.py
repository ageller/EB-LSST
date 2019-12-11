import numpy as np
from astropy import units
from dust_extinction.parameter_averages import F04

from SingleStar import SingleStar, getSingleStars

def getd2D(rPlummer):

	X1 = np.random.random(len(rPlummer))
	X2 = np.random.random(len(rPlummer))
	X3 = np.random.random(len(rPlummer))
	zeta = (X1**(-2/3.) - 1.)**(-0.5)
	r = zeta*rPlummer
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

		#required inputs
		self.age = None #Myr
		self.FeH = None
		self.dist = None #pc
		self.AV = None

		#optional inputs
		self.rPlummer = None #pc
		self.Mcl = None #mSun
		self.meanMass = 0.5 #mSun
		self.Nsing = None
		self.xBinary = 0.
		self.yBinary = 0.
		self.random_seed = 1234

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
		self.singles = None
		self.backgroundFlux = None
		self.backgroundMag = None
		self.xgrid = None
		self.ygrid = None
		self.fluxgrid = None

	def generateSingles(self):


		#draw random positions and sort by distance
		np.random.seed(seed = self.random_seed)

		if (self.rPlummer != None):
			#draw from a Plummer star clusters distribution
			Nstars = int(round(self.Mcl/self.meanMass))

			d2d, x, y, z = getd2D(np.ones(int(np.round(Nstars)))*self.rPlummer)
			dpc = ((self.xBinary - x)**2. + (self.yBinary - y)**2.)**0.5
			dAng =  np.array(np.arctan2(dpc, self.dist))*180./np.pi*3600.

			#take only those within the limits
			use = np.where(dAng < self.dLim*self.seeing)
			xSingles = x[use]
			ySingles = y[use]
			zSingles = z[use]

			#sort by distance from the center
			d = np.array((xSingles**2. + ySingles**2. + zSingles**2.)**0.5)
			srt = np.argsort(d)
			xSinglesAng = np.arctan2(xSingles[srt] - self.xBinary, self.dist)*180./np.pi*3600.
			ySinglesAng = np.arctan2(ySingles[srt] - self.yBinary, self.dist)*180./np.pi*3600.

			self.Nsing = len(xSinglesAng)


		else:
			#take a uniform distribution 
			xSinglesAng = np.random.random(size = self.Nsing)*self.dLim*self.seeing - self.dLim*self.seeing/2.
			ySinglesAng = np.random.random(size = self.Nsing)*self.dLim*self.seeing - self.dLim*self.seeing/2.

		#generate single stars with COSMIC (actually wide binaries)
		sampler = getSingleStars(self.age, self.FeH, self.Nsing)
		sampler.random_seed = self.random_seed
		sampler.Initial_Single_Sample()
		sampler.EvolveSingles()

		#sort by mass (a crude way to get mass segregation)
		self.singles = sampler.SinglesEvolved.sort_values(by='mass_1', ascending=False)
		self.singles['xAng'] = xSinglesAng
		self.singles['yAng'] = ySinglesAng

		#get the fluxes
		flux = {}
		for f in self.filters:
			flux[f] = []
		
		for index, star in self.singles.iterrows():
			#sample a random star
			s = SingleStar()
			s.M_H = self.FeH
			s.dist = self.dist/1000. #kpc
			s.AV = self.AV

			s.m = star['mass_1']
			s.R = star['rad_1']
			s.L = star['lumin_1']
			s.T = star['teff_1']
			s.initialize()
			#print(s.m, s.Fv, s.appMagMean)
			for f in self.filters:
				flux[f].append(s.Fv[f])
				
		for f in self.filters:
			self.singles['flux_'+f] = flux[f]    


	def integrateFlux(self):
		#integrate up the flux in all the pixels within the seeing area
		extent = int(np.ceil(self.seeing/2./self.pixel))
		xvals = np.linspace(-extent*self.pixel, extent*self.pixel, 2*extent+1)
		yvals = np.linspace(-extent*self.pixel, extent*self.pixel, 2*extent+1)
		zvals = np.zeros((len(xvals), len(yvals)))
		self.xgrid, self.ygrid = np.meshgrid(xvals, yvals)
		self.fluxgrid = {}
		for f in self.filters:
			self.fluxgrid[f] = zvals
		for i,x in enumerate(xvals):
			for j,y in enumerate(yvals):
				for index, star in self.singles.iterrows():
					for f in self.filters:
						amp = star['flux_'+f]
						self.fluxgrid[f][i,j] += gauss2D(amp, x, star['xAng'], self.seeing, y, star['yAng'], self.seeing)

		dx = xvals[1] - xvals[0]
		dy = yvals[1] - yvals[0]
		dA = dx*dy
		self.backgroundFlux = {}
		self.backgroundMag = {}
		ext = F04(Rv=self.RV)
		for f in self.filters:
			Ared = ext(self.wavelength[f]*units.nm)*self.AV

			self.backgroundFlux[f] = np.sum(self.fluxgrid[f]*dA)
			self.backgroundMag[f] = -2.5*np.log10(self.backgroundFlux[f]) + Ared

	def getCrowding(self):
		self.generateSingles()
		self.integrateFlux()
				