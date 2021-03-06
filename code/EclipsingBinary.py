import numpy as np
import time
from astropy import units, constants
from astropy.coordinates import SkyCoord
import pickle

#######################
#3rd party codes
#NOTE: ellc.lc is in arbitrary flux units... am I using this correctly?
import ellc

#could use this instead, seems to be linked more closely to astropy : https://dust-extinction.readthedocs.io/en/latest/index.html
#pip install git+https://github.com/karllark/dust_extinction.git
from dust_extinction.parameter_averages import F04

#for A_V
#import vespa.stars.extinction
from vespa_update import extinction

#python code radvel :https://radvel.readthedocs.io/en/latest/index.html
#https://radvel.readthedocs.io/_/downloads/en/latest/pdf/
#for a few angles
import radvel

######################
#my code
from SED import SED

class EclipsingBinary(object):
	def __init__(self, *args,**kwargs):

		self.MbolSun = 4.73 #as "recommended" for Flower's bolometric correction in http://iopscience.iop.org/article/10.1088/0004-6256/140/5/1158/pdf

		#these is defined by the user
		self.m1 = None #*units.solMass
		self.m2 = None #*units.solMass
		self.r1 = None #*units.solRad
		self.r2 = None #*units.solRad
		self.L1 = None #*units.solLum
		self.L2 = None #*units.solLum
		self.k1 = None #BSE stellar type
		self.k2 = None #BSE stellar type
		self.period = None #*units.day 
		self.eccentricity = None
		self.OMEGA = None
		self.omega = None
		self.inclination = None
		self.t_zero = None
		self.dist = None #*units.kpc
		self.xGx = None #*units.parsec
		self.yGx = None #*units.parsec
		self.zGx = None #*units.parsec
		self.verbose = False
		self.RV = 3.1
		self.M_H = 0. #metallicity
		self.TRILEGALrmag = -999. #r magnitude from TRILEGAL

		#for SED
		self.filterFilesRoot = '../input/filters/'
		self.SED1 = None
		self.SED2 = None

		#for Galaxy
		self.Galaxy = None
		self.SEDsingle = None

		#for star cluster crowding model
		self.crowding = None

		#from https://www.lsst.org/scientists/keynumbers
		#in nm
		self.wavelength = {
			'u_': (324. + 395.)/2.,
			'g_': (405. + 552.)/2.,
			'r_': (552. + 691.)/2.,
			'i_': (691. + 818.)/2.,
			'z_': (818. + 921.)/2.,
			'y_': (922. + 997. )/2.
		}

		#these will be calculated after calling self.initialize()
		self.RL1 = None
		self.RL2 = None
		self.T1 = None
		self.T2 = None
		self.T12 = None
		self.L1 = None
		self.L2 = None
		self.g1 = None
		self.g2 = None
		self.a = None 
		self.q = None
		self.f_c = None
		self.f_s = None
		self.R_1 = None
		self.R_2 = None
		self.sbratio = None
		self.RA = None
		self.Dec = None
		self.Mbol = None
		self.AV = None
		self.appMagMean = dict()
		self.Fv1 = dict()
		self.Fv2 = dict()
		self.appMagMeanAll = None
		self.Ared = dict()
		self.BC = dict()

		#for light curves
		self.SED = None
		self.OpSim = None
		self.filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']
		self.ld_1 = 'claret'
		self.ld_2 = 'claret'
		self.grid_1 = 'default'
		self.grid_2 = 'default'
		self.shape_1 = 'sphere'
		self.shape_2 = 'sphere'
		self.sigma_sys = 0.005  #systematic photometric error
		self.obsDates = dict()
		self.m_5 = dict()
		self.appMag = dict()
		self.appMagObs = dict()
		self.appMagObsErr = dict()
		self.deltaMag = dict()
		self.eclipseDepthFrac = dict()
		self.maxDeltaMag = 0.
		self.useOpSimDates = True
		self.observable = True
		self.appmag_failed = 0
		self.eclipseDepth_failed = 0
		self.incl_failed = 0
		self.period_failed = 0
		self.radius_failed = 0
		self.OpSimi = 0
		self.years = 10.
		self.totaltime = 365.* self.years
		self.cadence = 3.
		self.Nfilters = 6.
		self.nobs = 0
		self.light_3 = {}
		for f in self.filters:
			self.light_3[f] = 0.

		#this is for the magnitude uncertainties
		#https://arxiv.org/pdf/0805.2366.pdf
		self.sigmaDict = {
			'u_': {
				'gamma'	: 0.037,
				'seeing': 0.77,
				'm_sky'	: 22.9,
				'C_m' 	: 22.92,
				'k_m' 	: 0.451},
			'g_': {
				'gamma'	: 0.038,
				'seeing': 0.73,
				'm_sky'	: 22.3,
				'C_m'	: 24.29,
				'k_m'	:0.163},
			'r_': {
				'gamma'	: 0.039,
				'seeing': 0.70,
				'm_sky'	: 21.2,
				'C_m'	: 24.33,
				'k_m'	: 0.087},
			'i_': {
				'gamma'	: 0.039,
				'seeing': 0.67,
				'm_sky'	: 20.5,
				'C_m'	: 24.20,
				'k_m'	: 0.065},
			'z_': {
				'gamma'	: 0.040,
				'seeing': 0.65,
				'm_sky'	: 19.6,
				'C_m'	: 24.07,
				'k_m'	: 0.043},
			'y_': {
		#from Ivezic et al 2008 - https://arxiv.org/pdf/0805.2366.pdf - Table 2 (p26)
				'gamma'	: 0.0039,
				'seeing': 0.65, #not sure where this is from - not in Ivezic; still the z value
				'm_sky'	: 18.61,
				'C_m'	: 23.73,
				'k_m'	: 0.170}
		}


		self.magLims = np.array([15.8, 25.]) #lower and upper limits on the magnitude detection assumed for LSST: 15.8 = rband saturation from Science Book page 57, before Section 3.3; 24.5 is the desired detection limit

		self.eclipseDepthLim = 3. #depth / error

		#set within the "driver" code, for gatspy
		self.LSS = dict()
		self.LSSmodel = dict()
		self.LSM = -999.
		self.LSMmodel = None

		self.seed = None


	def getFlowerBCV(self, Teff):
		#from http://iopscience.iop.org/article/10.1088/0004-6256/140/5/1158/pdf
		#which updates/corrects from Flower, P. J. 1996, ApJ, 469, 355
		lt = np.log10(Teff)
		a = 0.
		b = 0.
		c = 0.
		d = 0.
		e = 0.
		f = 0.
		if (lt < 3.7):
			a = -0.190537291496456*10.**5.
			b =  0.155144866764412*10.**5.
			c = -0.421278819301717*10.**4.
			d =  0.381476328422343*10.**3.
		if (lt >= 3.7 and lt < 3.9):
			a = -0.370510203809015*10.**5.
			b =  0.385672629965804*10.**5.
			c = -0.150651486316025*10.**5.
			d =  0.261724637119416*10.**4.
			e = -0.170623810323864*10.**3.
		if (lt >= 3.9):
			a = -0.118115450538963*10.**6.
			b =  0.137145973583929*10.**6.
			c = -0.636233812100225*10.**5.
			d =  0.147412923562646*10.**5.
			e = -0.170587278406872*10.**4.
			f =  0.788731721804990*10.**2.


		BCV = a + b*lt + c*lt**2. + d*lt**3. + e*lt**4 + f*lt**5.

		return BCV

	#Some approximate function for deriving stellar parameters
	def getRad(self, logg, m):
		#g = GM/r**2
		g = 10.**logg * units.cm/units.s**2.
		r = ((constants.G*m*units.Msun/g)**0.5).decompose().to(units.Rsun).value
		return r

	def getRadOld(self, m):
		#(not needed with Katie's model, but included here in case needed later)
		#use stellar mass to get stellar radius (not necessary, read out by Katie's work)
		if (m > 1):  #*units.solMass
			eta = 0.57
		else:
			eta = 0.8
		return (m)**eta #* units.solRad (units.solMass)

	def getTeff(self, L, R):
		#use stellar radius and stellar luminosity to get the star's effective temperature
		logTeff = 3.762 + 0.25*np.log10(L) - 0.5*np.log10(R) 
		return 10.**logTeff
	
	def getlogg(self,m, L, T):
		#use stellar mass, luminosity, and effective temperature to get log(gravity)
		return np.log10(m) + 4.*np.log10(T) - np.log10(L) - 10.6071
	
	def getLum(self, m):
		#https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
		#(not needed with Katie's model, but included here in case needed later)
		#use stellar mass to return stellar luminosity (not necessary, read out by Katie's work)
		if (m<0.43):
			cons = 0.23
			coeff = 2.3
		if (m>=0.43 and m<2.0):
			cons = 1
			coeff = 4
		if (m>=2.0 and m<55.0):
			cons = 1.4
			coeff = 3.5
		if (m >= 55):
			cons= 32000
			coeff = 1
		return cons*(m**coeff)

	def getafromP(self, m1, m2, P):
		#returns the semimajor axis from the period and stellar masses
		return (((P**2.) * constants.G * (m1 + m2) / (4*np.pi**2.))**(1./3.)).decompose().to(units.AU)

	def Eggleton_RL(self, q,a):
		#Eggleton (1983) Roche Lobe
		#assuming synchronous rotation
		#but taking the separation at pericenter
		return a*0.49*q**(2./3.)/(0.6*q**(2./3.) + np.log(1. + q**(1./3.)))


	def setLightCurve(self, filt, t_vis=30., X=1., useDates=[None], useT0=None, light_3=None):
		
		def getSig2Rand(filt, magnitude, m_5 = [None]):
			#returns 2 sigma random error based on the pass band (y-values may be wonky - need to check for seeing and 
			# against others)
			#X = 1. #function of distance??
			#t_vis = 30. #seconds
			if (m_5[0] is None):
				m_5 = self.sigmaDict[filt]['C_m'] + (0.50*(self.sigmaDict[filt]['m_sky'] - 21.)) + (2.50*np.log10(0.7/self.sigmaDict[filt]['seeing'])) + (1.25*np.log10(t_vis/30.)) - (self.sigmaDict[filt]['k_m']*(X-1.))
			return (0.04 - self.sigmaDict[filt]['gamma'])*(10**(0.4*(magnitude - m_5))) + self.sigmaDict[filt]['gamma']*((10**(0.4*(magnitude - m_5)))**2)*(magnitude**2)

		# Function to get y-band LDCs for any Teff, logg, M_H
		# written by Andrew Bowen, Northwestern undergraduate, funded by LSSTC grant (summer 2018)
		def get_y_LDC(Teff, logg, M_H):
			
			# All filters/wavelength arrays
			SDSSfilters = ['u_','g_','r_','i_','z_', "J", 'H', "K" ]  #Only 2MASS/SDSS filters (8 in total)
			SDSSwavelength = np.array([354, 464, 621.5, 754.5, 870, 1220, 1630, 2190])
			y_wavelength = np.array(1004)
			
			# Getting coefficients from ELLC and appending them to specific coeff arrays
			a1_array = np.array([])
			a2_array = np.array([])
			a3_array = np.array([])
			a4_array = np.array([])
			# Gets LDCs for all filters
			for w,f in zip(SDSSwavelength, SDSSfilters):
				ldy_filt = ellc.ldy.LimbGravityDarkeningCoeffs(f)
				a1, a2, a3, a4, y = ldy_filt(Teff, logg, M_H)
				a1_array = np.append(a1_array, a1)
				a2_array = np.append(a2_array, a2)
				a3_array = np.append(a3_array, a3)
				a4_array = np.append(a4_array, a4)

			# Sets up interpolation for y-band for each coeff
			find_y_a1 = np.interp(y_wavelength, SDSSwavelength, a1_array)
			find_y_a2 = np.interp(y_wavelength, SDSSwavelength, a2_array)
			find_y_a3 = np.interp(y_wavelength, SDSSwavelength, a3_array)
			find_y_a4 = np.interp(y_wavelength, SDSSwavelength, a4_array)
			
			return find_y_a1, find_y_a2, find_y_a3, find_y_a4
	

		#in case the user did not initialize
		if (self.T1 is None):
			self.initialize()

		#limb darkenning
		# T1 = np.clip(self.T1, 3500., 50000.)
		# T2 = np.clip(self.T2, 3500., 50000.)
		# g1 = np.clip(self.g1, 0., 5.)
		# g2 = np.clip(self.g2, 0., 5.)
		# MH = np.clip(self.M_H, -2.5, 0.5) 
		#there is a complicated exclusion region in the limb darkening.  See testing/limbDarkening/checkClaret.ipynb .  
		#I could possibly account for that, but for now I will simply not use limb darkening in those cases.
		T1 = np.clip(self.T1, 3500., 40000.)
		T2 = np.clip(self.T2, 3500., 40000.)
		g1 = np.clip(self.g1, 0., 5.)
		g2 = np.clip(self.g2, 0., 5.)
		MH = np.clip(self.M_H, -5, 1.) 
		# print(T1, T2, g1, g2, self.g1, self.g2, self.M_H)
		if (filt == 'y_'):
			a1_1, a2_1, a3_1, a4_1 = get_y_LDC(T1, g1, MH)
			a1_2, a2_2, a3_2, a4_2 = get_y_LDC(T2, g2, MH)
		else:
			ldy_filt = ellc.ldy.LimbGravityDarkeningCoeffs(filt)
			a1_1, a2_1, a3_1, a4_1, y = ldy_filt(T1, g1, MH)
			a1_2, a2_2, a3_2, a4_2, y = ldy_filt(T2, g2, MH)
		ldc_1 = [a1_1, a2_1, a3_1, a4_1] 
		ldc_2 = [a1_2, a2_2, a3_2, a4_2]
		# print(ldc_1, ldc_2)
		#light curve
		# self.period = 5
		# self.inclination = 90
		# self.R_1 = 0.05
		# self.R_2 = 0.05
		# self.sbratio = 1.
		# self.q = 1.
		# print(self.t_zero, self.period, self.a, self.q,
		# 	self.R_1, self.R_2, self.inclination, self.sbratio)
		#This is in arbitrary units... H ow do we get this into real units??
		# print("calling ellc", filt, self.obsDates[filt], ldc_1, ldc_2, self.t_zero, self.period, self.a, self.q,
		# 		self.f_c, self.f_s, self.ld_1,  self.ld_2, self.R_1, self.R_2, self.inclination, self.sbratio, 
		# 		self.shape_1, self.shape_2, self.grid_1,self.grid_2)

		t_zero = self.t_zero
		if (useT0 is not None):
			t_zero = useT0
		dates = self.obsDates[filt]
		if (useDates[0] is not None):
			dates = useDates
		if (light_3 is None):
			light_3 = self.light_3[filt]
		#print('using light_3', filt, light_3)
		if (np.isfinite(ldc_1[0]) and np.isfinite(ldc_2[0])):
			lc = ellc.lc(dates, ldc_1=ldc_1, ldc_2=ldc_2, 
				t_zero=t_zero, period=self.period, a=self.a, q=self.q,
				f_c=self.f_c, f_s=self.f_s, ld_1=self.ld_1,  ld_2=self.ld_2,
				radius_1=self.R_1, radius_2=self.R_2, incl=self.inclination, sbratio=self.sbratio, 
				shape_1=self.shape_1, shape_2=self.shape_2, grid_1=self.grid_1,grid_2=self.grid_2, light_3=light_3) 

		else:
			print(f"WARNING: nan's in ldc filter={filt}, ldc_1={ldc_1}, T1={T1}, logg1={g1}, ldc_2={ldc_2}, T2={T2}, logg2={g2}, [M/H]={MH}")
			lc = ellc.lc(dates, 
				t_zero=t_zero, period=self.period, a=self.a, q=self.q,
				f_c=self.f_c, f_s=self.f_s, 
				radius_1=self.R_1, radius_2=self.R_2, incl=self.inclination, sbratio=self.sbratio,
				shape_1=self.shape_1, shape_2=self.shape_2, grid_1=self.grid_1,grid_2=self.grid_2, light_3=light_3)


		lc = lc/np.max(lc) #maybe there's a better normalization?

		if (min(lc) > 0):
			#this is mathematically the same as below
			# #let's redefine these here, but with the lc accounted for
			# absMag = self.MbolSun - 2.5*np.log10( (self.L1f[filt] + self.L2f[filt])*lc) #This may not be strictly correct?  Should I be using the Sun's magnitude in the given filter? But maybe this is OK because, L1f and L2f are in units of LSun, which is related to the bolometric luminosity?
			# self.appMag[filt] = absMag + 5.*np.log10(self.dist*100.) + self.Ared[filt]  #multiplying by 1000 to get to parsec units

			Fv = self.Fv1[filt] + self.Fv2[filt]
			self.appMag[filt] = -2.5*np.log10(Fv*lc) + self.Ared[filt] #AB magnitude 

			# plt.plot((self.obsDates[filt] % self.period), lc,'.')
			# plt.ylim(min(lc), max(lc))
			# plt.show()
			# plt.plot((self.obsDates[filt] % self.period), self.appMag[filt],'.', color='red')
			# plt.plot((self.obsDates[filt] % self.period), self.appMagMean[filt] - 2.5*np.log10(lc), '.', color='blue')
			# plt.ylim(max(self.appMag[filt]), min(self.appMag[filt]))
			# plt.show()
			# print( (self.appMagMean[filt] - 2.5*np.log10(lc)) - self.appMag[filt])
			# raise

			m_5 = [None]				
			if (self.useOpSimDates):
				if (self.m_5):
					m_5 = self.m_5[filt]
				else:
					m_5 = np.full(len(useDates), np.mean(self.OpSim.m_5[self.OpSimi][filt]))

			#Ivezic 2008, https://arxiv.org/pdf/0805.2366.pdf , Table 2
			sigma2_rand = getSig2Rand(filt, self.appMag[filt], m_5 = m_5)   #random photometric error
			self.appMagObsErr[filt] = ((self.sigma_sys**2.) + (sigma2_rand))**(1./2.)

			#now add the uncertainty onto the magnitude
			self.appMagObs[filt] = np.array([np.random.normal(loc=x, scale=sig) for (x,sig) in zip(self.appMag[filt], self.appMagObsErr[filt])])

			#self.deltaMag[filt] = abs(min(self.appMagObs[filt]) - max(self.appMagObs[filt]))
			#self.eclipseDepthFrac[filt] = abs(self.deltaMag[filt]/np.mean(self.appMagObsErr[filt]))
			maxpos = np.argmax(self.appMagObs[filt])
			base = -2.5*np.log10(Fv) + self.Ared[filt] #AB magnitude 
			self.deltaMag[filt] = abs(self.appMagObs[filt][maxpos] - base)
			self.eclipseDepthFrac[filt] = abs(self.deltaMag[filt]/self.appMagObsErr[filt][maxpos])

	def checkEclipse(self, r1,r2,rp,inclination):
		ratio = (r1 + r2)/rp #rp = 2a for circular orbits
		if (ratio <= 1):
			theta = np.arcsin(ratio)*180./np.pi
			min_incl = 90. - theta
			max_incl = 90. + theta
			if (inclination <= min_incl or inclination >= max_incl):
				return False
		return True
	
	def preCheckIfObservable(self):
		
		#check for eclipse (accounting for eccentricity)
		#primary
		tp = radvel.orbit.timetrans_to_timeperi(np.array([self.t_zero]), self.period, self.eccentricity, self.omega*np.pi/180.)
		ta = radvel.orbit.true_anomaly(np.array([self.t_zero]), tp, self.period, self.eccentricity)

		rp = self.a*(1. - self.eccentricity*self.eccentricity)/(1. + self.eccentricity*np.cos(ta))
		eclipse_pri = self.checkEclipse(self.r1, self.r2, rp, self.inclination)
		
		#secondary
		tp = radvel.orbit.timetrans_to_timeperi(np.array([self.t_zero]), self.period, self.eccentricity, self.omega*np.pi/180., secondary=True)
		ta = radvel.orbit.true_anomaly(np.array([self.t_zero]), tp, self.period, self.eccentricity)

		rp = self.a*(1. - self.eccentricity*self.eccentricity)/(1. + self.eccentricity*np.cos(ta))
		eclipse_sec = self.checkEclipse(self.r1, self.r2, rp, self.inclination)

		if (not eclipse_pri and not eclipse_sec):
			self.incl_failed = 1


		#check for overlap of radii at peri
		ta = 0.0
		rp1 = self.a*(1. - self.eccentricity*self.eccentricity)/(1. + self.eccentricity*np.cos(ta))
		rp2 = self.a*(1. - self.eccentricity*self.eccentricity)/(1. + self.eccentricity*np.cos(ta + np.pi))
		rp = rp1 + rp2
		ratio = (self.r1 + self.r2)/rp
		if (ratio > 1 or self.R_1*(1. - self.eccentricity) <= 0 or self.R_1*(1. - self.eccentricity) >=1 or self.R_2*(1. - self.eccentricity) <=0 or self.R_2*(1. - self.eccentricity) >= 1 or self.R_1e >=1 or self.R_2e >=1):
			self.radius_failed = 1

		if (self.useOpSimDates):
			#redefine the totaltime based on the maximum OpSim date range over all filters
			for filt in self.filters:
				#print("check filt, totaltime, obs", filt, self.totaltime, self.OpSim.obsDates[self.OpSimi])
				#print("check obs", filt, self.OpSim.obsDates[self.OpSimi][filt])
				if (self.OpSim.obsDates[self.OpSimi][filt][0] is not None):
					self.totaltime = max(self.totaltime, (max(self.OpSim.obsDates[self.OpSimi][filt]) - min(self.OpSim.obsDates[self.OpSimi][filt])))

		# if (self.period >= self.totaltime):
		# 	self.period_failed = 1
			

		if (self.radius_failed or self.period_failed or self.incl_failed):
			self.observable = False

		if (self.verbose):
				print("precheck observable", self.observable, self.radius_failed, self.period_failed, self.incl_failed)
	

	def magCheckIfObservable(self):

		if (self.appMagMean['r_'] <= self.magLims[0] or self.appMagMean['r_'] >= self.magLims[1]): #15.8 = rband saturation from Science Book page 57, before Section 3.3; 24.5 is the desired detection limit
			self.appmag_failed = 1
			self.observable = False

			#print("checking mag", self.appMagMean['r_'], self.observable)

		if (self.verbose):
			print("mag observable", self.observable)
			
	def initializeSeed(self):
		if (self.seed is None):
			np.random.seed()
		else:
			np.random.seed(seed = self.seed)

	def initialize(self):

		#should I initialize the seed here?  
		#No I am initializing it in LSSTEBworker.py
		#self.initializeSeed()

		self.q = self.m2/self.m1
		self.a = self.getafromP(self.m1*units.solMass, self.m2*units.solMass, self.period*units.day).to(units.solRad).value
		self.R_1 = self.r1/self.a
		self.R_2 = self.r2/self.a
		self.R_1e = self.r1/self.Eggleton_RL(self.m1/self.m2, self.a * (1. - self.eccentricity))
		self.R_2e = self.r2/self.Eggleton_RL(self.m2/self.m1, self.a * (1. - self.eccentricity))
		self.f_c = np.sqrt(self.eccentricity)*np.cos(self.omega*np.pi/180.)
		self.f_s = np.sqrt(self.eccentricity)*np.sin(self.omega*np.pi/180.)

		if (self.RA is None):
			coord = SkyCoord(x=self.xGx, y=self.yGx, z=self.zGx, unit='pc', representation='cartesian', frame='galactocentric')
			self.RA = coord.icrs.ra.to(units.deg).value
			self.Dec = coord.icrs.dec.to(units.deg).value

		for f in self.filters:
			self.appMagMean[f] = -999. 
			self.deltaMag[f] = None
			self.LSS[f] = -999.
			self.appMagObs[f] = [None]
			self.appMagObsErr[f] = [None]
			self.deltaMag[f] = 0.
			self.obsDates[f] = [None]
			self.eclipseDepthFrac[f] = 0.
		self.appMagMean['r_'] = self.TRILEGALrmag #for binaries that don't pass the preCheck

		self.maxDeltaMag = 0.

		self.preCheckIfObservable()
		if (self.observable):
			if (self.T1 is None): self.T1 = self.getTeff(self.L1, self.r1)
			if (self.T2 is None): self.T2 = self.getTeff(self.L2, self.r2)
			if (self.g1 is None): self.g1 = self.getlogg(self.m1, self.L1, self.T1)
			if (self.g2 is None): self.g2 = self.getlogg(self.m2, self.L2, self.T2)
			self.sbratio = (self.L2/self.r2**2.)/(self.L1/self.r1**2.)

			#one option for getting the extinction
			if (self.AV is None):
				count = 0
				while (self.AV is None and count < 100):
					count += 1
					self.AV = extinction.get_AV_infinity(self.RA, self.Dec, frame='icrs')
					
					if (self.AV is None):
						print("WARNING: No AV found", self.RA, self.Dec, self.AV, count)
						time.sleep(30)

			self.SED1 = SED()
			self.SED1.filters = self.filters
			self.SED1.filterFilesRoot = self.filterFilesRoot
			self.SED1.T = self.T1*units.K
			self.SED1.R = self.r1*units.solRad
			self.SED1.L = self.L1*units.solLum
			self.SED1.logg = self.g1
			self.SED1.M_H = self.M_H
			self.SED1.EBV = self.AV/self.RV #could use this to account for reddening in SED
			self.SED1.initialize()

			self.SED2 = SED()
			self.SED2.filters = self.filters
			self.SED2.filterFilesRoot = self.filterFilesRoot
			self.SED2.T = self.T2*units.K
			self.SED2.R = self.r2*units.solRad
			self.SED2.L = self.L2*units.solLum
			self.SED2.logg = self.g2
			self.SED2.M_H = self.M_H
			self.SED2.EBV = self.AV/self.RV #could use this to account for reddening in SED
			self.SED2.initialize()

			#estimate a combined Teff value, as I do in the N-body codes (but where does this comes from?)
			logLb = np.log10(self.L1 + self.L2)
			logRb = 0.5*np.log10(self.r1**2. + self.r2**2.)
			self.T12 = 10.**(3.762 + 0.25*logLb - 0.5*logRb)
			#print(self.L1, self.L2, self.T1, self.T2, self.T12)



			#account for reddening and the different filter throughput functions (currently related to a blackbody)
			self.appMagMeanAll = 0.

			#one option for getting the extinction
			#ext = F04(Rv=self.RV)
			ext = F04(Rv=self.RV)
			#a check
			# self.Ltest = self.SED.getL(self.T1*units.K, self.r1*units.solRad)
			#definitely necessary for Kurucz because these models are not normalized
			#for the bb  I'm getting a factor of 2 difference for some reason
			Lconst1 = self.SED1.getLconst()
			Lconst2 = self.SED2.getLconst()
			#print(np.log10(Lconst1), np.log10(Lconst2))
			for f in self.filters:
				#print(extinction.fitzpatrick99(np.array([self.wavelength[f]*10.]), self.AV, self.RV, unit='aa')[0] , ext(self.wavelength[f]*units.nm)*self.AV)
				#self.Ared[f] = extinction.fitzpatrick99(np.array([self.wavelength[f]*10.]), self.AV, self.RV, unit='aa')[0] #or ccm89
				self.Ared[f] = ext(self.wavelength[f]*units.nm)*self.AV

				self.Fv1[f] = self.SED1.getFvAB(self.dist*units.kpc, f, Lconst = Lconst1)
				self.Fv2[f] = self.SED2.getFvAB(self.dist*units.kpc, f, Lconst = Lconst2)
				Fv = self.Fv1[f] + self.Fv2[f]
				self.appMagMean[f] = -2.5*np.log10(Fv) + self.Ared[f] #AB magnitude 

				#print(self.wavelength[f], self.appMagMean[f], self.Ared[f], self.T1)

				self.appMagMeanAll += self.appMagMean[f]

			self.appMagMeanAll /= len(self.filters)

			#check if we can observe this based on the magnitude
			self.magCheckIfObservable()

		#if we're using OpSim, then get the field ID
		#get the field ID number from OpSim where this binary would be observed
		if (self.useOpSimDates and self.observable and self.OpSim.fieldID[0] is None):
			self.OpSim.setFieldID(self.RA, self.Dec)

		#this now covers the galaxy and the cluster, if available
		if (self.observable and self.crowding is not None):
			self.crowding.input_rFlux = self.Fv1['r_'] + self.Fv2['r_'] #NOTE; if I don't have a r_ band, this will crash
			self.crowding.getCrowding()
			for f in self.filters:
				self.light_3[f] = self.crowding.backgroundFlux[f]/(self.Fv1[f] + self.Fv2[f])
				



	def observe(self, filt, light_3=None):

		#get the observation dates
		if (self.useOpSimDates):
			#print("using OpSimDates...")
			#check if we already have the observing dates
			if (self.OpSim is not None):
				#print("have OpSim", self.OpSim.obsDates)
				if (filt in self.OpSim.obsDates[self.OpSimi]):
					self.obsDates[filt] = self.OpSim.obsDates[self.OpSimi][filt]
					self.m_5[filt] = self.OpSim.m_5[self.OpSimi][filt]

			#otherwise get them
			if (filt not in self.obsDates):
				#print("getting dates")
				self.obsDates[filt], self.m_5[filt] = self.OpSim.getDates(self.OpSim.fieldID[self.OpSimi], filt)
				#print("received dates", filt, self.obsDates[filt])

			if (self.verbose):
				print(f'observing with OpSim in filter {filt}, have {len(self.obsDates[filt])} observations')
		else:
			#print("not using OpSimDates...")
			nobs = int(round(self.totaltime / (self.cadence * self.Nfilters)))
			self.obsDates[filt] = np.sort(self.totaltime * np.random.random(size=nobs))

		self.nobs += len(self.obsDates[filt])
		#get the light curve, and related information
		if (self.obsDates[filt][0] is not None): 
			#print("calling light curve", filt)
			self.setLightCurve(filt, light_3=light_3)

			if (filt == 'r_'):
				if (self.eclipseDepthFrac[filt] < self.eclipseDepthLim):
					self.eclipseDepth_failed = 1
					self.observable = False

	def outputLCtoFile(self, fname):
		obs = dict()
		for filt in self.filters:
			obs[filt] = dict()
			obs[filt]['OpSimDates'] = self.obsDates[filt]
			obs[filt]['mag'] = self.appMag[filt]
			obs[filt]['magObs'] = self.appMagObs[filt]
			obs[filt]['e_magObs'] = self.appMagObsErr[filt]

		file = open(fname, 'wb')
		pickle.dump(obs, file)
		file.close()

