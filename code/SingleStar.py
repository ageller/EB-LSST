import time
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import independent
from cosmic.evolve import Evolve

from dust_extinction.parameter_averages import F04
import numpy as np
from astropy import units

from SED import SED

#for A_V
#import vespa.stars.extinction
from vespa_update import extinction

class getSingleStars(object):
	def __init__(self, age, Z, Nsing):
		self.age = age
		self.Z = Z
		self.Nsing = Nsing

		self.random_seed = 1234
		
		self.initialSingles = None
		self.evolvedSingles = None

		self.BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': -1000, 'bwind': 0.0, 'lambdaf': 0.5, 'mxns': 2.5, 'beta': 0.125, 'tflag': 1, 'acc2': 1.5, 'nsflag': 3, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': -3000, 'sigma': 265.0, 'gamma': -1.0, 'pisn': 45.0, 'natal_kick_array' : [-100.0,-100.0,-100.0,-100.0,-100.0,-100.0], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.4, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 2, 'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 0, 'bdecayfac' : 1}

	# create "singles" from wide binaries with 0Msun companions
	def Initial_Single_Sample(self):
		"""

		Creates and evolves a set of "single" = wide binaries with 0Msun comparions with given 
		age (to evolve to), number of stars, and  metallicity.

		"""
		# Initial (input) binares -- using sampler method from cosmic #1234 - random seed
		print("initial single input:",self.age, self.Z, self.Nsing)
		# InitialSingles, sampled_mass, n_sampled = InitialBinaryTable.sampler('multidim',\
		# 	[0,12], [0,12],self.random_seed,1, 'delta_burst', self.age, self.Z, self.Nsing)
		InitialSingles, sampled_mass, mass_binaries, n_singles, n_binaries = InitialBinaryTable.sampler('independent', \
			[0,12], [0,12], primary_model='kroupa93', ecc_model='uniform', SFH_model='delta_burst', \
			binfrac_model=1.0, component_age=self.age, met=self.Z, size=self.Nsing)

		#change the periods and secondary masses
		for i, row in InitialSingles.iterrows():
			InitialSingles.at[i,'mass2_binary'] = 0
			InitialSingles.at[i,'porb'] = 1e10

		#print(InitialSingles)
		if (len(InitialSingles) > self.Nsing):
			InitialSingles = InitialSingles[0:self.Nsing]
		if (len(InitialSingles) < self.Nsing):
			print('!!!!WARNING: to few singles', len(InitialSingles), self.Nsing)

		self.InitialSingles = InitialSingles

	# Evolving hard binaries from our initial binary table above
	def EvolveSingles(self):

		"""Takes Initial (hard) binaries from above and evolves them"""
		bpp, bcm, initC  = Evolve.evolve(initialbinarytable = self.InitialSingles, BSEDict = self.BSEDict)
		##################
		#we need to grab only the final values at the age of the cluster
		###############
		self.SinglesEvolved = bcm.loc[(bcm['tphys'] == self.age)]


#class to get the SED values for singles
class SingleStar(object):
	def __init__(self, *args,**kwargs):
		
		self.SED = None
		self.m = None #*units.solMass
		self.R = None #*units.solRad
		self.L = None #*units.solLum
		self.M_H = 0. #metallicity
		self.AV = None
		self.RA = None
		self.Dec = None
		self.dist = None #*units.kpc
		
		#these will be calculated
		self.T = None #Kelvin
		self.logg = None 
		self.Fv = dict()
		self.Ared = dict()
		self.BC = dict()
		self.appMagMean = dict()

		#don't touch these
		self.filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']
		self.filterFilesRoot = '../input/filters/'
		self.RV = 3.1
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
		
	#also in EclipsingBinary
	def getTeff(self, L, R):
		#use stellar radius and stellar luminosity to get the star's effective temperature
		logTeff = 3.762 + 0.25*np.log10(L) - 0.5*np.log10(R) 
		return 10.**logTeff

	#Some approximate function for deriving stellar parameters
	def getRad(self, logg, m):
		#g = GM/r**2
		g = 10.**logg * units.cm/units.s**2.
		r = ((constants.G*m*units.Msun/g)**0.5).decompose().to(units.Rsun).value
		return r

	def getlogg(self,m, L, T):
		#use stellar mass, luminosity, and effective temperature to get log(gravity)
		return np.log10(m) + 4.*np.log10(T) - np.log10(L) - 10.6071
	
	def initialize(self):
		if (self.R is None): self.R = self.getRad(self.logg, self.m)
		if (self.T is None): self.T = self.getTeff(self.L, self.R)
		if (self.logg is None): self.logg = self.getlogg(self.m, self.L, self.T)  
		#one option for getting the extinction
		if (self.AV is None):
			count = 0
			while (self.AV is None and count < 100):
				count += 1
				self.AV = extinction.get_AV_infinity(self.RA, self.Dec, frame='icrs')
				if (self.AV is None):
					print("WARNING: No AV found", self.RA, self.Dec, self.AV, count)
					time.sleep(30)
					
		#initialize the SED
		self.SED = SED()
		self.SED.filters = self.filters
		self.SED.filterFilesRoot = self.filterFilesRoot
		self.SED.T = self.T*units.K
		self.SED.R = self.R*units.solRad
		self.SED.L = self.L*units.solLum
		self.SED.logg = self.logg
		self.SED.M_H = self.M_H
		self.SED.EBV = self.AV/self.RV #could use this to account for reddening in SED
		self.SED.initialize()
		
		#one option for getting the extinction
		ext = F04(Rv=self.RV)
		Lconst = self.SED.getLconst()
		for f in self.filters:
			self.Ared[f] = ext(self.wavelength[f]*units.nm)*self.AV
			self.Fv[f] = self.SED.getFvAB(self.dist*units.kpc, f, Lconst = Lconst)
			self.appMagMean[f] = -2.5*np.log10(self.Fv[f]) + self.Ared[f] #AB magnitude 

