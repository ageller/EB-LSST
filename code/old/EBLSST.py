## ellc.lc is in arbitrary flux units... am I using this correctly?

from datetime import datetime
import os
import math
import scipy.special as ss
import scipy.stats
from scipy.interpolate import interp1d
from scipy.integrate import quad
import multiprocessing 
import logging
import numpy as np
import os
import time
import pandas as pd
import csv
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#OpSim (in OpSimRun1) - http://ops2.lsst.org/docs/current/architecture.html
import sqlite3

import astropy.stats as astroStats
from astropy import units, constants
from astropy.coordinates import SkyCoord, ICRS
from astropy.io import fits

#pysynphot from STScI
#https://pysynphot.readthedocs.io/en/latest/index.html#pysynphot-installation-setup
#need to download grid of models manually, and set path accordingly
import pysynphot as pyS
# import os
# #export PYSYN_CDBS=/my/local/dir/cdbs/
# print(os.environ['PYSYN_CDBS'])

#3rd party codes
import ellc
import gatspy
from gatspy import datasets, periodic
from gatspy.periodic import LombScargleMultiband, LombScargle, LombScargleFast, LombScargleMultibandFast
import emcee

#for TRILEGAL and maybe also A_V
import vespa
from vespa_update import trilegal as trilegal_update
import subprocess
p = os.environ['PATH']
pv = os.path.join(os.getcwd(),'vespa_update')
p2 = pv+':'+p
os.environ['PATH'] = p2
#check that it recognizes the correct paths and executables
# print(f"PATH={os.environ['PATH']}")
# cmd = 'echo $PATH'
# proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
# print(proc.communicate())
# cmd = 'which get_trilegal'
# proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
# print(proc.communicate())

#extinction will allow me to convert A_V to any wavelength.  Not sure which reference is best.  I will use ccm89, for now. 
#import extinction

#could use this instead, seems to be linked more closely to astropy : https://dust-extinction.readthedocs.io/en/latest/index.html
#pip install git+https://github.com/karllark/dust_extinction.git
from dust_extinction.parameter_averages import F04

filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']


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

		#for SED
		self.filterFilesRoot = '../input/filters/'
		self.SED1 = None
		self.SED2 = None

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
		self.filters = filters
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
		self.maxDeltaMag = 0.
		self.useOpSimDates = True
		self.observable = True
		self.appmag_failed = 0
		self.incl_failed = 0
		self.period_failed = 0
		self.radius_failed = 0
		self.OpSimi = 0
		self.years = 10.
		self.totaltime = 365.* self.years
		self.cadence = 3.
		self.Nfilters = 6.
		self.nobs = 0
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
	def getRad(self, m):
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
		#(not needed with Katie's model, but included here in case needed later)
		#use stellar mass to return stellar luminosity (not necessary, read out by Katie's work)
		if (m<0.43):
			cons = 0.23
			coeff = 2.3
		if m>0.43 and m<2.0:
			cons = 1
			coeff = 4
		if m>2.0 and m<20.0:
			cons = 1.5
			coeff = 3.5
		else:
			cons= 3200
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


	def setLightCurve(self, filt, t_vis=30., X=1.):
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
			SDSSfiltVals = np.array([])
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
	
		self.appMagObs[filt] = [None]
		self.deltaMag[filt] = [None]

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
		if (np.isfinite(ldc_1[0]) and np.isfinite(ldc_2[0])):
			lc = ellc.lc(self.obsDates[filt], ldc_1=ldc_1, ldc_2=ldc_2, 
				t_zero=self.t_zero, period=self.period, a=self.a, q=self.q,
				f_c=self.f_c, f_s=self.f_s, ld_1=self.ld_1,  ld_2=self.ld_2,
				radius_1=self.R_1, radius_2=self.R_2, incl=self.inclination, sbratio=self.sbratio, 
				shape_1=self.shape_1, shape_2=self.shape_2, grid_1=self.grid_1,grid_2=self.grid_2) 
		else:
			print(f"WARNING: nan's in ldc filter={filt}, ldc_1={ldc_1}, T1={T1}, logg1={g1}, ldc_2={ldc_2}, T2={T2}, logg2={g2}, [M/H]={MH}")
			lc = ellc.lc(self.obsDates[filt], 
				t_zero=self.t_zero, period=self.period, a=self.a, q=self.q,
				f_c=self.f_c, f_s=self.f_s, 
				radius_1=self.R_1, radius_2=self.R_2, incl=self.inclination, sbratio=self.sbratio,
				shape_1=self.shape_1, shape_2=self.shape_2, grid_1=self.grid_1,grid_2=self.grid_2)

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
				m_5 = self.m_5[filt]
			#Ivezic 2008, https://arxiv.org/pdf/0805.2366.pdf , Table 2
			sigma2_rand = getSig2Rand(filt, self.appMag[filt], m_5 = m_5)   #random photometric error
			self.appMagObsErr[filt] = ((self.sigma_sys**2.) + (sigma2_rand))**(1./2.)

			#now add the uncertainty onto the magnitude
			self.appMagObs[filt] = np.array([np.random.normal(loc=x, scale=sig) for (x,sig) in zip(self.appMag[filt], self.appMagObsErr[filt])])

			self.deltaMag[filt] = abs(min(self.appMagObs[filt]) - max(self.appMagObs[filt]))


	def checkIfObservable(self):

		if (self.appMagMean['r_'] <= 15.8 or self.appMagMean['r_'] >= 24.5): #15.8 = rband saturation from Science Book page 57, before Section 3.3; 24.5 is the desired detection limit
			self.appmag_failed = 1
		
		ratio = (self.r1 + self.r2)/(2.*self.a)
		if (ratio <= 1):
			theta = np.arcsin(ratio)*180./np.pi
			min_incl = 90. - theta 
			max_incl = 90. + theta
			if (self.inclination <= min_incl or self.inclination >= max_incl):
				self.incl_failed = 1
		else:
			self.radius_failed = 1

		if (self.useOpSimDates):
			#redefine the totaltime based on the maximum OpSim date range over all filters
			for filt in filters:
				#print("check filt, totaltime, obs", filt, self.totaltime, self.OpSim.obsDates[self.OpSimi])
				#print("check obs", filt, self.OpSim.obsDates[self.OpSimi][filt])
				if (self.OpSim.obsDates[self.OpSimi][filt][0] is not None):
					self.totaltime = max(self.totaltime, (max(self.OpSim.obsDates[self.OpSimi][filt]) - min(self.OpSim.obsDates[self.OpSimi][filt])))

		if (self.period >= self.totaltime):
			self.period_failed = 1
				
		if (self.R_1 <= 0 or self.R_1 >=1 or self.R_2 <=0 or self.R_2 >= 1 or self.R_1e >=1 or self.R_2e >=1):
			self.radius_failed = 1
			
		if (self.radius_failed or self.period_failed or self.incl_failed or self.appmag_failed):
			self.observable = False

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
		self.T1 = self.getTeff(self.L1, self.r1)
		self.T2 = self.getTeff(self.L2, self.r2)
		self.g1 = self.getlogg(self.m1, self.L1, self.T1)
		self.g2 = self.getlogg(self.m2, self.L2, self.T2)
		self.a = self.getafromP(self.m1*units.solMass, self.m2*units.solMass, self.period*units.day).to(units.solRad).value
		self.f_c = np.sqrt(self.eccentricity)*np.cos(self.omega*np.pi/180.)
		self.f_s = np.sqrt(self.eccentricity)*np.sin(self.omega*np.pi/180.)
		self.R_1 = (self.r1/self.a)
		self.R_2 = (self.r2/self.a)
		self.sbratio = (self.L2/self.r2**2.)/(self.L1/self.r1**2.)
		self.R_1e = self.r1/self.Eggleton_RL(self.m1/self.m2, self.a * (1. - self.eccentricity))
		self.R_2e = self.r2/self.Eggleton_RL(self.m2/self.m1, self.a * (1. - self.eccentricity))

		self.SED1 = SED()
		self.SED1.filterFilesRoot = self.filterFilesRoot
		self.SED1.T = self.T1*units.K
		self.SED1.R = self.r1*units.solRad
		self.SED1.L = self.L1*units.solLum
		self.SED1.logg = self.g1
		self.SED1.M_H = self.M_H
		self.SED1.EBV = self.AV/self.RV #could use this to account for reddening in SED
		self.SED1.initialize()

		self.SED2 = SED()
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

		if (self.RA is None):
			coord = SkyCoord(x=self.xGx, y=self.yGx, z=self.zGx, unit='pc', representation='cartesian', frame='galactocentric')
			self.RA = coord.icrs.ra.to(units.deg).value
			self.Dec = coord.icrs.dec.to(units.deg).value

		#account for reddening and the different filter throughput functions (currently related to a blackbody)
		self.appMagMeanAll = 0.

		#one option for getting the extinction
		if (self.AV is None):
			self.AV = vespa.stars.extinction.get_AV_infinity(self.RA, self.Dec, frame='icrs')
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

			self.LSS[f] = -999.
			self.appMagMeanAll += self.appMagMean[f]
		self.appMagMeanAll /= len(self.filters)

		#check if we can observe this (not accounting for the location in galaxy)
		self.checkIfObservable()

		#if we're using OpSim, then get the field ID
		#get the field ID number from OpSim where this binary would be observed
		if (self.useOpSimDates and self.observable and self.OpSim.fieldID[0] is None):
			self.OpSim.setFieldID(self.RA, self.Dec)

	def observe(self, filt):

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
		self.setLightCurve(filt)

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
class OpSim(object):

	def __init__(self, *args,**kwargs):
		self.dbFile = '../input/db/minion_1016_sqlite.db' #for the OpSim database

		self.verbose = False
		self.fieldCursor = None
		self.summaryCursor = None

		self.fieldID = [None]
		self.RA = [None]
		self.Dec = [None]
		self.Nobs = [None]
		self.m_5 = [None]
		self.obsDates = [None]
		self.NobsDates = [None]
		self.totalNobs = [None]

		self.obsDist = None

	#database manipulation
	def getCursors(self):
		#gets SQlite cursor to pull information from OpSim
		#https://www.lsst.org/scientists/simulations/opsim/summary-table-column-descriptions-v335
		#http://ops2.lsst.org/docs/current/architecture.html
		db = sqlite3.connect(self.dbFile)
		cursor = db.cursor()

		cursor.execute("SELECT fieldid, expDate, filter, fiveSigmaDepth FROM summary") 
		self.summaryCursor = np.array(cursor.fetchall()) #NOTE: this takes a LONG time
		print("have summary cursor.")

		cursor.execute("SELECT fieldid, fieldra, fielddec FROM field")
		self.fieldCursor = np.array(cursor.fetchall()) #NOTE: this takes a LONG time
		print("have field cursor.")


	#For OpSim database
	def setFieldID(self, myRA, myDEC, deglim = 3.5/2.):
		#uses RA/Dec (from galactic coordinates) to return locatiom's fieldID according to OpSim
		#field-of-view == 3.5-degree diameter (also returned with fieldFov key)

		RA = self.fieldCursor[:,1].astype(float)
		Dec = self.fieldCursor[:,2].astype(float)
		dbCoord = SkyCoord(ra = RA*units.degree, dec = Dec*units.degree, frame='icrs')
		inCoord = SkyCoord(ra = myRA*units.degree, dec = myDEC*units.degree, frame='icrs')

		imin, sep2d, dist3d = inCoord.match_to_catalog_sky(dbCoord)

		dbID = (self.fieldCursor[imin,0]).astype('int') 

		mask = np.where(sep2d.to(units.degree).value > deglim)
		#this check apparently isn't necessary because it looks like the entire sky is covered with fieldIDs, but I suppose some of these fieldIDs don't have any observation dates (in the northern hemisphere)
		if (len(mask[0]) > 0):
			print(mask[0])
			print("WARNING: coordinate outside LSST FOV", myRA[mask], myDec[mask])
			dbID[mask] = -999

		if (self.verbose):
			print("have Field ID", dbID)

		self.fieldID = [dbID]

	def getDates(self, ID, filtin):
		#matches FieldID to existing OpSim database ID and matches observation filters to get dates (in seconds since the start of the 
		# survey)
		FieldID = self.summaryCursor[:,0].astype('int')
		date = self.summaryCursor[:,1].astype('float')
		filt = self.summaryCursor[:,2]
		fiveSigmaDepth = self.summaryCursor[:,3].astype('float')

		posIDFilt = np.where(np.logical_and(FieldID == ID, filt == filtin[:-1]))
		if (self.verbose):
			print("posIDFilt = ", posIDFilt, filtin)

		OpSimdates = posIDFilt[0]

		if (len(OpSimdates) < 1):
			return [None], [None]
		else:
			if (self.obsDist is None): #default, use the OpSim dates
				if (self.verbose):
					print('OpSimdates =', OpSimdates)
				dates = np.array([float(d) for d in date[OpSimdates] ])/86400. #converting seconds to days\
				m_5 = np.array([float(x) for x in fiveSigmaDepth[OpSimdates] ])
			else: #alternatively, can use a CDF of dt values to construct dates
				N = round(self.obsDist[filtin]['Nobs'])
				dt = []
				m5 = []
				for i in range(N):
					y = np.random.random()
					dt.append(10.**np.interp(y, self.obsDist[filtin]['cdf'], self.obsDist[filtin]['bins']))
				dates = np.cumsum(np.array(dt))
				mpos = np.random.randint(0, high=len(OpSimdates), size=N)
				m_5 = np.array([float(x) for x in fiveSigmaDepth[OpSimdates[mpos]] ])
				#print("in getDate", dates, m_5)
			return dates, m_5

	def setDates(self, i, filters):
		self.obsDates[i] = dict()
		self.NobsDates[i] = dict()
		self.m_5[i] = dict()
		self.totalNobs[i] = 0
		for filt in filters:
			self.obsDates[i][filt], self.m_5[i][filt] = self.getDates(self.fieldID[i], filt)
			#print("in setDates", i, filt, self.obsDates[i][filt])

			self.NobsDates[i][filt] = 0
			if (self.obsDates[i][filt][0] is not None):
				self.NobsDates[i][filt] = len(self.obsDates[i][filt])
			self.totalNobs[i] += self.NobsDates[i][filt]
			if (self.verbose):
				print(f'observing with OpSim in filter {filt}, have {self.NobsDates[i][filt]} observations')

	def getAllOpSimFields(self):
		print("getting OpSim fields...")
		self.getCursors()
		FieldID = self.summaryCursor[:,0].astype('int')
		date = self.summaryCursor[:,1].astype('float')

		self.fieldID = np.array([])
		self.RA = np.array([])
		self.Dec = np.array([])
		self.Nobs = np.array([])

		for x in self.fieldCursor:
			inS = np.where(FieldID == int(x[0]))[0]
			self.Nobs = np.append(self.Nobs, len(inS))
			self.fieldID = np.append(self.fieldID, x[0])
			self.RA = np.append(self.RA, x[1])
			self.Dec = np.append(self.Dec, x[2])
		self.obsDates = np.full_like(self.RA, dict(), dtype=dict)
		self.NobsDates = np.full_like(self.RA, dict(), dtype=dict)
		self.totalNobs = np.full_like(self.RA, 0)
		self.m_5 = np.full_like(self.RA, dict(), dtype=dict)

		print(f'returned {len(self.fieldID)} fields')
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

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

		# SET TIME TO TRACK COMPUTATION TIME
		##############################################################################
		start_time = time.time()
		#np.random.seed()

		# CONSTANTS
		##############################################################################
		G = 6.67384* 10.**-11.0
		c = 2.99792458* 10.**8.0
		parsec = 3.08567758* 10.**16.
		Rsun = 6.955* 10.**8.
		Msun = 1.9891* 10.**30.
		day = 86400.0
		rsun_in_au = 215.0954
		day_in_year = 365.242
		sec_in_day = 86400.0
		sec_in_hour = 3600.0
		hrs_in_day = 24.0
		sec_in_year = 3.15569*10**7.0
		Tobs = 3.15569*10**7.0
		geo_mass = G/c**2
		m_in_AU = 1.496*10**11.0

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
				
		dts = {'names':('binNum','tBornBin','Time','tBorn1','tBorn2','commonEnv','id1','id2',
				'm1','m2','m1Init','m2Init','Lum1','Lum2','rad1','rad2',
				'T1','T2','massc1','massc2','radc1','radc2','menv1','menv2',
				'renv1','renv2','spin1','spin2','rrol1','rrol2','porb','sep','ecc'),
			'formats':('i','f','f','f','f','i','i','i',
				'f','f','f','f','f','f','f','f',
				'f','f','f','f','f','f','f','f',
				'f','f','f','f','f','f','f','f','f')}
				
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
		bwM1 = astroStats.scott_bin_width(m1Trans)
				
		m2Trans = ss.logit(paramTransform(self.fixedPop['m2']))
		bwM2 = astroStats.scott_bin_width(m2Trans)
				
		porbTrans = ss.logit(paramTransform(self.fixedPop['logp']))
		bwPorb = astroStats.scott_bin_width(porbTrans)
				
		Lum1Trans = ss.logit(paramTransform(self.fixedPop['logL1']))
		bwLum1 = astroStats.scott_bin_width(self.fixedPop['logL1'])

		Lum2Trans = ss.logit(paramTransform(self.fixedPop['logL2']))
		bwLum2 = astroStats.scott_bin_width(self.fixedPop['logL2'])
					
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
			bwEcc = astroStats.scott_bin_width(eccTrans)
		else:
			bwEcc = 100.0

		rad1Trans = ss.logit(paramTransform(self.fixedPop['logr1']))
		bwRad1 = astroStats.scott_bin_width(rad1Trans)

		rad2Trans = ss.logit(paramTransform(self.fixedPop['logr2']))
		bwRad2 = astroStats.scott_bin_width(rad2Trans)
		#print(bwEcc,bwPorb,bwM1,bwM2,bwLum1,bwLum2,bwRad1,bwRad2)
		#popBw = min(bwEcc,bwPorb,bwM1,bwM2,bwLum1,bwLum2,bwRad1,bwRad2)
				
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

	  
		# CONSTANTS
		##############################################################################
		G = 6.67384*math.pow(10, -11.0)
		c = 2.99792458*math.pow(10, 8.0)
		parsec = 3.08567758*math.pow(10, 16)
		Rsun = 6.955*math.pow(10, 8)
		Msun = 1.9891*math.pow(10,30)
		day = 86400.0
		rsun_in_au = 215.0954
		day_in_year = 365.242
		sec_in_day = 86400.0
		sec_in_hour = 3600.0
		hrs_in_day = 24.0
		sec_in_year = 3.15569*10**7.0
		geo_mass = G/c**2
		m_in_AU = 1.496*10**11.0
		
	  
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
		
		# sample the number of binaries given by nBin
		#################################################
		power = []
		freq = []
		n = 0
		nWrite = 0
		eccBin = 0
		
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
		norm_r = 1/2.5
		a_0_r = np.random.uniform(0, 1.0, len(m1))
		r = -2.5*np.log(1-a_0_r)
		# Assign the azimuthal position of the star
		phi = np.random.uniform(0, 2*np.pi, len(m1))
		# Assign the z position of the star with r as a parameter
		norm_zr = 0.71023
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
		
		if (output is None):
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


###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

class TRILEGAL(object):
	def __init__(self, *args,**kwargs):
		self.area = 10.
		self.maglim = 26
		self.sigma_AV = 0.1 #default
		self.binaries = False
		self.filterset = 'lsst' 
		self.tmpfname = 'TRILEGAL_model.h5'
		self.tmpdir = '.'

		self.model = None
		self.KDE = None

		self.RA = None
		self.Dec = None
		self.fieldID = None
		self.Nstars = 0

		self.shuffle = True

	def setModel(self):
		print(f'downloading TRILEGAL model for ID={self.fieldID}, RA={self.RA}, DEC={self.Dec}')
		passed = False
		area0 = self.area
		while (not passed):
			passed = trilegal_update.get_trilegal(self.tmpfname, self.RA, self.Dec, folder=self.tmpdir, galactic=False, \
				filterset=self.filterset, area=self.area, maglim=self.maglim, binaries=self.binaries, \
				trilegal_version='1.6', sigma_AV=self.sigma_AV, convert_h5=True)
			if (not passed):
				self.area *= 0.9
				print(f"reducing TRILEGAL area to {self.area}...")
		self.model = pd.read_hdf(os.path.join(self.tmpdir,self.tmpfname))
		self.Nstars = len(self.model) * (area0/self.area)**2.

		#add the distance
		logDist = np.log10( 10.**(self.model['m-M0'].values/5.) *10. / 1000.) #log(d [kpc])
		self.model['logDist'] = logDist

		if (self.shuffle):
			self.model = self.model.sample(frac=1).reset_index(drop=True)
		os.remove(os.path.join(self.tmpdir,self.tmpfname))

		data = np.vstack((self.model['logL'].values, self.model['logTe'].values, self.model['logg'].values, \
						self.model['logDist'].values, self.model['Av'].values, self.model['[M/H]'].values))
		self.KDE = scipy.stats.gaussian_kde(data)



###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

class SED(object):
	def __init__(self, *args,**kwargs):
		self.filterFilesRoot = '../input/filters/'
		#['u_band_Response.dat','g_band_Response.dat','r_band_Response.dat','i_band_Response.dat','z_band_Response.dat','y_band_Response.dat']
		self.filters = filters
		self.filterThroughput = dict()

		self.useSpecModel = True
		self.specModelName = None
		self.specModel = None
		self.extinctionModel = None

		self.T = None
		self.R = None
		self.L = None
		self.logg = None
		self.M_H = None
		self.EBV = None

	def readFilters(self):
		for f in self.filters:
			#these are nearly identical to what is below, but the others seem like they're in a more obvious location online
			# #https://github.com/lsst-pst/syseng_throughputs/tree/master/components/camera/filters
			# fname = self.filterFilesRoot + f + 'band_Response.dat'
			# df = pd.read_csv(fname, delim_whitespace=True, header=None, names=['w','t'])

			#https://github.com/lsst/throughputs/tree/master/baseline
			fname = self.filterFilesRoot + 'filter_'+f[0]+'.dat' 
			df = pd.read_csv(fname, delim_whitespace=True, header=None, names=['w','t'], skiprows = 6)
			df['nu'] = (constants.c/ (df['w'].values*units.nm)).to(units.Hz).value
			wrng = df[(df['t'] > 0)]
			wmin = min(wrng['w'])
			wmax = max(wrng['w'])
			numin = min(wrng['nu'])
			numax = max(wrng['nu'])
			df.sort_values(by='nu', inplace=True)
			self.filterThroughput[f] = {'nu':df['nu'].values, 't':df['t'].values, 'numin':numin, 'numax':numax, 'wmin':wmin, 'wmax':wmax}

	# def samplespecModel(self, nu): 
	# 	w = (constants.c/(nu*units.Hz)).to(units.angstrom).value
	# 	fac = ((w*units.angstrom)**2. / constants.c).value #not sure what to do about units, want to return Fnu
	# 	fK = np.interp(w, self.specModel.wave, self.specModel.flux)
	# 	print(fK, fac, w)
	# 	return fK*fac

	def intspec(self):
		nu = self.specModel.nu
		dnu = np.ediff1d(nu)*-1. #because it goes in the oposite direction
		fnu = np.sum(self.specModel.fnu[:-1] *dnu)
		return fnu

	def intspecFilter(self, filt):
		nu = self.specModel.nu
		w = (constants.c/(nu*units.Hz)).to(units.angstrom).value
		dnu = np.ediff1d(nu)*-1. #because it goes in the oposite direction
		ft = np.interp(nu, self.filterThroughput[filt]['nu'], self.filterThroughput[filt]['t'])
		#ext = self.extinctionModel(w)
		#ffnu = np.sum(self.specModel.fnu[:-1] * ft[:-1] * ext[:-1] * dnu)
		ffnu = np.sum(self.specModel.fnu[:-1] * ft[:-1] * dnu)
		return ffnu


	def bb(self, w):
		#in cgs is the Ba / s
		#expects w in nm but without a unit attached
		w *= units.nm
		Bl = 2.*constants.h*constants.c**2./w**5. / (np.exp( constants.h*constants.c / (w*constants.k_B*self.T)) -1.)
		return Bl.cgs.value

	#this return inf when integrated to infs	
	def bbv(self, nu):
		#expects nu in 1/s, but without a unit attached
		#return in g/s**2
		nu *= units.Hz
		Bl = 2.*constants.h/constants.c**2. *nu**3. / (np.exp( constants.h*nu / (constants.k_B*self.T)) -1.)
		return Bl.cgs.value

	def filter(self, nu, filt):
		ft = np.interp(nu, self.filterThroughput[filt]['nu'], self.filterThroughput[filt]['t'])
		return ft

	def bbvFilter(self, nu, filt):
		fB = self.bbv(nu)
		ft = self.filter(nu, filt)
		return fB*ft

	def getL(self):
		if (self.useSpecModel):
			fB = self.intspec()
			fB *= units.g/units.s**2.*units.Hz
		else:
			#integrating this over nu returns infs??!!
			fB, fB_err = quad(self.bb, 0, np.inf, limit=1000)
			fB *= units.Ba/units.s*units.nm
		LB = 4.*np.pi*self.R**2. * fB 

		#print("LB = ", LB.to(units.solLum))
		return LB.to(units.solLum)

	def getLconst(self):
		#to account for differences between blackbody and true stellar atmosphere, if true (bolometric) luminosity is known (seems unecessary)
		LB = self.getL()
		return self.L/LB

	def getFvAB(self, dist, filt, Lconst = 1.):
		#http://burro.case.edu/Academics/Astr221/Light/blackbody.html
		#F = sigma*T**4
		#L = 4*pi**2 * R**2 * F
		#Lv*dv = 4*pi**2 * R**2 * Bv*dv
		#Fv*dv = Bv*dv


		#quad misses the filter, and just returns zero when given np.inf limits! and is slow.  So I will do the summation with a small dw

		#dnu = (constants.c/(w*units.nm)**2.*(dw*units.nm)).to(units.Hz).value
		if (self.useSpecModel):
			nu = self.specModel.nu
			dnu = np.ediff1d(nu)*-1. #because it goes in the oposite direction
			fBf = self.intspecFilter(filt) *units.g/units.s**2  * units.Hz 
		else:
			dw = 1e-4
			w = np.arange(self.filterThroughput[filt]['wmin'], self.filterThroughput[filt]['wmax'], dw)
			nu = (constants.c/(w*units.nm)).to(units.Hz).value
			dnu = np.ediff1d(nu)*-1. #because it goes in the oposite direction
			fBf = np.sum(self.bbvFilter(nu,filt)[:-1]*dnu) *units.g/units.s**2 *units.Hz
		f = np.sum(3631.*units.Jansky*self.filter(nu,filt)[:-1]*dnu ) * units.Hz #AB magnitude zero point

		# #the dnu will divide out anyway
		# f = np.sum(self.filter(nu,filt))
		# fBf = np.sum(self.bbvFilter(nu,filt)) *units.g/units.s**2 	


		# fBv = fBf/f * A/(4.*np.pi*dist**2.) 

		# print("T, Lconst", T, self.Lconst)
		fBv = fBf/f * self.R**2./dist**2. * Lconst

		#print(f, fBf, fBv)

		# mAB = -2.5*np.log10(fBv/(3631.*units.Jansky))
		# print("mAB =", mAB)

		return fBv

	def initialize(self):
		self.readFilters()
		if (self.useSpecModel):
			# if (self.T > 3500*units.K):
			#for Kurucz
			self.specModelName = 'ck04models'
			g = np.clip(self.logg, 3, 5.)
			T = np.clip(self.T.to(units.K).value, 3500., 50000.0)
			MH = np.clip(self.M_H, -2.5, 0.5) 
			# else:
			# 	#phoenix for cooler stars, but appear to be giving more discrepant results than just using the Kurucz model
			# 	self.specModelName = 'phoenix'
			# 	g = np.clip(self.logg, 0, 4.5)
			# 	T = np.clip(self.T.to(units.K).value, 2000., 7000.0)
			# 	MH = np.clip(self.M_H, -4, 1)
			#print("parameters", self.logg,g, self.T.to(units.K).value,T, self.M_H, MH)
			self.specModel = pyS.Icat(self.specModelName, T, MH, g)
			self.specModel.nu = (constants.c/(self.specModel.wave*units.angstrom)).to(units.Hz).value
			self.specModel.fnu = ((((self.specModel.wave*units.angstrom)**2./constants.c) * (self.specModel.flux*units.Ba/units.s)).to(units.g/units.s**2.)).value
		#not using this (gives essentially the same values as above, but using an outdated reddening law)
		#self.extinctionModel = pyS.Extinction(self.EBV, 'gal1') #This seems to be a work in progress on their end, I can only access gal1, which is deprecated


###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

class LSSTEBworker(object):

	def __init__(self, *args,**kwargs):

		#NOTE: these need to be defined on the command line.  The default, if not defined, will be False
		self.do_plot = False 
		self.verbose = False
		self.useOpSimDates = True

		self.useFast = True
		self.doLSM = True
		self.do_parallel = False 

		self.years = 10.
		self.totaltime = 365.* self.years
		self.cadence = 3.

		self.filters = filters

		self.n_bin = 100000
		self.n_band = 2
		self.n_base = 2  
		self.n_cores = 1

		self.ofile = 'output_file.csv' #output file name
		self.dbFile = '../input/db/minion_1016_sqlite.db' #for the OpSim database
		self.filterFilesRoot = '../input/filters/'
		self.GalaxyFile ='../input/Breivik/dat_ThinDisk_12_0_12_0.h5' #for Katie's model
		self.GalaxyFileLogPrefix ='../input/Breivik/fixedPopLogCm_'
		self.galDir = './'

		#dictionaries -- could be handled by the multiprocessing manager, redefined in driver
		self.return_dict = dict()

		self.csvwriter = None #will hold the csvwriter object

		#some counters
		self.n_totalrun = 0
		self.n_appmag_failed = 0
		self.n_incl_failed = 0
		self.n_period_failed = 0
		self.n_radius_failed = 0

		self.NobsLim = 10 #limit total number of obs below which we will not run it through anything (in initialize)

		self.OpSim = None #will hold the OpSim object
		self.Galaxy = None #will hold TRILEGAL object
		self.Breivik = None
		self.BreivikGal = None

		self.seed = None

		#for emcee 
		self.emcee_nthreads = 1 #note: currently, I *think* this won't work with >1 thread.  Not pickleable as written.
		self.emcee_nsamples = 2000
		self.emcee_nwalkers = 100
		self.emcee_nburn = 100
		self.emcee_sampler = None

	def make_gatspy_plots(self, j):
		EB = self.return_dict[j]

		#colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
		#print([ matplotlib.colors.to_hex(c) for c in colors])
		colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

		f, ax = plt.subplots(len(self.filters)+1, 2)

		LSM = EB.LSM
		period = EB.period
		pds = np.linspace(0.2, 2.*period, 10000)
		for ii,filt in enumerate(self.filters):

			drng = max(EB.obsDates[filt]) - min(EB.obsDates[filt])

			phase_obs = np.array([(tt % period)/period for tt in EB.obsDates[filt]])
			scores = EB.LSSmodel[filt].score(pds)
			mag_obs = EB.appMagObs[filt]
			mag = EB.appMag[filt]
			LSS = EB.LSS[filt]

			sx = np.argsort(phase_obs)
			ax[ii][0].plot(phase_obs[sx], np.array(mag_obs)[sx], 'o', mfc='none', mec = colors[ii])
			ax[ii][0].plot(phase_obs[sx], np.array(mag)[sx], color = "black")
			ax[ii][0].set_ylim(ax[ii][0].get_ylim()[::-1])
			ax[ii][0].set_xlim(0,1)
			ax[ii][0].set_ylabel(filt)
			ax[ii][0].set_xticklabels([])

			ax[ii][1].plot(pds, scores, color = colors[ii])
			ax[ii][1].plot([LSS,LSS],[0,1], color = "dimgray", lw = 2)
			ax[ii][1].plot([period,period],[0,1],'--', color = "black")
			ax[ii][1].set_xlim(0, 2.*period)
			ax[ii][1].set_ylim(0, max(scores))
			ax[ii][1].set_xticklabels([])
			ax[ii][1].set_yticklabels([])

		if (self.doLSM):
			plt.locator_params(axis='y', nticks=2)
			P_multi = EB.LSMmodel.periodogram(pds)
			ii = len(self.filters)
			ax[ii][1].plot(pds, P_multi, color = colors[ii])
			ax[ii][1].set_xlim(0, 2.*period)
			ax[ii][1].set_ylim(0, max(P_multi))
			ax[ii][1].plot([period,period],[0,1],'--', color = "black")
			ax[ii][1].plot([LSM,LSM],[0,1], ':', color = "dimgray")

		f.subplots_adjust(hspace=0.1, wspace=0.1)
		f.delaxes(ax[ii][0])
		ax[ii-1][0].set_xlabel("phase")
		ax[ii][1].set_xlabel("period (days)")
		ax[ii][1].set_yticklabels([])

		f.savefig("lc_gatspy_fig_"+str(seed).rjust(10,'0')+".png", bbox_inches='tight')


	def run_ellc_gatspy(self, j):
		#this is the general simulation - ellc light curves and gatspy periodograms

		EB = self.return_dict[j]

		#for the multiband gatspy fit
		allObsDates = np.array([])
		allAppMagObs = np.array([])
		allAppMagObsErr = np.array([])
		allObsFilters = np.array([])
		minNobs = 1e10
		if (self.verbose):
			print("in run_ellc_gatspy")


		for i, filt in enumerate(self.filters):

			#observe the EB (get dates, create the light curve for this filter)
			EB.observe(filt)
			EB.LSS[filt] = -999.

			if (EB.obsDates[filt][0] is not None and min(EB.appMagObs[filt]) > 0):

				#run gatspy for this filter
				drng = max(EB.obsDates[filt]) - min(EB.obsDates[filt])
				minNobs = min(minNobs, len(EB.obsDates[filt]))
				#print("filter, nobs", filt, len(EB.obsDates[filt]))
				if (self.useFast and len(EB.obsDates[filt]) > 50):
					model = LombScargleFast(fit_period = True, silence_warnings=True, optimizer_kwds={"quiet": True})
				else:
					model = LombScargle(fit_period = True, optimizer_kwds={"quiet": True})
				model.optimizer.period_range = (0.2, drng)
				model.fit(EB.obsDates[filt], EB.appMagObs[filt], EB.appMagObsErr[filt])
				EB.LSS[filt] = model.best_period
				EB.LSSmodel[filt] = model
				EB.maxDeltaMag = max(EB.deltaMag[filt], EB.maxDeltaMag)

				#to use for the multiband fit
				allObsDates = np.append(allObsDates, EB.obsDates[filt])
				allAppMagObs = np.append(allAppMagObs, EB.appMagObs[filt])
				allAppMagObsErr = np.append(allAppMagObsErr, EB.appMagObsErr[filt])
				allObsFilters = np.append(allObsFilters, np.full(len(EB.obsDates[filt]), filt))

				if (self.verbose): 
					print(j, 'filter = ', filt)  
					print(j, 'obsDates = ', EB.obsDates[filt][0:10])
					print(j, 'appMagObs = ', EB.appMagObs[filt][0:10])
					print(j, 'delta_mag = ', EB.deltaMag[filt])
					print(j, 'LSS = ',EB.LSS[filt])

		if (len(allObsDates) > 0 and self.doLSM): 
			drng = max(allObsDates) - min(allObsDates)
			if (self.useFast and minNobs > 50):
				model = LombScargleMultibandFast(fit_period = True, optimizer_kwds={"quiet": True})
			else:
				model = LombScargleMultiband(Nterms_band=self.n_band, Nterms_base=self.n_base, fit_period = True, optimizer_kwds={"quiet": True})
			model.optimizer.period_range = (0.2, drng)
			model.fit(allObsDates, allAppMagObs, allAppMagObsErr, allObsFilters)
			EB.LSM = model.best_period
			EB.LSMmodel = model
			if (self.verbose): 
				print(j, 'LSM =', EB.LSM)


		#not sure if I need to do this...
		self.return_dict[j] = EB




	def getEB(self, line, OpSimi=0):
		EB = EclipsingBinary()

		# EB.seed = self.seed + i
		EB.initializeSeed()
		EB.filterFilesRoot = self.filterFilesRoot

		#solar units
		EB.m1 = line[0]
		EB.m2 = line[1]
		EB.r1 = line[4]
		EB.r2 = line[5]
		EB.L1 = line[6]
		EB.L2 = line[7]
		EB.period = 10.**line[2] #days
		EB.eccentricity = line[3]
		EB.inclination = line[12] *180./np.pi #degrees
		EB.OMEGA = line[13] *180./np.pi #degrees
		EB.omega = line[14] *180./np.pi #degrees

		EB.dist = line[11] #kpc
		if (self.Galaxy is None):
			#pc
			EB.xGx = line[8] 
			EB.yGx = line[9] 
			EB.zGx = line[10] 
		else:
			EB.OpSimi = OpSimi
			EB.RA = self.OpSim.RA[OpSimi]
			EB.Dec = self.OpSim.Dec[OpSimi]

		if (len(line) >= 16):
			EB.AV = line[15]
		if (len(line) >= 17):
			EB.M_H = line[16]


		EB.t_zero = np.random.random() * EB.period

		#for observations
		EB.useOpSimDates = self.useOpSimDates
		EB.years = self.years
		EB.totaltime = self.totaltime 
		EB.cadence= self.cadence 
		EB.Nfilters = len(self.filters)
		EB.verbose = self.verbose
		if (self.useOpSimDates):
			#print("sending OpSim to EB", self.OpSim.obsDates)
			EB.OpSim = self.OpSim
		EB.initialize()

		#some counters for how many EBs we could potentially observe with LSST
		self.n_totalrun += 1
		self.n_appmag_failed += EB.appmag_failed
		self.n_incl_failed += EB.incl_failed
		self.n_period_failed += EB.period_failed
		self.n_radius_failed += EB.radius_failed
			
		return EB


	def writeOutputLine(self, EB, OpSimi=0, header = False, noRun = False):
		cols = ['p', 'm1', 'm2', 'r1', 'r2', 'e', 'i', 'd', 'nobs','Av','[M/H]','appMagMean_r', 'maxDeltaMag', 'deltaMag_r','mag_failure', 'incl_failure', 'period_failure', 'radius_failure', 'u_LSS_PERIOD', 'g_LSS_PERIOD', 'r_LSS_PERIOD', 'i_LSS_PERIOD', 'z_LSS_PERIOD', 'y_LSS_PERIOD','LSM_PERIOD']
		if (header):
			#print(self.useOpSimDates, self.Galaxy, self.OpSim)
			ng = 0
			if (self.Galaxy is not None):
				ng = self.Galaxy.Nstars
			if (self.useOpSimDates and self.OpSim is not None):
				print("writing header")
				self.csvwriter.writerow(['OpSimID','OpSimRA','OpSimDec','NstarsTRILEGAL', 'NOpSimObs_u', 'NOpSimObs_g', 'NOpSimObs_r', 'NOpSimObs_i', 'NOpSimObs_z', 'NOpSimObs_y'])
				self.csvwriter.writerow([self.OpSim.fieldID[OpSimi], self.OpSim.RA[OpSimi], self.OpSim.Dec[OpSimi], ng, self.OpSim.NobsDates[OpSimi]['u_'], self.OpSim.NobsDates[OpSimi]['g_'], self.OpSim.NobsDates[OpSimi]['r_'], self.OpSim.NobsDates[OpSimi]['i_'], self.OpSim.NobsDates[OpSimi]['z_'], self.OpSim.NobsDates[OpSimi]['y_']])

			output = cols

		elif (noRun):
			output = [-1 for x in range(len(cols))]

		else:
			output = [EB.period, EB.m1, EB.m2, EB.r1, EB.r2, EB.eccentricity, EB.inclination, EB.dist, EB.nobs, EB.AV, EB.M_H, EB.appMagMean['r_'], EB.maxDeltaMag, EB.deltaMag['r_'],EB.appmag_failed, EB.incl_failed, EB.period_failed, EB.radius_failed]

			#this is for gatspt
			for filt in self.filters:
				output.append(EB.LSS[filt]) 
			output.append(EB.LSM) 

		self.csvwriter.writerow(output)	


	def matchBreivikTRILEGAL(self):

		print("matching Breivik to TRILEGAL")
		# compute the log likelihood
		def lnlike(theta, logm1, logr1, logL1, pT):
			logm2, logr2, logL2, ecc, logp = theta
			
			def paramTransform(key, val):
				datMin = self.Breivik.datMinMax[key][0]
				datMax = self.Breivik.datMinMax[key][1]
						
				return (val - datMin)/(datMax-datMin)
			
		#NOTE: this is confusing, but in g.fixedPop rad and lum are already in the log! 
		#And here I have already transformed porb to logP
			m1Trans = ss.logit(paramTransform('m1', 10.**logm1))
			m2Trans = ss.logit(paramTransform('m2', 10.**logm2))
			r1Trans = ss.logit(paramTransform('logr1', logr1))
			r2Trans = ss.logit(paramTransform('logr2', logr2))
			L1Trans = ss.logit(paramTransform('logL1', logL1))
			L2Trans = ss.logit(paramTransform('logL2', logL2))
			pTrans = ss.logit(paramTransform('logp', logp))
			eccTrans = np.clip(ecc, 1e-4, 0.999)
				
			pB = self.Breivik.sampleKernel( (m1Trans, m2Trans, pTrans, eccTrans, r1Trans, r2Trans, L1Trans, L2Trans) )
			lk = pB# * pT
			
			if (lk <= 0):
				return -np.inf
			
			return np.log(lk).squeeze()

		# compute the log prior
		def lnprior(theta):
			#some reasonable limits to place, so that Breivik's KDE can be sampled properly
			logm2, logr2, logL2, ecc, logp = theta
			if ( (-2 < logm2 < 2) and (-3 < logr2 < 3) and (-5 < logL2 < 5) and (0 < ecc < 1) and (-3 < logp < 10)):
				return 0.0
			return -np.inf

		# compute the log of the likelihood multiplied by the prior
		def lnprob(theta):
			lnp = lnprior(theta)
			
			#get the primary star from the TRILEGAL model
			sample = self.Galaxy.KDE.resample(size=1)
			logL1 = sample[0,0]
			logT1 = sample[1,0]
			logg1 = sample[2,0]
			logD = sample[3,0]
			Av = sample[4,0]
			MH = sample[5,0]
			pT = self.Galaxy.KDE( (logL1, logT1, logg1, logD, Av, MH) )

			logr1 = 2.*(0.25*logL1 - logT1 + 3.762) #taken from my N-body notes to get logT <-- from Jarrod Hurley
			
			#np.log10(constants.G.to(u.cm**3. / u.g / u.s**2.).value) = -7.175608591905032
			#print(np.log10((1.*u.solMass).to(u.g).value)) = 33.29852022592346
			logm1 = logg1 + 7.175608591905032 + 2.*(logr1 + 10.84242200335765) - 33.29852022592346

			lnl = lnlike(theta, logm1, logr1, logL1, pT)
			
			if (not np.isfinite(lnp) or not np.isfinite(lnl)):
				#return -np.inf, np.array([0., 0., 0., 0., 0., 0.])
				return -np.inf, np.array([None, None, None, None, None, None])

			#returning the TRILEGAL parameters as "blobs" so that I can1 use them later
			return lnp + lnl, np.array([Av, MH, logD, logm1, logr1, logL1])

		#now set up the emcee	


		paramsNames = ['m2', 'r2', 'L2', 'ecc', 'logp']
		outNames = ['logm2', 'logr2', 'logL2', 'ecc', 'logp']
		reNames = {}
		for x,y in zip(paramsNames, outNames):
			reNames[x] = y

		BreivikBin = self.Breivik.GxSample(int(self.emcee_nwalkers))

		#take the log of m2 and rename the columns accordingly
		walkers = pd.concat( [BreivikBin[paramsNames[0]].apply(np.log10), 
							  BreivikBin[paramsNames[1]].apply(np.log10),
							  BreivikBin[paramsNames[2]].apply(np.log10),
							  BreivikBin[paramsNames[3:]] 
							 ], axis=1)
		walkers.rename(columns = reNames, inplace=True)

		tot = self.emcee_nsamples*self.emcee_nwalkers - self.emcee_nburn
		if (tot < self.n_bin):
			print(f'WARNING: number of emcee samples={tot}, but number of requested binaries={self.n_bin}.  Increasing emcee sample')
			self.emcee_nsamples = 1.5*int(np.ceil((self.n_bin + self.emcee_nburn)/self.emcee_nwalkers))
			
		print(f'{datetime.now()} running emcee with nwalkers={self.emcee_nwalkers}, nsamples={self.emcee_nsamples}, nthreads={self.emcee_nthreads}, ')
		self.emcee_sampler = emcee.EnsembleSampler(self.emcee_nwalkers, len(outNames), lnprob, threads = self.emcee_nthreads)

		#this will run it through emcee
		foo = self.emcee_sampler.run_mcmc(walkers.values, self.emcee_nsamples)

		#now gather the output
		outNames = ['logm2', 'logr2', 'logL2', 'ecc', 'logp']
		samples = self.emcee_sampler.chain[:, self.emcee_nburn:, :].reshape((-1, len(outNames)))
		blobs = np.array(self.emcee_sampler.blobs[self.emcee_nburn:][:][:])
		extras = blobs.reshape((samples.shape[0], blobs.shape[-1]))
		result = np.hstack((extras, samples))

		self.BreivikGal = result


	def sampleBreivikGal(self):
		ind = range(len(self.BreivikGal))
		indices = np.random.choice(ind, size=self.n_bin, replace=False)
		s = self.BreivikGal[indices].T
		#outNames = ['Av', '[M/H]', 'logD', logm1', 'logr1', 'logL1', 'logm2', 'logr2', 'logL2', 'ecc', 'logp']

		Av = s[0]
		MH = s[1]
		d = 10.**s[2]
		m1 = 10.**s[3]
		r1 = 10.**s[4]
		L1 = 10.**s[5]
		m2 = 10.**s[6]
		r2 = 10.**s[7]
		L2 = 10.**s[8]
		ecc = s[9]
		logp = s[10]
		inc = np.arccos(2.*np.random.uniform(0,1,self.n_bin) - 1.)
		omega = np.random.uniform(0,2*np.pi,self.n_bin)
		OMEGA = np.random.uniform(0,2*np.pi,self.n_bin)
		x = np.zeros(self.n_bin)

		#we don't need position, but we do need distance
		#[m1, m2, logp, ecc, r1, r2, L1, L2, x,y,z, dist, inc, OMEGA, omega, Av, MH]
		#binDat = np.vstack((m1, m2, logp, ecc, rad1, rad2, Lum1, Lum2, xGX, yGX, zGX, dist_kpc, inc, OMEGA, omega)).T

		return (np.vstack( (m1, m2, logp, ecc, r1, r2, L1, L2, x, x, x, d, inc, OMEGA, omega, Av, MH) ).T).squeeze()

	def initialize(self, OpSimi=0):
		if (self.seed is None):
			np.random.seed()
		else:
			np.random.seed(seed = self.seed)


		if (self.useOpSimDates and self.OpSim is None):
			self.OpSim = OpSim()
			#get the OpSim fields
			self.OpSim.getAllOpSimFields()


		#check if we need to run this
		if (self.useOpSimDates):
			self.OpSim.setDates(OpSimi, self.filters)
			print(f'total number of OpSim observation dates (all filters) = {self.OpSim.totalNobs[OpSimi]}')
			if (self.OpSim.totalNobs[OpSimi] < self.NobsLim):
				return False

		#I need to move this up if I still want galaxy to get total number of stars, even if we're not observing it
		if (self.Galaxy is None):
			self.Galaxy = TRILEGAL()
			self.Galaxy.RA = self.OpSim.RA[OpSimi]
			self.Galaxy.Dec = self.OpSim.Dec[OpSimi]
			self.Galaxy.fieldID = self.OpSim.fieldID[OpSimi]
			self.Galaxy.tmpdir = self.galDir
			self.Galaxy.tmpfname = 'TRILEGAL_model_fID'+str(int(self.OpSim.fieldID[OpSimi]))+'.h5'
			self.Galaxy.setModel()	

		if (self.Breivik is None):
			self.Breivik = BreivikGalaxy()
			self.Breivik.GalaxyFile = self.GalaxyFile
			self.Breivik.GalaxyFileLogPrefix = self.GalaxyFileLogPrefix
			self.Breivik.setKernel()

		if (self.BreivikGal is None):
			self.matchBreivikTRILEGAL()

		return True
