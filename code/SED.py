import numpy as np
import pandas as pd
from astropy import units, constants
from scipy.integrate import quad

#######################
#3rd party codes
#pysynphot from STScI
#https://pysynphot.readthedocs.io/en/latest/index.html#pysynphot-installation-setup
#need to download grid of models manually, and set path accordingly
import pysynphot as pyS


class SED(object):
	def __init__(self, *args,**kwargs):
		self.filterFilesRoot = '../input/filters/'
		#['u_band_Response.dat','g_band_Response.dat','r_band_Response.dat','i_band_Response.dat','z_band_Response.dat','y_band_Response.dat']
		self.filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']
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
		#w = (constants.c/(nu*units.Hz)).to(units.angstrom).value
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
