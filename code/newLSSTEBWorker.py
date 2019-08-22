import numpy as np
import pandas as pd
import scipy.special as ss
import datetime
from astropy.modeling import models, fitting
from astropy import units as u
from astropy import constants 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


#######################
#3rd party codes
from gatspy.periodic import LombScargleMultiband, LombScargle, LombScargleFast, LombScargleMultibandFast

#for A_V
#import vespa.stars.extinction
from vespa_update import extinction

######################
#my code
from EclipsingBinary import EclipsingBinary
from OpSim import OpSim
from TRILEGAL import TRILEGAL


class LSSTEBWorker(object):

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

		self.filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']

		self.OpSim = None

		self.n_bin = 100000
		self.n_band = 2
		self.n_base = 2  
		self.n_cores = 1

		self.ofile = 'output_file.csv' #output file name
		self.dbFile = '../input/db/baseline2018a.db' #for the OpSim database
		self.filterFilesRoot = '../input/filters/'

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

		self.seed = None

		self.Galaxy = None
		self.mTol = 0.001 #tolerance on the mass to draw from the trilegal sample

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

			#drng = max(EB.obsDates[filt]) - min(EB.obsDates[filt])

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

		f.savefig("lc_gatspy_fig_"+str(self.seed).rjust(10,'0')+".png", bbox_inches='tight')


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
			#print("checking observe", filt, EB.obsDates[filt][0])
			EB.appMagObs[filt] = [0.]
			EB.appMagObsErr[filt] = [0.]
			EB.deltaMag[filt] = [0.]
			EB.LSS[filt] = -999.

			EB.observe(filt)

			if (EB.obsDates[filt][0] != None and min(EB.appMagObs[filt]) > 0):

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
		EB.filters = self.filters

		#solar units
		EB.m1 = line[0]
		EB.m2 = line[1]
		EB.r1 = line[4]
		EB.r2 = line[5]
		EB.L1 = line[6]
		EB.L2 = line[7]
		EB.T1 = line[17]
		EB.T2 = line[18]
		EB.period = 10.**line[2] #days
		EB.eccentricity = line[3]
		EB.inclination = line[12] *180./np.pi #degrees
		EB.OMEGA = line[13] *180./np.pi #degrees
		EB.omega = line[14] *180./np.pi #degrees

		EB.dist = line[11] #kpc
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
			if (self.useOpSimDates and self.OpSim != None):
				print("writing header")
				ng = 0
				if (self.Galaxy != None):
					ng = self.Galaxy.Nstars
				self.csvwriter.writerow(['OpSimID','OpSimRA','OpSimDec','NstarsTRILEGAL', 'NOpSimObs_u', 'NOpSimObs_g', 'NOpSimObs_r', 'NOpSimObs_i', 'NOpSimObs_z', 'NOpSimObs_y'])
				self.csvwriter.writerow([self.OpSim.fieldID[OpSimi], self.OpSim.RA[OpSimi], self.OpSim.Dec[OpSimi], ng, self.OpSim.NobsDates[OpSimi]['u_'], self.OpSim.NobsDates[OpSimi]['g_'], self.OpSim.NobsDates[OpSimi]['r_'], self.OpSim.NobsDates[OpSimi]['i_'], self.OpSim.NobsDates[OpSimi]['z_'], self.OpSim.NobsDates[OpSimi]['y_']])

			output = cols

		elif (noRun):
			output = [-1 for x in range(len(cols))]

		else:
			output = [EB.period, EB.m1, EB.m2, EB.r1, EB.r2, EB.eccentricity, EB.inclination, EB.dist, EB.nobs, EB.AV, EB.M_H, EB.appMagMean['r_'], EB.maxDeltaMag, EB.deltaMag['r_'],EB.appmag_failed, EB.incl_failed, EB.period_failed, EB.radius_failed]

			#this is for gatspy
			for filt in self.filters:
				output.append(EB.LSS[filt]) 
			output.append(EB.LSM) 

		self.csvwriter.writerow(output)	


	def getGalaxy(self, OpSimi, deleteModel = True):
		self.Galaxy = TRILEGAL()
		self.Galaxy.RA = self.OpSim.RA[OpSimi]
		self.Galaxy.Dec = self.OpSim.Dec[OpSimi]
		self.Galaxy.fieldID = self.OpSim.fieldID[OpSimi]
		self.Galaxy.tmpdir = self.galDir
		self.Galaxy.tmpfname = 'TRILEGAL_model_fID'+str(int(self.OpSim.fieldID[OpSimi]))+'.h5'
		self.Galaxy.deleteModel = deleteModel
		self.Galaxy.setModel()

	def sampleGalaxy(self, OpSimi):


		#for the binary fraction
		def fitRagfb():
			x = [0.05, 0.1, 1, 8, 15]  #estimates of midpoints in bins, and using this: https://sites.uni.edu/morgans/astro/course/Notes/section2/spectralmasses.html
			y = [0.20, 0.35, 0.50, 0.70, 0.75]
			init = models.PowerLaw1D(amplitude=0.5, x_0=1, alpha=-1.)
			fitter = fitting.LevMarLSQFitter()
			fit = fitter(init, x, y)

			return fit

		#uniform distribution for mass ratio
		def getq():
			return np.random.random()

		#uniform distribution for eccentricity
		def getecc():
			return np.random.random()

		#log-normal distribution for period
		def getlogp():
			return np.random.normal(loc=5.03, scale=2.28)

		def getRad(logg, m):
			#g = GM/r**2
			g = 10.**logg * u.cm/u.s**2.
			r = ((constants.G*m*u.Msun/g)**0.5).decompose().to(u.Rsun).value
			return r

#test for the Sun
#print(getRad(4.43, 1)) 
		print("creating binaries...")
		m2Use = 0.5
		mTolUse = 0.001

		fbFit= fitRagfb()
		#print(fbFit)

		m1 = []
		rad1 = []
		lum1 = []
		teff1 = []

		m2 = []
		rad2 = []
		lum2 = []
		teff2 = []

		logp = []
		ecc = []

		dist = []
		Av = [] 
		MH = []
		nWarn = 0.
		maxTol = self.mTol
		while len(m1) < self.n_bin:
			s = self.Galaxy.model.sample()
			fb = fbFit(s['m_ini'].iloc[0]) #I think I should base this on the initial mass, since these binary fractions are for unevolved stars
			xx = np.random.random()
			if (xx < fb):
				m1.append(s['Mact'].iloc[0])
				rad1.append(getRad(s['logg'].iloc[0], s['Mact'].iloc[0]))
				lum1.append(10.**s['logL'].iloc[0])
				teff1.append(10.**s['logTe'].iloc[0])
				
				m2Use = s['Mact'].iloc[0]*getq()
				#rad2, lum2, teff2 need to be drawn from TRILEGAL given m2
				done = False
				mTolUse = self.mTol
				counter = 0.
				while (not done):
					df_sort = self.Galaxy.model.loc[ (self.Galaxy.model['Mact'] - m2Use).abs() < mTolUse]
					if (len(df_sort) > 0):
						done = True
						ss = df_sort.sample()
						m2.append(ss['Mact'].iloc[0])
						rad2.append(getRad(ss['logg'].iloc[0], ss['Mact'].iloc[0]))
						lum2.append(10.**ss['logL'].iloc[0])
						teff2.append(10.**ss['logTe'].iloc[0])
					else:
						#print('WARNING: increasing tolerance', mTolUse)
						mTolUse *=2
						nWarn += 1
						maxTol = np.max([mTolUse, maxTol])
					if (counter > 100):
						print('WARNING: did not reach tolerance, will probably die...')
					counter += 1
				logp.append(getlogp())
				ecc.append(getecc())
				
				dist.append(10.**s['logDist'].iloc[0])
				Av.append(s['Av'].iloc[0]) #is this measure OK?
				MH.append(s['[M/H]'].iloc[0])

		if (nWarn > 0):
			print(f'WARNING: had to increase mass tolerance {nWarn} times. Max tolerance = {maxTol}.')
		m1 = np.array(m1)
		rad1 = np.array(rad1)
		lum1 = np.array(lum1)
		m2 = np.array(m2)
		rad2 = np.array(rad2)
		lum2 = np.array(lum2)
		logp = np.array(logp)
		ecc = np.array(ecc)
		dist = np.array(dist)
		Av = np.array(Av)
		MH = np.array(MH)

		#random angles
		inc = np.arccos(2.*np.random.uniform(0,1,self.n_bin) - 1.)
		omega = np.random.uniform(0,2*np.pi,self.n_bin)
		OMEGA = np.random.uniform(0,2*np.pi,self.n_bin)
		x = np.zeros(self.n_bin)


		#we don't need position, but we do need distance
		output = np.vstack( (m1, m2, logp, ecc, rad1, rad2, lum1, lum2, x, x, x, dist, inc, OMEGA, omega, Av, MH, teff1, teff2) ).T

		return output


	def initialize(self, OpSimi=0):
		if (self.seed == None):
			np.random.seed()
		else:
			np.random.seed(seed = self.seed)

		#OpSim
		self.OpSim.setDates(OpSimi, self.filters)
		print(f'total number of OpSim observation dates (all filters) = {self.OpSim.totalNobs[OpSimi]}')
		if (self.OpSim.totalNobs[OpSimi] < self.NobsLim):
			return False

		return True

