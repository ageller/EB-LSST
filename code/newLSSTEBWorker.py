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
from crowding import crowding
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
		self.seeing = 0.5 #arcsec

		self.OpSim = None
		self.EB = None

		self.n_bin = 100000
		self.n_band = 2
		self.n_base = 2  
		self.n_cores = 1

		self.ofile = 'output_file.csv' #output file name
		self.dbFile = '../input/db/baseline2018a.db' #for the OpSim database
		self.filterFilesRoot = '../input/filters/'

		self.doCrowding = True

		self.csvwriter = None #will hold the csvwriter object

		self.NobsLim = 10 #limit total number of obs below which we will not run it through anything (in initialize)

		self.OpSim = None #will hold the OpSim object

		self.seed = None

		self.Galaxy = None
		self.galDir = ''
		self.mTol = 0.001 #tolerance on the mass to draw from the trilegal sample

		self.magLims = np.array([15.8, 24.]) #lower and upper limits on the magnitude detection assumed for LSST: 15.8 = rband saturation from Science Book page 57, before Section 3.3; 24.5 is the desired detection limit
		self.eclipseDepthLim = 3. #depth / error


	def make_gatspy_plots(self, j):

		#colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
		#print([ matplotlib.colors.to_hex(c) for c in colors])
		colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

		f, ax = plt.subplots(len(self.filters)+1, 2)

		LSM = self.EB.LSM
		period = self.EB.period
		pds = np.linspace(0.2, 2.*period, 10000)
		for ii,filt in enumerate(self.filters):

			#drng = max(self.EB.obsDates[filt]) - min(self.EB.obsDates[filt])

			phase_obs = np.array([(tt % period)/period for tt in self.EB.obsDates[filt]])
			scores = self.EB.LSSmodel[filt].score(pds)
			mag_obs = self.EB.appMagObs[filt]
			mag = self.EB.appMag[filt]
			LSS = self.EB.LSS[filt]

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
			P_multi = self.EB.LSMmodel.periodogram(pds)
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


	def run_ellc(self, light_3=None):
		for i, filt in enumerate(self.filters):

			#observe the EB (get dates, create the light curve for this filter)
			#print("checking observe", filt, self.EB.obsDates[filt][0])
			self.EB.appMagObs[filt] = [0.]
			self.EB.appMagObsErr[filt] = [0.]
			self.EB.deltaMag[filt] = [0.]

			uselight_3 = None
			if (light_3 is not None):
				uselight_3 = light_3[filt]

			self.EB.observe(filt, light_3=uselight_3)

	def run_gatspy(self):
		#this is the general simulation - ellc light curves and gatspy periodograms

		#for the multiband gatspy fit
		allObsDates = np.array([])
		allAppMagObs = np.array([])
		allAppMagObsErr = np.array([])
		allObsFilters = np.array([])
		minNobs = 1e10
		if (self.verbose):
			print("in run_ellc_gatspy")


		for i, filt in enumerate(self.filters):

			self.EB.LSS[filt] = -999.

			if (self.EB.obsDates[filt][0] is not None and min(self.EB.appMagObs[filt]) > 0):

				#run gatspy for this filter
				drng = max(self.EB.obsDates[filt]) - min(self.EB.obsDates[filt])
				minNobs = min(minNobs, len(self.EB.obsDates[filt]))
				#print("filter, nobs", filt, len(self.EB.obsDates[filt]))
				if (self.useFast and len(self.EB.obsDates[filt]) > 50):
					model = LombScargleFast(fit_period = True, silence_warnings=True, optimizer_kwds={"quiet": True})
				else:
					model = LombScargle(fit_period = True, optimizer_kwds={"quiet": True})
				model.optimizer.period_range = (0.2, drng)
				model.fit(self.EB.obsDates[filt], self.EB.appMagObs[filt], self.EB.appMagObsErr[filt])
				self.EB.LSS[filt] = model.best_period
				self.EB.LSSmodel[filt] = model
				self.EB.maxDeltaMag = max(self.EB.deltaMag[filt], self.EB.maxDeltaMag)

				#to use for the multiband fit
				allObsDates = np.append(allObsDates, self.EB.obsDates[filt])
				allAppMagObs = np.append(allAppMagObs, self.EB.appMagObs[filt])
				allAppMagObsErr = np.append(allAppMagObsErr, self.EB.appMagObsErr[filt])
				allObsFilters = np.append(allObsFilters, np.full(len(self.EB.obsDates[filt]), filt))

				if (self.verbose): 
					print('filter = ', filt)  
					print('obsDates = ', self.EB.obsDates[filt][0:10])
					print('appMagObs = ', self.EB.appMagObs[filt][0:10])
					print('delta_mag = ', self.EB.deltaMag[filt])
					print('LSS = ',self.EB.LSS[filt])

		if (len(allObsDates) > 0 and self.doLSM): 
			drng = max(allObsDates) - min(allObsDates)
			if (self.useFast and minNobs > 50):
				model = LombScargleMultibandFast(fit_period = True, optimizer_kwds={"quiet": True})
			else:
				model = LombScargleMultiband(Nterms_band=self.n_band, Nterms_base=self.n_base, fit_period = True, optimizer_kwds={"quiet": True})
			model.optimizer.period_range = (0.2, drng)
			model.fit(allObsDates, allAppMagObs, allAppMagObsErr, allObsFilters)
			self.EB.LSM = model.best_period
			self.EB.LSMmodel = model
			if (self.verbose): 
				print('LSM =', self.EB.LSM)



	def getEB(self, line, OpSimi=0):
		self.EB = EclipsingBinary()
		self.EB.magLims = self.magLims
		self.EB.eclipseDepthLim = self.eclipseDepthLim

		self.EB.Galaxy = self.Galaxy
		
		# self.EB.seed = self.seed + i
		self.EB.initializeSeed()
		self.EB.filterFilesRoot = self.filterFilesRoot
		self.EB.filters = self.filters

		#solar units
		self.EB.m1 = line[0]
		self.EB.m2 = line[1]
		self.EB.r1 = line[4]
		self.EB.r2 = line[5]
		self.EB.L1 = line[6]
		self.EB.L2 = line[7]
		self.EB.T1 = line[17]
		self.EB.T2 = line[18]
		self.EB.g1 = line[19]
		self.EB.g2 = line[20]
		self.EB.period = 10.**line[2] #days
		self.EB.eccentricity = line[3]
		self.EB.inclination = line[12] *180./np.pi #degrees
		self.EB.OMEGA = line[13] *180./np.pi #degrees
		self.EB.omega = line[14] *180./np.pi #degrees

		self.EB.dist = line[11] #kpc
		self.EB.OpSimi = OpSimi
		self.EB.RA = self.OpSim.RA[OpSimi]
		self.EB.Dec = self.OpSim.Dec[OpSimi]

		self.EB.AV = line[15]
		self.EB.M_H = line[16]

		self.EB.TRILEGALrmag = line[21]

		self.EB.t_zero = np.random.random() * self.EB.period

		#for observations
		self.EB.useOpSimDates = self.useOpSimDates
		self.EB.years = self.years
		self.EB.totaltime = self.totaltime 
		self.EB.cadence= self.cadence 
		self.EB.Nfilters = len(self.filters)
		self.EB.verbose = self.verbose
		if (self.useOpSimDates):
			#print("sending OpSim to EB", self.OpSim.obsDates)
			self.EB.OpSim = self.OpSim

		#set up the crowding class
		if (self.doCrowding):
			self.EB.crowding = crowding()
			self.EB.crowding.filterFilesRoot = self.filterFilesRoot
			self.EB.crowding.Galaxy = self.Galaxy


		self.EB.initialize()
			

	def writeOutputLine(self, OpSimi=0, header = False, noRun = False):
		cols = ['p', 'm1', 'm2', 'r1', 'r2', 'e', 'i', 'd', 'nobs','Av','[M/H]','appMagMean_r', 'maxDeltaMag','deltaMag_r','eclipseDepthFrac_r','mag_failure', 'incl_failure', 'period_failure', 'radius_failure', 'eclipseDepth_failure', 'u_LSS_PERIOD', 'g_LSS_PERIOD', 'r_LSS_PERIOD', 'i_LSS_PERIOD', 'z_LSS_PERIOD', 'y_LSS_PERIOD','LSM_PERIOD']
		if (header):
			if (self.useOpSimDates and self.OpSim is not None):
				print("writing header")
				ng = 0
				if (self.Galaxy is not None):
					ng = self.Galaxy.Nstars
				self.csvwriter.writerow(['OpSimID','OpSimRA','OpSimDec','NstarsTRILEGAL', 'NOpSimObs_u', 'NOpSimObs_g', 'NOpSimObs_r', 'NOpSimObs_i', 'NOpSimObs_z', 'NOpSimObs_y'])
				self.csvwriter.writerow([self.OpSim.fieldID[OpSimi], self.OpSim.RA[OpSimi], self.OpSim.Dec[OpSimi], ng, self.OpSim.NobsDates[OpSimi]['u_'], self.OpSim.NobsDates[OpSimi]['g_'], self.OpSim.NobsDates[OpSimi]['r_'], self.OpSim.NobsDates[OpSimi]['i_'], self.OpSim.NobsDates[OpSimi]['z_'], self.OpSim.NobsDates[OpSimi]['y_']])

			output = cols

		elif (noRun):
			output = [-1 for x in range(len(cols))]

		else:
			output = [self.EB.period, self.EB.m1, self.EB.m2, self.EB.r1, self.EB.r2, self.EB.eccentricity, self.EB.inclination, self.EB.dist, self.EB.nobs, self.EB.AV, self.EB.M_H, self.EB.appMagMean['r_'], self.EB.maxDeltaMag, self.EB.deltaMag['r_'], self.EB.eclipseDepthFrac['r_'], self.EB.appmag_failed, self.EB.incl_failed, self.EB.period_failed, self.EB.radius_failed, self.EB.eclipseDepth_failed]

			#this is for gatspy
			for filt in self.filters:
				output.append(self.EB.LSS[filt]) 
			output.append(self.EB.LSM) 

		self.csvwriter.writerow(output)	


	def getGalaxy(self, OpSimi, deleteModel = True, downloadModel = True, area0frac = 1.):
		self.Galaxy = TRILEGAL()
		self.Galaxy.RA = self.OpSim.RA[OpSimi]
		self.Galaxy.Dec = self.OpSim.Dec[OpSimi]
		self.Galaxy.fieldID = self.OpSim.fieldID[OpSimi]
		self.Galaxy.tmpdir = self.galDir
		self.Galaxy.tmpfname = 'TRILEGAL_model_fID'+str(int(self.OpSim.fieldID[OpSimi]))+'.h5'
		self.Galaxy.deleteModel = deleteModel
		self.Galaxy.seeing = self.seeing
		self.Galaxy.area0frac = area0frac
		self.Galaxy.setModel(download = downloadModel)


	def makeBinaryFromGalaxy(self, s):
		#uniform distribution for mass ratio
		def getq():
			return np.random.random()

		#uniform distribution for eccentricity
		def getecc():
			return np.random.random()

		#log-normal distribution for period
		def getlogp(maxlP = np.log10(self.totaltime)):#10yr
			x = 2*maxlP
			while (x > maxlP):
				x = np.random.normal(loc=5.03, scale=2.28)
			return x

		def getRad(logg, m):
			#g = GM/r**2
			g = 10.**logg * u.cm/u.s**2.
			r = ((constants.G*m*u.Msun/g)**0.5).decompose().to(u.Rsun).value
			return r

		nWarn = 0.
		maxTol = self.mTol
		m2Use = 0.5
		mTolUse = 0.001

		m1 = s['Mact'].iloc[0]
		rad1 = getRad(s['logg'].iloc[0], s['Mact'].iloc[0])
		lum1 = 10.**s['logL'].iloc[0]
		teff1 = 10.**s['logTe'].iloc[0]
		logg1 = s['logg'].iloc[0]
		rmag = s['r_mag'].iloc[0]
		m2 = None
		rad2 = None
		lum2 = None
		teff2 = None
		logg2 = None

		m2Use = s['Mact'].iloc[0]*getq()
		done = False
		mTolUse = self.mTol
		counter = 0.
		maxCount = 100
		while (not done) and (counter < maxCount):
			df_sort = self.Galaxy.model.loc[ (self.Galaxy.model['Mact'] - m2Use).abs() < mTolUse]
			if (len(df_sort) > 0):
				done = True
				ss = df_sort.sample()
				m2 = ss['Mact'].iloc[0]
				rad2 = getRad(ss['logg'].iloc[0], ss['Mact'].iloc[0])
				lum2 = 10.**ss['logL'].iloc[0]
				teff2 = 10.**ss['logTe'].iloc[0]
				logg2 = ss['logg'].iloc[0]
			else:
				#print('WARNING: increasing tolerance', mTolUse)
				mTolUse *=2
				nWarn += 1
				maxTol = np.max([mTolUse, maxTol])
			if (counter > maxCount):
				print('WARNING: did not reach tolerance, will probably die...')
				done = True
			counter += 1
		if (maxTol > 0.1):
			print(f'WARNING: had to increase mass tolerance {nWarn} times. Max tolerance = {maxTol}.')
		logp = getlogp()
		#some accounting for tides
		if (logp <= 1):
			ecc = 0.
		else:
			ecc = getecc()
		
		dist = 10.**s['logDist'].iloc[0]
		Av = s['Av'].iloc[0] #is this measure OK?
		MH = s['[M/H]'].iloc[0]

		#random angles
		inc = np.arccos(2.*np.random.uniform(0,1) - 1.)
		omega = np.random.uniform(0,2*np.pi)
		OMEGA = np.random.uniform(0,2*np.pi)

		return {
			'm1':m1,
			'rad1':rad1,
			'lum1':lum1,
			'teff1':teff1,
			'logg1':logg1,
			'm2':m2,
			'rad2':rad2,
			'lum2':lum2,
			'teff2':teff2,
			'logg2':logg2,
			'logp':logp,
			'ecc':ecc,
			'inc':inc,
			'omega':omega,
			'OMEGA':OMEGA,
			'dist':dist,
			'rmag':rmag,
			'Av':Av,
			'MH':MH
		}

	def sampleGalaxy(self):


		#for the binary fraction
		def fitRagfb():
			x = [0.05, 0.1, 1, 8, 15]  #estimates of midpoints in bins, and using this: https://sites.uni.edu/morgans/astro/course/Notes/section2/spectralmasses.html
			y = [0.20, 0.35, 0.50, 0.70, 0.75]
			init = models.PowerLaw1D(amplitude=0.5, x_0=1, alpha=-1.)
			fitter = fitting.LevMarLSQFitter()
			fit = fitter(init, x, y)

			return fit


#test for the Sun
#print(getRad(4.43, 1)) 
		print("creating binaries...")


		fbFit= fitRagfb()
		#print(fbFit)

		m1 = []
		rad1 = []
		lum1 = []
		teff1 = []
		logg1 = []

		m2 = []
		rad2 = []
		lum2 = []
		teff2 = []
		logg2 = []

		logp = []
		ecc = []
		inc = []
		omega = []
		OMEGA = []

		rmag = [] #so that I can use this rmag for binaries that don't pass the other criteria?
		dist = []
		Av = [] 
		MH = []

		Ntrial = 0 #for safety
		while (len(m1) < self.n_bin) and (Ntrial < 100*self.n_bin):
			Ntrial += 1
			s = self.Galaxy.model.sample()
			fb = fbFit(s['m_ini'].iloc[0]) #I think I should base this on the initial mass, since these binary fractions are for unevolved stars
			xx = np.random.random()
			if (xx < fb):
				binary = self.makeBinaryFromGalaxy(s)
				if (binary['m2'] is not None):
					m1.append(binary['m1'])
					rad1.append(binary['rad1'])
					lum1.append(binary['lum1'])
					teff1.append(binary['teff1'])
					logg1.append(binary['logg1'])

					m2.append(binary['m2'])
					rad2.append(binary['rad2'])
					lum2.append(binary['lum2'])
					teff2.append(binary['teff2'])
					logg2.append(binary['logg2'])

					logp.append(binary['logp'])
					ecc.append(binary['ecc'])
					inc.append(binary['inc'])
					omega.append(binary['omega'])
					OMEGA.append(binary['OMEGA'])

					rmag.append(binary['rmag'])
					dist.append(binary['dist'])
					Av.append(binary['Av'])
					MH.append(binary['MH'])


		m1 = np.array(m1)
		rad1 = np.array(rad1)
		lum1 = np.array(lum1)
		logg1 = np.array(logg1)
		m2 = np.array(m2)
		rad2 = np.array(rad2)
		lum2 = np.array(lum2)
		logg2 = np.array(logg2)
		logp = np.array(logp)
		ecc = np.array(ecc)
		inc = np.array(inc)
		omega = np.array(omega)
		OMEGA = np.array(OMEGA)
		dist = np.array(dist)
		Av = np.array(Av)
		MH = np.array(MH)
		rmag = np.array(rmag)

		#filler
		x = np.zeros(self.n_bin)

		#we don't need position, but we do need distance
		output = np.vstack( (m1, m2, logp, ecc, rad1, rad2, lum1, lum2, x, x, x, dist, inc, OMEGA, omega, Av, MH, teff1, teff2, logg1, logg2, rmag) ).T

		return output

	def initialize(self, OpSimi=0):
		if (self.seed is None):
			np.random.seed()
			self.seed = np.random.randint(0,100000)
		else:
			np.random.seed(seed = self.seed)

		#OpSim
		self.OpSim.setDates(OpSimi, self.filters)
		print(f'total number of OpSim observation dates (all filters) = {self.OpSim.totalNobs[OpSimi]}')
		if (self.OpSim.totalNobs[OpSimi] < self.NobsLim):
			return False

		return True

