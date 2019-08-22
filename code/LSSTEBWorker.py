import numpy as np
import pandas as pd
import scipy.special as ss
import datetime

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


#######################
#3rd party codes
from gatspy.periodic import LombScargleMultiband, LombScargle, LombScargleFast, LombScargleMultibandFast
import emcee

######################
#my code
import EclipsingBinary
import OpSim
import TRILEGAL
import BreivikGalaxy

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

		self.filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']


		self.n_bin = 100000
		self.n_band = 2
		self.n_base = 2  
		self.n_cores = 1

		self.ofile = 'output_file.csv' #output file name
		self.dbFile = '../input/db/baseline2018a.db' #for the OpSim database
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
			EB.observe(filt)
			EB.LSS[filt] = -999.

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
		EB.period = 10.**line[2] #days
		EB.eccentricity = line[3]
		EB.inclination = line[12] *180./np.pi #degrees
		EB.OMEGA = line[13] *180./np.pi #degrees
		EB.omega = line[14] *180./np.pi #degrees

		EB.dist = line[11] #kpc
		if (self.Galaxy == None):
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
			if (self.Galaxy != None):
				ng = self.Galaxy.Nstars
			if (self.useOpSimDates and self.OpSim != None):
				print("writing header")
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
		_ = self.emcee_sampler.run_mcmc(walkers.values, self.emcee_nsamples)

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
		if (self.seed == None):
			np.random.seed()
		else:
			np.random.seed(seed = self.seed)


		if (self.useOpSimDates and self.OpSim == None):
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
		if (self.Galaxy == None):
			self.Galaxy = TRILEGAL()
			self.Galaxy.RA = self.OpSim.RA[OpSimi]
			self.Galaxy.Dec = self.OpSim.Dec[OpSimi]
			self.Galaxy.fieldID = self.OpSim.fieldID[OpSimi]
			self.Galaxy.tmpdir = self.galDir
			self.Galaxy.tmpfname = 'TRILEGAL_model_fID'+str(int(self.OpSim.fieldID[OpSimi]))+'.h5'
			self.Galaxy.setModel()	

		if (self.Breivik == None):
			self.Breivik = BreivikGalaxy()
			self.Breivik.GalaxyFile = self.GalaxyFile
			self.Breivik.GalaxyFileLogPrefix = self.GalaxyFileLogPrefix
			self.Breivik.setKernel()

		if (self.BreivikGal == None):
			self.matchBreivikTRILEGAL()

		return True
