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

#for A_V
#import vespa.stars.extinction
from vespa_update import extinction

######################
#my code
from EclipsingBinary import EclipsingBinary
from OpSim import OpSim

#Andrew's code
from getClusterBinaries import getClusterBinaries

class LSSTEBClusterWorker(object):

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

		#cluster info
		self.clusterName = None
		self.clusterMass = None
		self.clusterDistance = None
		self.clusterMetallicity = None
		self.clusterAge = None
		self.clusterRhm = None
		self.clusterVdisp = None
		self.clusterType = None

		self.clusterAV = [None]

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
		EB.k1 = line[19]
		EB.k2 = line[20]
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


		if (EB.AV == None):
			EB.AV = self.clusterAV[OpSimi]

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
		cols = ['p', 'm1', 'm2', 'r1', 'r2', 'k1', 'k2', 'e', 'i', 'd', 'nobs','Av','[M/H]','appMagMean_r', 'maxDeltaMag', 'deltaMag_r','mag_failure', 'incl_failure', 'period_failure', 'radius_failure', 'u_LSS_PERIOD', 'g_LSS_PERIOD', 'r_LSS_PERIOD', 'i_LSS_PERIOD', 'z_LSS_PERIOD', 'y_LSS_PERIOD','LSM_PERIOD']
		if (header):
			if (self.useOpSimDates and self.OpSim != None):
				print("writing header")
				self.csvwriter.writerow(['OpSimID','OpSimRA','OpSimDec','NOpSimObs_u', 'NOpSimObs_g', 'NOpSimObs_r', 'NOpSimObs_i', 'NOpSimObs_z', 'NOpSimObs_y', 'clusterName', 'clusterMass', 'clusterDistance', 'clusterMetallicity','clusterAge', 'clusterRhm', 'clusterVdisp'])
				self.csvwriter.writerow([self.OpSim.fieldID[OpSimi], self.OpSim.RA[OpSimi], self.OpSim.Dec[OpSimi], self.OpSim.NobsDates[OpSimi]['u_'], self.OpSim.NobsDates[OpSimi]['g_'], self.OpSim.NobsDates[OpSimi]['r_'], self.OpSim.NobsDates[OpSimi]['i_'], self.OpSim.NobsDates[OpSimi]['z_'], self.OpSim.NobsDates[OpSimi]['y_'], self.clusterName[OpSimi], self.clusterMass[OpSimi], self.clusterDistance[OpSimi], self.clusterMetallicity[OpSimi], self.clusterAge[OpSimi], self.clusterRhm[OpSimi], self.clusterVdisp[OpSimi]])

			output = cols

		elif (noRun):
			output = [-1 for x in range(len(cols))]

		else:
			output = [EB.period, EB.m1, EB.m2, EB.r1, EB.r2, EB.k1, EB.k2, EB.eccentricity, EB.inclination, EB.dist, EB.nobs, EB.AV, EB.M_H, EB.appMagMean['r_'], EB.maxDeltaMag, EB.deltaMag['r_'],EB.appmag_failed, EB.incl_failed, EB.period_failed, EB.radius_failed]

			#this is for gatspt
			for filt in self.filters:
				output.append(EB.LSS[filt]) 
			output.append(EB.LSM) 

		self.csvwriter.writerow(output)	

	def sampleCluster(self, i):
		"""
		get the output from Andrew's cluster code
		should output as follows
		EB.m1 = line[0] #Msun
		EB.m2 = line[1] #Msun
		EB.period = 10.**line[2] #days
		EB.eccentricity = line[3] 
		EB.r1 = line[4] #Rsun
		EB.r2 = line[5] #Rsun
		EB.L1 = line[6] #Lsun
		EB.L2 = line[7] #Lsun
		EB.xGx = line[8] #unused kpc
		EB.yGx = line[9] #unused kpc
		EB.zGx = line[10] #unutsed kpc
		EB.dist = line[11] #kpc
		EB.inclination = line[12] *180./np.pi #degrees
		EB.OMEGA = line[13] *180./np.pi #degrees
		EB.omega = line[14] *180./np.pi #degrees
		EB.AV = line[15] #optional, if not available, make it None
		EB.M_H = line[16]
		"""
		print("sampling cluster", self.clusterName[i])
		sampler = getClusterBinaries(self.clusterAge[i], self.clusterMetallicity[i], self.clusterVdisp[i], self.n_bin)
		sampler.random_seed = self.seed
		sampler.dist = self.clusterDistance[i]
		sampler.runAll()

		return sampler.output


	def initialize(self, OpSimi=0):
		if (self.seed == None):
			np.random.seed()
		else:
			np.random.seed(seed = self.seed)


		#get the extinction
		count = 0
		if (len(self.clusterAV) < len(self.clusterName)):
			self.clusterAV = np.array([None for x in range(len(self.clusterName))])
		
		while (self.clusterAV[OpSimi] == None and count < 100):
			self.clusterAV[OpSimi] = extinction.get_AV_infinity(self.OpSim.RA[OpSimi], self.OpSim.Dec[OpSimi], frame='icrs')
			if (self.clusterAV[OpSimi] == None):
				print("WARNING: No AV found", self.OpSim.RA[OpSimi], self.OpSim.Dec[OpSimi], self.clusterAV[OpSimi], count)

		self.OpSim.setDates(OpSimi, self.filters)
		print(f'total number of OpSim observation dates (all filters) = {self.OpSim.totalNobs[OpSimi]}')
		if (self.OpSim.totalNobs[OpSimi] < self.NobsLim):
			return False

		return True

