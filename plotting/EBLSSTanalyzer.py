# should add Andrew's corner plots on here!

import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy import units, constants
from astropy.modeling import models, fitting
import scipy.stats
from scipy.integrate import quad

#for Quest
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class EBLSSTanalyzer(object):

	def __init__(self, 
				directory = 'output_files', 
				plotsDirectory ='plots',
				doIndividualPlots = True,
				cluster = False,
				mMean = 0.5,
				filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_', 'all'],
				Pcut = 0.1,
				Nlim = 1,
				onlyDWD = False):

		self.directory = directory
		self.plotsDirectory = plotsDirectory
		self.doIndividualPlots = doIndividualPlots
		self.cluster = cluster
		self.mMean = mMean #assumed mean stellar mass (for cluster analysis)
		self.filters = filters
		self.Pcut = Pcut #cutoff in percent error for "recovered"
		self.Nlim = Nlim #minimum number of lines to consider (for all, obs, rec, etc.)
		self.onlyDWD = onlyDWD

		self.outputNumbers = dict()
		self.outputNumbers['RA'] = []
		self.outputNumbers['Dec'] = []
		self.outputNumbers['recFrac'] = []
		self.outputNumbers['recN'] = []
		self.outputNumbers['rawN'] = []
		self.outputNumbers['obsN'] = []
		self.outputNumbers['fileN'] = []
		self.outputNumbers['fileObsN'] = []
		self.outputNumbers['fileRecN'] = []
		self.outputNumbers['allNPrsa'] = []
		self.outputNumbers['obsNPrsa'] = []
		self.outputNumbers['recNPrsa'] = []
		self.outputNumbers['allNDWD'] = []
		self.outputNumbers['obsNDWD'] = []
		self.outputNumbers['recNDWD'] = []

		self.outputHists = dict()

		self.individualPlots = dict()

	def file_len(self, fname):
		i = 0
		with open(fname) as f:
			for i, l in enumerate(f):
				pass
		return i + 1

	def getPhs(self, sigma, m1=1*units.solMass, m2=1*units.solMass, m3=0.5*units.solMass):
		Phs = np.pi*constants.G/np.sqrt(2.)*(m1*m2/m3)**(3./2.)*(m1 + m2)**(-0.5)*sigma**(-3.)
		return Phs.decompose().to(units.day)

	def fitRagfb(self):
		x = [0.05, 0.1, 1, 8, 15]  #estimates of midpoints in bins, and using this: https://sites.uni.edu/morgans/astro/course/Notes/section2/spectralmasses.html
		y = [0.20, 0.35, 0.50, 0.70, 0.75]
		init = models.PowerLaw1D(amplitude=0.5, x_0=1, alpha=-1.)
		fitter = fitting.LevMarLSQFitter()
		fit = fitter(init, x, y)

		return fit

	def RagNormal(self, x, cdf = False):
		mean = 5.03
		std = 2.28
		if (cdf):
			return scipy.stats.norm.cdf(x,mean,std)

		return scipy.stats.norm.pdf(x,mean,std)



	def plotObsRecRatio(self, d, key, xtitle, fname, xlim = None):
		c1 = '#0294A5'  #turqoise
		c2 = '#d95f02' #orange from color brewer
		c3 = '#00353E' #slate
		c4 = '#508201' #olive
		f,(ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5, 12), sharex=True)

		histAll = d[key+'hAll']
		histObs = d[key+'hObs'] 
		allhistRec = d[key+'hRec']['all']
		bin_edges = d[key+'b']

		binHalf = (bin_edges[1] - bin_edges[0])/2.

		#CDF
		cdfAll = []
		cdfObs = []
		cdfRec = []
		for i in range(len(histAll)):
			cdfAll.append(np.sum(histAll[:i])/np.sum(histAll))
		for i in range(len(histObs)):
			cdfObs.append(np.sum(histObs[:i])/np.sum(histObs))
		for i in range(len(allhistRec)):
			cdfRec.append(np.sum(allhistRec[:i])/np.sum(allhistRec))
		ax1.step(bin_edges, cdfAll, color=c1, label='All')
		ax1.step(bin_edges, cdfObs, color=c2, label='Observable')
		ax1.step(bin_edges, cdfRec, color=c3, label='Recoverable')
		ax1.set_ylabel('CDF')


		#PDF
		ax2.step(bin_edges, histAll/np.sum(histAll), color=c1, label='All')
		ax2.step(bin_edges, histObs/np.sum(histObs), color=c2, label='Observable')
		ax2.step(bin_edges, allhistRec/np.sum(allhistRec), color=c3, label='Recoverable')
		ax2.set_ylabel('PDF')
		ax2.set_yscale('log')

		#ratio
		#prepend some values at y of zero so that the step plots look correct
		use = np.where(histAll > 0)[0]
		if (len(use) > 0):
			b = bin_edges[use]
			r = histObs[use]/histAll[use]
			ax3.step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c2, label='Observable/All')
			ax3.plot(b - binHalf, r, 'o',color=c1, markersize=5, markeredgecolor=c2)

			r = allhistRec[use]/histAll[use]
			ax3.step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c3, label='Recoverable/All')
			ax3.plot(b - binHalf, r, 'o',color=c1, markersize=5, markeredgecolor=c3)

			use = np.where(histObs > 0)[0]
			if (len(use) > 0):
				b = bin_edges[use]
				r =  allhistRec[use]/histObs[use]
				ax3.step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c3, label='Recoverable/Observable')
				ax3.plot(b - binHalf, r, 'o',color=c2, markersize=5, markeredgecolor=c3)
		#ax3.step(bin_edges[use], histRec[use]/histObs[use], color=c2, linestyle='--', dashes=(3, 3), linewidth=4)

		ax3.set_ylabel('Ratio')
		ax3.set_yscale('log')
		ax3.set_ylim(10**-4,1)
		ax3.set_xlabel(xtitle)

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		lAll = mlines.Line2D([], [], color=c1, label='All')
		lObs = mlines.Line2D([], [], color=c2, label='Obs.')
		lRec = mlines.Line2D([], [], color=c3, label='Rec.')
		lObsAll = mlines.Line2D([], [], color=c2, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c2, label='Obs./All')
		lRecAll = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c3, label='Rec./All')
		lRecObs = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c2, markersize=5, markeredgecolor=c3, label='Rec./Obs.')
		ax1.legend(handles=[lAll, lObs, lRec, lObsAll, lRecAll, lRecObs], loc='lower right')

		if (xlim != None):
			ax1.set_xlim(xlim[0], xlim[1])
			ax2.set_xlim(xlim[0], xlim[1])
			ax3.set_xlim(xlim[0], xlim[1])

		f.subplots_adjust(hspace=0)
		print(fname)
		f.savefig(fname+'_ObsRecRatio.pdf',format='pdf', bbox_inches = 'tight')
		plt.close(f)


	def plotObsRecOtherRatio(self, d1, d2, key, xtitle, fname, xlim = None, legendLoc='lower right'):
		c1 = '#0294A5'  #turqoise
		c2 = '#d95f02' #orange from color brewer
		c3 = '#00353E' #slate
		c4 = '#508201' #olive

		f,(ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5, 12), sharex=True)

		histAll = d1[key+'hAll']
		histObs = d1[key+'hObs']
		allhistRec = d1[key+'hRec']['all']
		bin_edges = d1[key+'b']

		histAllOD = d2[key+'hAll']
		histObsOD = d2[key+'hObs']
		allhistRecOD = d2[key+'hRec']['all']
		bin_edgesOD = d2[key+'b']		

		binHalf = (bin_edges[1] - bin_edges[0])/2.
		binHalfOD = (bin_edgesOD[1] - bin_edgesOD[0])/2.

		#CDF
		cdfAll = []
		cdfObs = []
		cdfRec = []
		cdfAllOD = []
		cdfObsOD = []
		cdfRecOD = []
		for i in range(len(histAll)):
			cdfAll.append(np.sum(histAll[:i])/np.sum(histAll))
		for i in range(len(histAllOD)):
			cdfAllOD.append(np.sum(histAllOD[:i])/np.sum(histAllOD))
		for i in range(len(histObs)):
			cdfObs.append(np.sum(histObs[:i])/np.sum(histObs))
		for i in range(len(histObsOD)):
			cdfObsOD.append(np.sum(histObsOD[:i])/np.sum(histObsOD))
		for i in range(len(allhistRec)):
			cdfRec.append(np.sum(allhistRec[:i])/np.sum(allhistRec))
		for i in range(len(allhistRecOD)):
			cdfRecOD.append(np.sum(allhistRecOD[:i])/np.sum(allhistRecOD))
		ax1.step(bin_edges, cdfAll, color=c1, label='All')
		ax1.step(bin_edges, cdfObs, color=c2, label='Observable')
		ax1.step(bin_edges, cdfRec, color=c3, label='Recoverable')
		ax1.step(bin_edgesOD, cdfAllOD, color=c1, linestyle=':')
		ax1.step(bin_edgesOD, cdfObsOD, color=c2, linestyle=':')
		ax1.step(bin_edgesOD, cdfRecOD, color=c3, linestyle=':')
		ax1.set_ylabel('CDF', fontsize=16)
		ax1.set_ylim(-0.01,1.01)

		#PDF --need to divide by the bin size
		# ax2.step(bin_edges, histAll/np.sum(histAll)/np.diff(bin_edges)[0], color=c1, label='All')
		# ax2.step(bin_edges, histObs/np.sum(histObs)/np.diff(bin_edges)[0], color=c2, label='Observable')
		# ax2.step(bin_edges, allhistRec/np.sum(allhistRec)/np.diff(bin_edges)[0], color=c3, label='Recoverable')
		# ax2.step(bin_edgesOD, histAllOD/np.sum(histAllOD)/np.diff(bin_edgesOD)[0], color=c1, linestyle=':')
		# ax2.step(bin_edgesOD, histObsOD/np.sum(histObsOD)/np.diff(bin_edgesOD)[0], color=c2, linestyle=':')
		# ax2.step(bin_edgesOD, allhistRecOD/np.sum(allhistRecOD)/np.diff(bin_edgesOD)[0], color=c3, linestyle=':')
		# ax2.set_ylabel('PDF', fontsize=16)
		#this is the fraction in each bin
		ax2.step(bin_edges, histAll/np.sum(histAll), color=c1, label='All')
		ax2.step(bin_edges, histObs/np.sum(histObs), color=c2, label='Observable')
		ax2.step(bin_edges, allhistRec/np.sum(allhistRec), color=c3, label='Recoverable')
		ax2.step(bin_edgesOD, histAllOD/np.sum(histAllOD), color=c1, linestyle=':')
		ax2.step(bin_edgesOD, histObsOD/np.sum(histObsOD), color=c2, linestyle=':')
		ax2.step(bin_edgesOD, allhistRecOD/np.sum(allhistRecOD), color=c3, linestyle=':')
		ax2.set_ylabel('PDF', fontsize=16)
		ax2.set_yscale('log')
		ax2.set_ylim(0.5e-5, 0.9)


		ratio = histObs/histAll
		check = np.isnan(ratio)
		ratio[check]=0.
		ax3.step(bin_edges, ratio, color=c2, label='Observable/All')
		ax3.plot(bin_edges - binHalf, ratio, 'o',color=c1, markersize=5, markeredgecolor=c2)

		ratio = allhistRec/histAll
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax3.step(bin_edges, ratio, color=c3, label='Recoverable/All')
		ax3.plot(bin_edges - binHalf, ratio, 'o',color=c1, markersize=5, markeredgecolor=c3)

		ratio = allhistRec/histObs
		check = np.isnan(ratio)
		ratio[check]=0.
		ax3.step(bin_edges, ratio, color=c3, label='Recoverable/Observable')
		ax3.plot(bin_edges - binHalf, ratio, 'o',color=c2, markersize=5, markeredgecolor=c3)

		ratio = histObsOD/histAllOD
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax3.step(bin_edgesOD, ratio, color=c2, linestyle=':')
		ax3.plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c1, markersize=3.5, markeredgecolor=c2)

		ratio = allhistRecOD/histAllOD
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax3.step(bin_edgesOD, ratio, color=c3, linestyle=':')
		ax3.plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c1, markersize=3.5, markeredgecolor=c3)

		ratio = allhistRecOD/histObsOD
		check = np.isnan(ratio)
		ratio[check]=0.
		ax3.step(bin_edgesOD, ratio, color=c3, linestyle=':')
		ax3.plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c2, markersize=3.5, markeredgecolor=c3)


		ax3.set_ylabel('Ratio', fontsize=16)
		ax3.set_yscale('log')
		ax3.set_ylim(0.5e-5,1)
		ax3.set_xlabel(xtitle, fontsize=16)

		if (xlim != None):
			ax1.set_xlim(xlim[0], xlim[1])
			ax2.set_xlim(xlim[0], xlim[1])
			ax3.set_xlim(xlim[0], xlim[1])

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		lAll = mlines.Line2D([], [], color=c1, label='All')
		lObs = mlines.Line2D([], [], color=c2, label='Obs.')
		lRec = mlines.Line2D([], [], color=c3, label='Rec.')
		lObsAll = mlines.Line2D([], [], color=c2, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c2, label='Obs./All')
		lRecAll = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c3, label='Rec./All')
		lRecObs = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c2, markersize=5, markeredgecolor=c3, label='Rec./Obs.')
		ax1.legend(handles=[lAll, lObs, lRec, lObsAll, lRecAll, lRecObs], loc=legendLoc)


		f.subplots_adjust(hspace=0)
		f.savefig(fname+'_ObsRecOtherRatio.pdf',format='pdf', bbox_inches = 'tight')
		plt.close(f)

	def saveHist(self, d, key, xtitle, fname, xlim = None, filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_','all']):
		c1 = '#0294A5'  #turqoise
		c2 = '#d95f02' #orange from color brewer
		c3 = '#00353E' #slate
		fig,(ax1, ax2) = plt.subplots(2,1,figsize=(5, 8), sharex=True)

		histAll = d[key+'hAll']
		histObs = d[key+'hObs']
		histRec = d[key+'hRec']
		bin_edges = d[key+'b']

		#PDF
		ax1.step(bin_edges, histAll/np.sum(histAll), color=c1)
		ax1.step(bin_edges, histObs/np.sum(histObs), color=c2)
		for f in filters:
			lw = 1
			if (f == 'all'):
				lw = 0.5
			ax1.step(bin_edges, histRec[f]/np.sum(histRec[f]), color=c3, linewidth=lw)
		ax1.set_ylabel('PDF')
		ax1.set_yscale('log')

		#CDF
		cdfAll = []
		cdfObs = []
		cdfRec = dict()
		for f in filters:
			cdfRec[f] = []

		for i in range(len(histAll)):
			cdfAll.append(np.sum(histAll[:i])/np.sum(histAll))
		for i in range(len(histObs)):
			cdfObs.append(np.sum(histObs[:i])/np.sum(histObs))
		for f in filters:
			for i in range(len(histRec[f])):
				cdfRec[f].append(np.sum(histRec[f][:i])/np.sum(histRec[f]))
		ax2.step(bin_edges, cdfAll, color=c1)
		ax2.step(bin_edges, cdfObs, color=c2)
		for f in filters:
			lw = 1
			if (f == 'all'):
				lw = 0.5
			ax2.step(bin_edges, cdfRec[f], color=c3, linewidth=lw)
		ax2.set_ylabel('CDF')

		ax2.set_xlabel(xtitle)
		fig.subplots_adjust(hspace=0)
		fig.savefig(fname+'.pdf',format='pdf', bbox_inches = 'tight')
		plt.close(fig)

		#write to a text file (I could do this much easier in pandas, but this already works)
		with open(fname+'.csv','w') as fl:
			outline = 'binEdges,histAll,histObs'
			for f in filters:
				outline += ','+f+'histRec'
			outline += '\n'
			fl.write(outline)
			for i in range(len(bin_edges)):
				outline = str(bin_edges[i])+','+str(histAll[i])+','+str(histObs[i])
				for f in filters:
					outline += ','+str(histRec[f][i])
				outline += '\n'
				fl.write(outline)

		self.plotObsRecRatio(d, key, xtitle, fname, xlim)

	def compileData(self):
		print(self.directory)

		#get the Raghavan binary fraction fit
		fbFit = self.fitRagfb()
		print(fbFit)
			
		if (self.doIndividualPlots):
			fmass, axmass = plt.subplots()
			fqrat, axqrat = plt.subplots()
			fecc, axecc = plt.subplots()
			flper, axlper = plt.subplots()
			fdist, axdist = plt.subplots()
			fmag, axmag = plt.subplots()
			frad, axrad = plt.subplots()

		#bins for all the histograms
		mbins = np.arange(0,10, 0.1, dtype='float')
		qbins = np.arange(0,1, 0.1, dtype='float')
		ebins = np.arange(0, 1.05, 0.05, dtype='float')
		lpbins = np.arange(-3, 10, 0.5, dtype='float')
		dbins = np.arange(0, 40, 1, dtype='float')
		magbins = np.arange(11, 25, 1, dtype='float')
		rbins = np.arange(0, 100, 0.2, dtype='float')

		# mbins = np.arange(0,10, 0.1, dtype='float')
		# qbins = np.arange(0,2, 0.05, dtype='float')
		# ebins = np.arange(0, 1.05, 0.05, dtype='float')
		# lpbins = np.arange(-2, 10, 0.2, dtype='float')
		# dbins = np.arange(0, 40, 1, dtype='float')
		# magbins = np.arange(11, 25, 0.5, dtype='float')
		# rbins = np.arange(0, 100, 0.2, dtype='float')

		#blanks for the histograms
		#All
		m1hAll = np.zeros_like(mbins)[1:]
		qhAll = np.zeros_like(qbins)[1:]
		ehAll = np.zeros_like(ebins)[1:]
		lphAll = np.zeros_like(lpbins)[1:]
		dhAll = np.zeros_like(dbins)[1:]
		maghAll = np.zeros_like(magbins)[1:]
		rhAll = np.zeros_like(rbins)[1:]
		#Observable
		m1hObs = np.zeros_like(mbins)[1:]
		qhObs = np.zeros_like(qbins)[1:]
		ehObs = np.zeros_like(ebins)[1:]
		lphObs = np.zeros_like(lpbins)[1:]
		dhObs = np.zeros_like(dbins)[1:]
		maghObs = np.zeros_like(magbins)[1:]
		rhObs = np.zeros_like(rbins)[1:]
		#Recovered
		m1hRec = dict()
		qhRec = dict()
		ehRec = dict()
		lphRec = dict()
		dhRec = dict()
		maghRec = dict()
		rhRec = dict()
		for f in self.filters:
			m1hRec[f] = np.zeros_like(mbins)[1:]
			qhRec[f] = np.zeros_like(qbins)[1:]
			ehRec[f] = np.zeros_like(ebins)[1:]
			lphRec[f] = np.zeros_like(lpbins)[1:]
			dhRec[f] = np.zeros_like(dbins)[1:]
			maghRec[f] = np.zeros_like(magbins)[1:]
			rhRec[f] = np.zeros_like(rbins)[1:]

		#Read in all the data and make the histograms
		files = os.listdir(self.directory)
		IDs = []
		for i, f in enumerate(files):
			print(round(i/len(files),4), f)
			fl = self.file_len(os.path.join(self.directory,f))
			if (fl >= 4):
				#read in the header
				header = pd.read_csv(os.path.join(self.directory,f), nrows=1)
		######################
		#NEED TO ACCOUNT FOR THE BINARY FRACTION when combining histograms
		#####################
				if (self.cluster):
					Nmult = header['clusterMass'][0]/self.mMean
				else:
					Nmult = header['NstarsTRILEGAL'][0]
				#Nmult = 1.


				#to normalize for the difference in period, since I limit this for the field for computational efficiency (but not for clusters, which I allow to extend to the hard-soft boundary)
				intNorm = 1.
				if (not self.cluster):
					intAll, err = quad(self.RagNormal, -20, 3650.)
					intCut, err = quad(self.RagNormal, -20, 20)
					intNorm = intCut/intAll


				rF = 0.
				rN = 0.
				raN = 0.
				raN = 0.
				obN = 0.
				fiN = 0.
				fioN = 0.
				firN = 0.
				Nall = 0.
				Nrec = 0.
				Nobs = 0.
				NallPrsa = 0.
				NobsPrsa = 0.
				NrecPrsa = 0.
				NallDWD = 0.
				NobsDWD = 0.
				NrecDWD = 0.

				#read in rest of the file
				data = pd.read_csv(os.path.join(self.directory,f), header = 2).fillna(-999)
				if (data['m1'][0] != -1): #these are files that were not even run
					data['r1'].replace(0., 1e-10, inplace = True)
					data['r2'].replace(0., 1e-10, inplace = True)
					data['m1'].replace(0., 1e-10, inplace = True)
					data['m2'].replace(0., 1e-10, inplace = True)
					logg1 = np.log10((constants.G*data['m1'].values*units.solMass/(data['r1'].values*units.solRad)**2.).decompose().to(units.cm/units.s**2.).value)
					logg2 = np.log10((constants.G*data['m2'].values*units.solMass/(data['r2'].values*units.solRad)**2.).decompose().to(units.cm/units.s**2.).value)
					data['logg1'] = logg1
					data['logg2'] = logg2

					Nall = len(data.index)/intNorm  #saving this in case we want to limit the entire analysis to DWDs, but still want the full sample size for the cumulative numbers

					if (self.onlyDWD):
						data = data.loc[(data['logg1'] > 6) & (data['logg1'] < 10) & (data['logg2'] > 6) & (data['logg2'] < 10)]

					prsa = data.loc[(data['appMagMean_r'] <= 19.5) & (data['appMagMean_r'] > 15.8) & (data['p'] < 1000) & (data['p'] > 0.5)]
					DWD = data.loc[(data['logg1'] > 6) & (data['logg1'] < 10) & (data['logg2'] > 6) & (data['logg2'] < 10)]

###is this correct? (and the only place I need to normalize?) -- I think yes (the observed binary distribution should be cut at a period of the survey duration)
					NallPrsa = len(prsa.index)/intNorm
					NallDWD = len(DWD.index)/intNorm



					if (len(data.index) >= self.Nlim):
						#create histograms
						#All
						m1hAll0, m1b = np.histogram(data["m1"], bins=mbins)
						qhAll0, qb = np.histogram(data["m2"]/data["m1"], bins=qbins)
						ehAll0, eb = np.histogram(data["e"], bins=ebins)
						lphAll0, lpb = np.histogram(np.ma.log10(data["p"].values).filled(-999), bins=lpbins)
						dhAll0, db = np.histogram(data["d"], bins=dbins)
						maghAll0, magb = np.histogram(data["appMagMean_r"], bins=magbins)
						rhAll0, rb = np.histogram(data["r2"]/data["r1"], bins=rbins)

						if (self.doIndividualPlots):
							axmass.step(m1b[0:-1], m1hAll0/np.sum(m1hAll0), color='black', alpha=0.1)
							axqrat.step(qb[0:-1], qhAll0/np.sum(qhAll0), color='black', alpha=0.1)
							axecc.step(eb[0:-1], ehAll0/np.sum(ehAll0), color='black', alpha=0.1)
							axlper.step(lpb[0:-1], lphAll0/np.sum(lphAll0), color='black', alpha=0.1)
							axdist.step(db[0:-1], dhAll0/np.sum(dhAll0), color='black', alpha=0.1)
							axmag.step(magb[0:-1], maghAll0/np.sum(maghAll0), color='black', alpha=0.1)
							axrad.step(rb[0:-1], rhAll0/np.sum(rhAll0), color='black', alpha=0.1)

						#account for the binary fraction, as a function of mass
						dm1 = np.diff(m1b)
						m1val = m1b[:-1] + dm1/2.
						fb = np.sum(m1hAll0/len(data.index)*fbFit(m1val))
						if (self.cluster):
							#account for the hard-soft boundary
							Phs = self.getPhs(header['clusterVdisp'].iloc[0]*units.km/units.s).to(units.day).value
							fb *= self.RagNormal(np.log10(Phs), cdf = True)
							print("   fb, Nbins, log10(Phs) = ", fb, len(data.index), np.log10(Phs))
						Nmult *= fb

									
						m1hAll += m1hAll0/Nall*Nmult
						qhAll += qhAll0/Nall*Nmult
						ehAll += ehAll0/Nall*Nmult
						lphAll += lphAll0/Nall*Nmult
						dhAll += dhAll0/Nall*Nmult
						maghAll += maghAll0/Nall*Nmult
						rhAll += rhAll0/Nall*Nmult

						#Obs
						#I want to account for all filters here too (maybe not necessary; if LSM is != -999 then they are all filled in, I think)...
						obs = data.loc[(data['u_LSS_PERIOD'] != -999) | (data['g_LSS_PERIOD'] != -999) | (data['r_LSS_PERIOD'] != -999) | (data['i_LSS_PERIOD'] != -999) | (data['z_LSS_PERIOD'] != -999) | (data['y_LSS_PERIOD'] != -999) | (data['LSM_PERIOD'] != -999)]
						prsaObs = data.loc[(data['appMagMean_r'] <= 19.5) & (data['appMagMean_r'] > 15.8) & (data['p'] < 1000) & (data['p'] >0.5) & ((data['u_LSS_PERIOD'] != -999) | (data['g_LSS_PERIOD'] != -999) | (data['r_LSS_PERIOD'] != -999) | (data['i_LSS_PERIOD'] != -999) | (data['z_LSS_PERIOD'] != -999) | (data['y_LSS_PERIOD'] != -999) | (data['LSM_PERIOD'] != -999))]
						DWDObs = data.loc[(data['logg1'] > 6) & (data['logg1'] < 10) & (data['logg2'] > 6) & (data['logg2'] < 10) & ((data['u_LSS_PERIOD'] != -999) | (data['g_LSS_PERIOD'] != -999) | (data['r_LSS_PERIOD'] != -999) | (data['i_LSS_PERIOD'] != -999) | (data['z_LSS_PERIOD'] != -999) | (data['y_LSS_PERIOD'] != -999) | (data['LSM_PERIOD'] != -999))]

						Nobs = len(obs.index)
						NobsPrsa = len(prsaObs.index)
						NobsDWD = len(DWDObs.index)
						if (Nobs >= self.Nlim):
							m1hObs0, m1b = np.histogram(obs["m1"], bins=mbins)
							qhObs0, qb = np.histogram(obs["m2"]/obs["m1"], bins=qbins)
							ehObs0, eb = np.histogram(obs["e"], bins=ebins)
							lphObs0, lpb = np.histogram(np.ma.log10(obs["p"].values).filled(-999), bins=lpbins)
							dhObs0, db = np.histogram(obs["d"], bins=dbins)
							maghObs0, magb = np.histogram(obs["appMagMean_r"], bins=magbins)
							rhObs0, rb = np.histogram(obs["r2"]/obs["r1"], bins=rbins)
							m1hObs += m1hObs0/Nall*Nmult
							qhObs += qhObs0/Nall*Nmult
							ehObs += ehObs0/Nall*Nmult
							lphObs += lphObs0/Nall*Nmult
							dhObs += dhObs0/Nall*Nmult
							maghObs += maghObs0/Nall*Nmult
							rhObs += rhObs0/Nall*Nmult

							#Rec
							recCombined = pd.DataFrame()
							prsaRecCombined = pd.DataFrame()
							DWDRecCombined = pd.DataFrame()
							for filt in self.filters:
								key = filt+'LSS_PERIOD'
								if (filt == 'all'):
									key = 'LSM_PERIOD'
								fullP = abs(data[key] - data['p'])/data['p']
								halfP = abs(data[key] - 0.5*data['p'])/(0.5*data['p'])
								twiceP = abs(data[key] - 2.*data['p'])/(2.*data['p'])
								rec = data.loc[(data[key] != -999) & ( (fullP < self.Pcut) | (halfP < self.Pcut) | (twiceP < self.Pcut))]
								prsaRec = data.loc[(data['appMagMean_r'] <= 19.5) & (data['appMagMean_r'] >15.8) & (data['p'] < 1000) & (data['p'] >0.5) & (data['LSM_PERIOD'] != -999) & ( (fullP < self.Pcut) | (halfP < self.Pcut) | (twiceP < self.Pcut))]
								DWDRec = data.loc[(data['logg1'] > 6) & (data['logg1'] < 10) & (data['logg2'] > 6) & (data['logg2'] < 10) & (data['LSM_PERIOD'] != -999) & ( (fullP < self.Pcut) | (halfP < self.Pcut) | (twiceP < self.Pcut))]

								if (len(rec) >= self.Nlim and filt != 'all'):
									m1hRec0, m1b = np.histogram(rec["m1"], bins=mbins)
									qhRec0, qb = np.histogram(rec["m2"]/rec["m1"], bins=qbins)
									ehRec0, eb = np.histogram(rec["e"], bins=ebins)
									lphRec0, lpb = np.histogram(np.ma.log10(rec["p"].values).filled(-999), bins=lpbins)
									dhRec0, db = np.histogram(rec["d"], bins=dbins)
									maghRec0, magb = np.histogram(rec["appMagMean_r"], bins=magbins)
									rhRec0, rb = np.histogram(rec["r2"]/rec["r1"], bins=rbins)
									m1hRec[filt] += m1hRec0/Nall*Nmult
									qhRec[filt] += qhRec0/Nall*Nmult
									ehRec[filt] += ehRec0/Nall*Nmult
									lphRec[filt] += lphRec0/Nall*Nmult
									dhRec[filt] += dhRec0/Nall*Nmult
									maghRec[filt] += maghRec0/Nall*Nmult
									rhRec[filt] += rhRec0/Nall*Nmult

								#I'd like to account for all filters here to have more accurate numbers
								recCombined = recCombined.append(rec)
								prsaRecCombined = prsaRecCombined.append(prsaRec)
								DWDRecCombined = DWDRecCombined.append(DWDRec)
								recCombined.drop_duplicates(inplace=True)
								prsaRecCombined.drop_duplicates(inplace=True)
								DWDRecCombined.drop_duplicates(inplace=True)
								if (len(recCombined) >= self.Nlim and filt == 'all'):
									Nrec = len(recCombined.index)
									NrecPrsa = len(prsaRecCombined.index)
									NrecDWD = len(DWDRecCombined.index)

									m1hRec0, m1b = np.histogram(recCombined["m1"], bins=mbins)
									qhRec0, qb = np.histogram(recCombined["m2"]/rec["m1"], bins=qbins)
									ehRec0, eb = np.histogram(recCombined["e"], bins=ebins)
									lphRec0, lpb = np.histogram(np.ma.log10(recCombined["p"].values).filled(-999), bins=lpbins)
									dhRec0, db = np.histogram(recCombined["d"], bins=dbins)
									maghRec0, magb = np.histogram(recCombined["appMagMean_r"], bins=magbins)
									rhRec0, rb = np.histogram(recCombined["r2"]/recCombined["r1"], bins=rbins)
									m1hRec[filt] += m1hRec0/Nall*Nmult
									qhRec[filt] += qhRec0/Nall*Nmult
									ehRec[filt] += ehRec0/Nall*Nmult
									lphRec[filt] += lphRec0/Nall*Nmult
									dhRec[filt] += dhRec0/Nall*Nmult
									maghRec[filt] += maghRec0/Nall*Nmult
									rhRec[filt] += rhRec0/Nall*Nmult

					rF = Nrec/Nall
					rN = Nrec/Nall*Nmult
					raN = Nmult
					obN = Nobs/Nall*Nmult
					fiN = Nall
					fioN = Nobs
					firN = Nrec

					NrecPrsa = NrecPrsa/Nall*Nmult
					NobsPrsa = NobsPrsa/Nall*Nmult
					NallPrsa = NallPrsa/Nall*Nmult		

					NrecDWD = NrecDWD/Nall*Nmult
					NobsDWD = NobsDWD/Nall*Nmult
					NallDWD = NallDWD/Nall*Nmult	



				self.outputNumbers['RA'].append(header['OpSimRA'])
				self.outputNumbers['Dec'].append(header['OpSimDec'])
				self.outputNumbers['recFrac'].append(rF)

				self.outputNumbers['recN'].append(rN)
				self.outputNumbers['rawN'].append(raN)
				self.outputNumbers['obsN'].append(obN)

				self.outputNumbers['fileN'].append(fiN)
				self.outputNumbers['fileObsN'].append(fioN)
				self.outputNumbers['fileRecN'].append(firN)

				self.outputNumbers['allNPrsa'].append(NallPrsa)
				self.outputNumbers['obsNPrsa'].append(NobsPrsa)
				self.outputNumbers['recNPrsa'].append(NrecPrsa)

				self.outputNumbers['allNDWD'].append(NallDWD)
				self.outputNumbers['obsNDWD'].append(NobsDWD)
				self.outputNumbers['recNDWD'].append(NrecDWD)



		#bins
		self.outputHists['m1b'] = m1b
		self.outputHists['qb'] = qb
		self.outputHists['eb'] = eb
		self.outputHists['lpb'] = lpb
		self.outputHists['db'] = db
		self.outputHists['magb'] = magb
		self.outputHists['rb'] = rb
		#All (inserting zeros at the start so that I can more easily plot these with the bin_edges)
		self.outputHists['m1hAll'] = np.insert(m1hAll,0,0)
		self.outputHists['qhAll'] = np.insert(qhAll,0,0)
		self.outputHists['ehAll'] = np.insert(ehAll,0,0)
		self.outputHists['lphAll'] = np.insert(lphAll,0,0)
		self.outputHists['dhAll'] = np.insert(dhAll,0,0)
		self.outputHists['maghAll'] = np.insert(maghAll,0,0)
		self.outputHists['rhAll'] = np.insert(rhAll,0,0)
		#Observable
		self.outputHists['m1hObs'] = np.insert(m1hObs,0,0)
		self.outputHists['qhObs'] = np.insert(qhObs,0,0)
		self.outputHists['ehObs'] = np.insert(ehObs,0,0)
		self.outputHists['lphObs'] = np.insert(lphObs,0,0)
		self.outputHists['dhObs'] = np.insert(dhObs,0,0)
		self.outputHists['maghObs'] = np.insert(maghObs,0,0)
		self.outputHists['rhObs'] = np.insert(rhObs,0,0)
		#Recovered
		self.outputHists['m1hRec'] = dict()
		self.outputHists['qhRec'] = dict()
		self.outputHists['ehRec'] = dict()
		self.outputHists['lphRec'] = dict()
		self.outputHists['dhRec'] = dict()
		self.outputHists['maghRec'] = dict()
		self.outputHists['rhRec'] = dict()
		for f in self.filters:
			self.outputHists['m1hRec'][f] = np.insert(m1hRec[f],0,0)
			self.outputHists['qhRec'][f] = np.insert(qhRec[f],0,0)
			self.outputHists['ehRec'][f] = np.insert(ehRec[f],0,0)
			self.outputHists['lphRec'][f] = np.insert(lphRec[f],0,0)
			self.outputHists['dhRec'][f] = np.insert(dhRec[f],0,0)
			self.outputHists['maghRec'][f] = np.insert(maghRec[f],0,0)
			self.outputHists['rhRec'][f] = np.insert(rhRec[f],0,0)

		if (self.doIndividualPlots):
			self.individualPlots['fmass'] = fmass
			self.individualPlots['fqrat'] = fqrat
			self.individualPlots['fecc'] = fecc
			self.individualPlots['flper'] = flper
			self.individualPlots['fdist'] = fdist
			self.individualPlots['fmag'] = fmag
			self.individualPlots['frad'] = frad

	def makePlots(self):

		if not os.path.exists(self.plotsDirectory):
			os.makedirs(self.plotsDirectory)

		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'

		#plot and save the histograms
		self.saveHist(self.outputHists, 'm1', 'm1 (Msolar)', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=[0,3])
		self.saveHist(self.outputHists, 'q', 'q (m2/m1)', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1])
		self.saveHist(self.outputHists, 'e', 'e', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1])
		self.saveHist(self.outputHists, 'lp', 'log(P [days])', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5])
		self.saveHist(self.outputHists, 'd', 'd (kpc)', os.path.join(self.plotsDirectory,'EBLSST_dhist'+suffix), xlim=[0,25])
		self.saveHist(self.outputHists, 'mag', 'mag', os.path.join(self.plotsDirectory,'EBLSST_maghist'+suffix), xlim=[12, 25])
		self.saveHist(self.outputHists, 'r', 'r2/r1', os.path.join(self.plotsDirectory,'EBLSST_rhist'+suffix), xlim=[0,3])


		#make the mollweide
		coords = SkyCoord(self.outputNumbers['RA'], self.outputNumbers['Dec'], unit=(units.degree, units.degree),frame='icrs')	
		lGal = coords.galactic.l.wrap_at(180.*units.degree).degree
		bGal = coords.galactic.b.wrap_at(180.*units.degree).degree
		RAwrap = coords.ra.wrap_at(180.*units.degree).degree
		Decwrap = coords.dec.wrap_at(180.*units.degree).degree

		f, ax = plt.subplots(subplot_kw={'projection': "mollweide"}, figsize=(8,5))
		ax.grid(True)
		#ax.set_xlabel(r"$l$",fontsize=16)
		#ax.set_ylabel(r"$b$",fontsize=16)
		#mlw = ax.scatter(lGal.ravel()*np.pi/180., bGal.ravel()*np.pi/180., c=np.log10(np.array(recFrac)*100.), cmap='viridis_r', s = 4)
		ax.set_xlabel("RA",fontsize=16)
		ax.set_ylabel("Dec",fontsize=16)
		mlw = ax.scatter(np.array(RAwrap).ravel()*np.pi/180., np.array(Decwrap).ravel()*np.pi/180., c=np.array(self.outputNumbers['recFrac'])*100., cmap='viridis_r', s = 4, vmin=0, vmax=0.07)
		cbar = f.colorbar(mlw, shrink=0.7)
		cbar.set_label(r'% recovered')
		f.savefig(os.path.join(self.plotsDirectory,'mollweide_pct'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)

		f, ax = plt.subplots(subplot_kw={'projection': "mollweide"}, figsize=(8,5))
		ax.grid(True)
		#ax.set_xlabel(r"$l$",fontsize=16)
		#ax.set_ylabel(r"$b$",fontsize=16)
		#mlw = ax.scatter(lGal.ravel()*np.pi/180., bGal.ravel()*np.pi/180., c=np.log10(np.array(recN)), cmap='viridis_r', s = 4)
		ax.set_xlabel("RA",fontsize=16)
		ax.set_ylabel("Dec",fontsize=16)
		mlw = ax.scatter(np.array(RAwrap).ravel()*np.pi/180., np.array(Decwrap).ravel()*np.pi/180., c=np.log10(np.array(self.outputNumbers['recN'])), cmap='viridis_r', s = 4, vmin=0, vmax=3.7)
		cbar = f.colorbar(mlw, shrink=0.7)
		cbar.set_label(r'log10(N) recovered')
		f.savefig(os.path.join(self.plotsDirectory,'mollweide_N'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)

		if (self.doIndividualPlots):
			self.individualPlots['fmass'].savefig(os.path.join(self.plotsDirectory,'massPDFall'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
			self.individualPlots['fqrat'].savefig(os.path.join(self.plotsDirectory,'qPDFall'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
			self.individualPlots['fecc'].savefig(os.path.join(self.plotsDirectory,'eccPDFall'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
			self.individualPlots['flper'].savefig(os.path.join(self.plotsDirectory,'lperPDFall'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
			self.individualPlots['fdist'].savefig(os.path.join(self.plotsDirectory,'distPDFall'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
			self.individualPlots['fmag'].savefig(os.path.join(self.plotsDirectory,'magPDFall'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
			self.individualPlots['frad'].savefig(os.path.join(self.plotsDirectory,'radPDFall'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
			plt.close(self.individualPlots['fmass'])
			plt.close(self.individualPlots['fqrat'])
			plt.close(self.individualPlots['fecc'])
			plt.close(self.individualPlots['flper'])
			plt.close(self.individualPlots['fdist'])
			plt.close(self.individualPlots['fmag'])
			plt.close(self.individualPlots['frad'])

		print("###################")
		print("number of binaries in input files (raw, log):",np.sum(self.outputNumbers['fileN']), np.log10(np.sum(self.outputNumbers['fileN'])))
		print("number of binaries in tested with gatspy (raw, log):",np.sum(self.outputNumbers['fileObsN']), np.log10(np.sum(self.outputNumbers['fileObsN'])))
		print("number of binaries in recovered with gatspy (raw, log):",np.sum(self.outputNumbers['fileRecN']), np.log10(np.sum(self.outputNumbers['fileRecN'])))
		print("recovered/observable*100 with gatspy:",np.sum(self.outputNumbers['fileRecN'])/np.sum(self.outputNumbers['fileObsN'])*100.)
		print("###################")
		print("total in sample (raw, log):",np.sum(self.outputNumbers['rawN']), np.log10(np.sum(self.outputNumbers['rawN'])))
		print("total observable (raw, log):",np.sum(self.outputNumbers['obsN']), np.log10(np.sum(self.outputNumbers['obsN'])))
		print("total recovered (raw, log):",np.sum(self.outputNumbers['recN']), np.log10(np.sum(self.outputNumbers['recN'])))
		print("recovered/observable*100:",np.sum(self.outputNumbers['recN'])/np.sum(self.outputNumbers['obsN'])*100.)
		print("###################")
		print("total in Prsa 15.8<r<19.5 P<1000d sample (raw, log):",np.sum(self.outputNumbers['allNPrsa']), np.log10(np.sum(self.outputNumbers['allNPrsa'])))
		print("total observable in Prsa 15.8<r<19.5 P<1000d sample (raw, log):",np.sum(self.outputNumbers['obsNPrsa']), np.log10(np.sum(self.outputNumbers['obsNPrsa'])))
		print("total recovered in Prsa 15.8<r<19.5 P<1000d sample (raw, log):",np.sum(self.outputNumbers['recNPrsa']), np.log10(np.sum(self.outputNumbers['recNPrsa'])))
		print("Prsa 15.8<r<19.5 P<1000d rec/obs*100:",np.sum(self.outputNumbers['recNPrsa'])/np.sum(self.outputNumbers['obsNPrsa'])*100.)
		print("###################")
		print("total in DWD sample (raw, log):",np.sum(self.outputNumbers['allNDWD']), np.log10(np.sum(self.outputNumbers['allNDWD'])))
		print("total observable in DWD sample (raw, log):",np.sum(self.outputNumbers['obsNDWD']), np.log10(np.sum(self.outputNumbers['obsNDWD'])))
		print("total recovered in DWD sample (raw, log):",np.sum(self.outputNumbers['recNDWD']), np.log10(np.sum(self.outputNumbers['recNDWD'])))
		print("DWD rec/obs*100:",np.sum(self.outputNumbers['recNDWD'])/np.sum(self.outputNumbers['obsNDWD'])*100.)

		#save the numbers to a file
		df = pd.DataFrame(self.outputNumbers)
		df.to_csv(os.path.join(self.plotsDirectory,'numbers.csv'), index=False)

	def plotAllObsRecOtherRatio(self, d1, d2):
		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'
		self.plotObsRecOtherRatio(d1, d2, 'm1', 'm1 (Msolar)', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=[0,3])
		self.plotObsRecOtherRatio(d1, d2, 'q', 'q (m2/m1)', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1])
		self.plotObsRecOtherRatio(d1, d2, 'e', 'e', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1])
		self.plotObsRecOtherRatio(d1, d2, 'lp', 'log(P [days])', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5])
		self.plotObsRecOtherRatio(d1, d2, 'd', 'd (kpc)', os.path.join(self.plotsDirectory,'EBLSST_dhist'+suffix), xlim=[0,25])
		self.plotObsRecOtherRatio(d1, d2, 'mag', 'mag', os.path.join(self.plotsDirectory,'EBLSST_maghist'+suffix), xlim=[12, 25])
		self.plotObsRecOtherRatio(d1, d2, 'r', 'r2/r1', os.path.join(self.plotsDirectory,'EBLSST_rhist'+suffix), xlim=[0,3])

	def run(self):
		self.compileData()
		self.makePlots()

if __name__ == "__main__":

	x = EBLSSTanalyzer(directory = '/Users/ageller/WORK/LSST/WDtest/test/output_files', 
				plotsDirectory ='/Users/ageller/WORK/LSST/WDtest/test/plots',
				doIndividualPlots = True,
				cluster = True)
	x.run()

	y = EBLSSTanalyzer(directory = '/Users/ageller/WORK/LSST/WDtest/test/withCrowding/output_files', 
			plotsDirectory ='/Users/ageller/WORK/LSST/WDtest/test/withCrowding/plots',
			doIndividualPlots = True,
			cluster = True)
	y.run()

	x.plotAllObsRecOtherRatio(x.outputHists, y.outputHists)



