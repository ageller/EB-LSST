# should add Andrew's corner plots on here!

import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy import units, constants
from astropy.modeling import models, fitting
import scipy.stats
from scipy.integrate import quad
import pickle

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

		self.m1xlim = [0.,3.]
		self.m1bin = 0.1

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



	def plotObsRecRatio(self, d, key, xtitle, fname, xlim = None, ax = [None], showLegend = True,):
		c1 = '#0294A5'  #turqoise
		c2 = '#d95f02' #orange from color brewer
		c3 = '#00353E' #slate
		c4 = '#508201' #olive

		saveit = False
		if (ax[0] is None):
			saveit = True
			f,ax = plt.subplots(3,1,figsize=(5, 12), sharex=True)

		histAll = d[key+'hAllCDF']
		histObs = d[key+'hObsCDF'] 
		allhistRec = d[key+'hRecCDF']['all']
		bin_edges = d[key+'bCDF']

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
		ax[0].step(bin_edges, cdfAll, color=c1, label='All')
		ax[0].step(bin_edges, cdfObs, color=c2, label='Observable')
		ax[0].step(bin_edges, cdfRec, color=c3, label='Recoverable')
		if (saveit):
			ax[0].set_ylabel('CDF')


		histAll = d[key+'hAll']
		histObs = d[key+'hObs'] 
		allhistRec = d[key+'hRec']['all']
		bin_edges = d[key+'b']

		binHalf = (bin_edges[1] - bin_edges[0])/2.

		#PDF
		ax[1].step(bin_edges, histAll/np.sum(histAll), color=c1, label='All')
		ax[1].step(bin_edges, histObs/np.sum(histObs), color=c2, label='Observable')
		ax[1].step(bin_edges, allhistRec/np.sum(allhistRec), color=c3, label='Recoverable')
		ax[1].set_yscale('log')
		if (saveit):
			ax[1].set_ylabel('PDF')

		#ratio
		#prepend some values at y of zero so that the step plots look correct
		use = np.where(histAll > 0)[0]
		if (len(use) > 0):
			b = bin_edges[use]
			r = histObs[use]/histAll[use]
			ax[2].step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c2, label='Observable/All')
			ax[2].plot(b - binHalf, r, 'o',color=c1, markersize=5, markeredgecolor=c2)

			r = allhistRec[use]/histAll[use]
			ax[2].step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c3, label='Recoverable/All')
			ax[2].plot(b - binHalf, r, 'o',color=c1, markersize=5, markeredgecolor=c3)

			use = np.where(histObs > 0)[0]
			if (len(use) > 0):
				b = bin_edges[use]
				r =  allhistRec[use]/histObs[use]
				ax[2].step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c3, label='Recoverable/Observable')
				ax[2].plot(b - binHalf, r, 'o',color=c2, markersize=5, markeredgecolor=c3)
		#ax3.step(bin_edges[use], histRec[use]/histObs[use], color=c2, linestyle='--', dashes=(3, 3), linewidth=4)


		ax[2].set_yscale('log')
		ax[2].set_ylim(10**-4,1)
		ax[2].set_xlabel(xtitle)
		if (saveit):
			ax[2].set_ylabel('Ratio')

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		if (showLegend):
			lAll = mlines.Line2D([], [], color=c1, label='All')
			lObs = mlines.Line2D([], [], color=c2, label='Obs.')
			lRec = mlines.Line2D([], [], color=c3, label='Rec.')
			lObsAll = mlines.Line2D([], [], color=c2, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c2, label='Obs./All')
			lRecAll = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c3, label='Rec./All')
			lRecObs = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c2, markersize=5, markeredgecolor=c3, label='Rec./Obs.')
			ax[0].legend(handles=[lAll, lObs, lRec, lObsAll, lRecAll, lRecObs], loc='lower right')

		if (xlim is not None):
			ax[0].set_xlim(xlim[0], xlim[1])
			ax[1].set_xlim(xlim[0], xlim[1])
			ax[2].set_xlim(xlim[0], xlim[1])

		if (saveit):
			f.subplots_adjust(hspace=0)
			print(fname)
			f.savefig(fname+'_ObsRecRatio.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)


	def plotObsRecOtherRatio(self, d1, d2, key, xtitle, fname,  xlim = None, ax = [None], showLegend = True, legendLoc = 'lower right'):
		c1 = '#0294A5'  #turqoise
		c2 = '#d95f02' #orange from color brewer
		c3 = '#00353E' #slate
		c4 = '#508201' #olive

		saveit = False
		if (ax[0] is None):
			saveit = True
			f,ax = plt.subplots(3,1,figsize=(5, 12), sharex=True)

		histAll = d1[key+'hAllCDF']
		histObs = d1[key+'hObsCDF']
		allhistRec = d1[key+'hRecCDF']['all']
		bin_edges = d1[key+'bCDF']

		histAllOD = d2[key+'hAllCDF']
		histObsOD = d2[key+'hObsCDF']
		allhistRecOD = d2[key+'hRecCDF']['all']
		bin_edgesOD = d2[key+'bCDF']		

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
		ax[0].step(bin_edges, cdfAll, color=c1, label='All')
		ax[0].step(bin_edges, cdfObs, color=c2, label='Observable')
		ax[0].step(bin_edges, cdfRec, color=c3, label='Recoverable')
		ax[0].step(bin_edgesOD, cdfAllOD, color=c1, linestyle=':')
		ax[0].step(bin_edgesOD, cdfObsOD, color=c2, linestyle=':')
		ax[0].step(bin_edgesOD, cdfRecOD, color=c3, linestyle=':')
		ax[0].set_ylim(-0.01,1.01)
		if (saveit):
			ax[0].set_ylabel('CDF', fontsize=16)

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

		#PDF --need to divide by the bin size
		# ax2.step(bin_edges, histAll/np.sum(histAll)/np.diff(bin_edges)[0], color=c1, label='All')
		# ax2.step(bin_edges, histObs/np.sum(histObs)/np.diff(bin_edges)[0], color=c2, label='Observable')
		# ax2.step(bin_edges, allhistRec/np.sum(allhistRec)/np.diff(bin_edges)[0], color=c3, label='Recoverable')
		# ax2.step(bin_edgesOD, histAllOD/np.sum(histAllOD)/np.diff(bin_edgesOD)[0], color=c1, linestyle=':')
		# ax2.step(bin_edgesOD, histObsOD/np.sum(histObsOD)/np.diff(bin_edgesOD)[0], color=c2, linestyle=':')
		# ax2.step(bin_edgesOD, allhistRecOD/np.sum(allhistRecOD)/np.diff(bin_edgesOD)[0], color=c3, linestyle=':')
		# ax2.set_ylabel('PDF', fontsize=16)
		#this is the fraction in each bin
		ax[1].step(bin_edges, histAll/np.sum(histAll), color=c1, label='All')
		ax[1].step(bin_edges, histObs/np.sum(histObs), color=c2, label='Observable')
		ax[1].step(bin_edges, allhistRec/np.sum(allhistRec), color=c3, label='Recoverable')
		ax[1].step(bin_edgesOD, histAllOD/np.sum(histAll), color=c1, linestyle=':')
		ax[1].step(bin_edgesOD, histObsOD/np.sum(histObs), color=c2, linestyle=':')
		ax[1].step(bin_edgesOD, allhistRecOD/np.sum(allhistRec), color=c3, linestyle=':')
		ax[1].set_ylim(0.5e-5, 1.9)
		ax[1].set_yscale('log')
		if (saveit):
			ax[1].set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=16)

		ratio = histObs/histAll
		check = np.isnan(ratio)
		ratio[check]=0.
		ax[2].step(bin_edges, ratio, color=c2, label='Observable/All')
		ax[2].plot(bin_edges - binHalf, ratio, 'o',color=c1, markersize=5, markeredgecolor=c2)

		ratio = allhistRec/histAll
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax[2].step(bin_edges, ratio, color=c3, label='Recoverable/All')
		ax[2].plot(bin_edges - binHalf, ratio, 'o',color=c1, markersize=5, markeredgecolor=c3)

		ratio = allhistRec/histObs
		check = np.isnan(ratio)
		ratio[check]=0.
		ax[2].step(bin_edges, ratio, color=c3, label='Recoverable/Observable')
		ax[2].plot(bin_edges - binHalf, ratio, 'o',color=c2, markersize=5, markeredgecolor=c3)

		ratio = histObsOD/histAllOD
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax[2].step(bin_edgesOD, ratio, color=c2, linestyle=':')
		ax[2].plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c1, markersize=3.5, markeredgecolor=c2)

		ratio = allhistRecOD/histAllOD
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax[2].step(bin_edgesOD, ratio, color=c3, linestyle=':')
		ax[2].plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c1, markersize=3.5, markeredgecolor=c3)

		ratio = allhistRecOD/histObsOD
		check = np.isnan(ratio)
		ratio[check]=0.
		ax[2].step(bin_edgesOD, ratio, color=c3, linestyle=':')
		ax[2].plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c2, markersize=3.5, markeredgecolor=c3)

		if (saveit):
			ax[2].set_ylabel('Ratio', fontsize=16)
		ax[2].set_ylim(0.5e-5,1)
		ax[2].set_yscale('log')
		ax[2].set_xlabel(xtitle, fontsize=16)

		if (xlim is not None):
			ax[0].set_xlim(xlim[0], xlim[1])
			ax[1].set_xlim(xlim[0], xlim[1])
			ax[2].set_xlim(xlim[0], xlim[1])

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		if (showLegend):
			lAll = mlines.Line2D([], [], color=c1, label='All')
			lObs = mlines.Line2D([], [], color=c2, label='Obs.')
			lRec = mlines.Line2D([], [], color=c3, label='Rec.')
			lObsAll = mlines.Line2D([], [], color=c2, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c2, label='Obs./All')
			lRecAll = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c1, markersize=5, markeredgecolor=c3, label='Rec./All')
			lRecObs = mlines.Line2D([], [], color=c3, marker='o', markerfacecolor=c2, markersize=5, markeredgecolor=c3, label='Rec./Obs.')
			ax[0].legend(handles=[lAll, lObs, lRec, lObsAll, lRecAll, lRecObs], loc=legendLoc)


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOtherRatio.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)

	def plotObsRecOtherPDF(self, d1, d1C, d2, d2C, d3, d3C, key, xtitle, fname,  xlim = None, ax = None, showLegend = True, legendLoc = 'lower right'):
		c1 = '#0294A5'  #turqoise
		c2 = '#d95f02' #orange from color brewer
		c3 = '#00353E' #slate
		c4 = '#508201' #olive

		saveit = False
		if (ax is None):
			saveit = True
			f,ax = plt.subplots(figsize=(5, 4))


		histAll1 = d1[key+'hAll']
		histObs1 = d1[key+'hObs']
		allhistRec1 = d1[key+'hRec']['all']
		bin_edges1 = d1[key+'b']

		histAll1C = d1C[key+'hAll']
		histObs1C = d1C[key+'hObs']
		allhistRec1C = d1C[key+'hRec']['all']
		bin_edges1C = d1C[key+'b']

		histAll2 = d2[key+'hAll']
		histObs2 = d2[key+'hObs']
		allhistRec2 = d2[key+'hRec']['all']
		bin_edges2 = d2[key+'b']		

		histAll2C = d2C[key+'hAll']
		histObs2C = d2C[key+'hObs']
		allhistRec2C = d2C[key+'hRec']['all']
		bin_edges2C = d2C[key+'b']	

		histAll3 = d3[key+'hAll']
		histObs3 = d3[key+'hObs']
		allhistRec3 = d3[key+'hRec']['all']
		bin_edges3 = d3[key+'b']		

		histAll3C = d3C[key+'hAll']
		histObs3C = d3C[key+'hObs']
		allhistRec3C = d3C[key+'hRec']['all']
		bin_edges3C = d3C[key+'b']	

		binHalf1 = (bin_edges1[1] - bin_edges1[0])/2.
		binHalf1C = (bin_edges1C[1] - bin_edges1C[0])/2.
		binHalf2 = (bin_edges2[1] - bin_edges2[0])/2.
		binHalf2C = (bin_edges2C[1] - bin_edges2C[0])/2.
		binHalf3 = (bin_edges3[1] - bin_edges3[0])/2.
		binHalf3C = (bin_edges3C[1] - bin_edges3C[0])/2.

		#PDF --need to divide by the bin size
		#this is the fraction in each bin
		#ax.step(bin_edges1,  allhistRec1/np.sum(allhistRec1), color=c3, label='Field')
		#ax.step(bin_edges1C, allhistRec1C/np.sum(allhistRec1), color=c3,linestyle=':')
		#ax.step(bin_edges2,  allhistRec2/np.sum(allhistRec2), color=c2, label='GCs')
		#ax.step(bin_edges2C, allhistRec2C/np.sum(allhistRec2), color=c2,linestyle=':')
		#ax.step(bin_edges3,  allhistRec3/np.sum(allhistRec3), color=c1, label='OCs')
		#ax.step(bin_edges3C, allhistRec3C/np.sum(allhistRec3), color=c1,linestyle=':')

		ax.step(bin_edges1,  allhistRec1, color=c3, label='Field')
		ax.step(bin_edges1C, allhistRec1C, color=c3,linestyle=':')
		ax.step(bin_edges2,  allhistRec2, color=c2, label='GCs')
		ax.step(bin_edges2C, allhistRec2C, color=c2,linestyle=':')
		ax.step(bin_edges3,  allhistRec3, color=c1, label='OCs')
		ax.step(bin_edges3C, allhistRec3C, color=c1,linestyle=':')


		#ax.set_ylim(0.5e-5, 1.9)
		ax.set_ylim(1, 4e6)
		ax.set_yscale('log')
		ax.set_xlabel(xtitle, fontsize=16)
		if (saveit):
			#ax.set_ylabel(r'$N_i/\sum_i N_i$', fontsize=16)
			ax.set_ylabel(r'$N$', fontsize=16)


		if (xlim is not None):
			ax.set_xlim(xlim[0], xlim[1])

		if (showLegend):
			lAll = mlines.Line2D([], [], color=c3, label='Field')
			lObs = mlines.Line2D([], [], color=c2, label='GCs')
			lRec = mlines.Line2D([], [], color=c1, label='OCs')
			ax.legend(handles=[lAll, lObs, lRec], loc=legendLoc)


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOtherPDF.pdf',format='pdf', bbox_inches = 'tight')
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
		histAll = d[key+'hAllCDF']
		histObs = d[key+'hObsCDF']
		histRec = d[key+'hRecCDF']
		bin_edges = d[key+'bCDF']

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
		# with open(fname+'.csv','w') as fl:
		# 	outline = 'binEdges,histAll,histObs'
		# 	for f in filters:
		# 		outline += ','+f+'histRec'
		# 	outline += '\n'
		# 	fl.write(outline)
		# 	for i in range(len(bin_edges)):
		# 		outline = str(bin_edges[i])+','+str(histAll[i])+','+str(histObs[i])
		# 		for f in filters:
		# 			outline += ','+str(histRec[f][i])
		# 		outline += '\n'
		# 		fl.write(outline)

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
		mbSize = self.m1bin
		qbSize = 0.05
		ebSize = 0.05
		lpbSize = 0.25
		dbSize = 1.
		magbSize = 1.
		rbSize = 0.2
		mbins = np.arange(0,10+mbSize, mbSize, dtype='float')
		qbins = np.arange(0,1+qbSize, qbSize, dtype='float')
		ebins = np.arange(0, 1+ebSize, ebSize, dtype='float')
		lpbins = np.arange(-3, 10+lpbSize, lpbSize, dtype='float')
		dbins = np.arange(0, 40+dbSize, dbSize, dtype='float')
		magbins = np.arange(11, 25+magbSize, magbSize, dtype='float')
		rbins = np.arange(0, 5+rbSize, rbSize, dtype='float')

		CDFfac = 1000.
		mbinsCDF = np.arange(0,10+mbSize/CDFfac, mbSize/CDFfac, dtype='float')
		qbinsCDF = np.arange(0,1+qbSize/CDFfac, qbSize/CDFfac, dtype='float')
		ebinsCDF = np.arange(0, 1+ebSize/CDFfac, ebSize/CDFfac, dtype='float')
		lpbinsCDF = np.arange(-3, 10+lpbSize/CDFfac, lpbSize/CDFfac, dtype='float')
		dbinsCDF = np.arange(0, 40+dbSize/CDFfac, dbSize/CDFfac, dtype='float')
		magbinsCDF = np.arange(11, 25+magbSize/CDFfac, magbSize/CDFfac, dtype='float')
		rbinsCDF = np.arange(0, 5+rbSize/CDFfac, rbSize/CDFfac, dtype='float')

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

		#blanks for the CDFs
		#All
		m1hAllCDF = np.zeros_like(mbinsCDF)[1:]
		qhAllCDF = np.zeros_like(qbinsCDF)[1:]
		ehAllCDF = np.zeros_like(ebinsCDF)[1:]
		lphAllCDF = np.zeros_like(lpbinsCDF)[1:]
		dhAllCDF = np.zeros_like(dbinsCDF)[1:]
		maghAllCDF = np.zeros_like(magbinsCDF)[1:]
		rhAllCDF = np.zeros_like(rbinsCDF)[1:]
		#Observable
		m1hObsCDF = np.zeros_like(mbinsCDF)[1:]
		qhObsCDF = np.zeros_like(qbinsCDF)[1:]
		ehObsCDF = np.zeros_like(ebinsCDF)[1:]
		lphObsCDF = np.zeros_like(lpbinsCDF)[1:]
		dhObsCDF = np.zeros_like(dbinsCDF)[1:]
		maghObsCDF = np.zeros_like(magbinsCDF)[1:]
		rhObsCDF = np.zeros_like(rbinsCDF)[1:]
		#Recovered
		m1hRecCDF = dict()
		qhRecCDF = dict()
		ehRecCDF = dict()
		lphRecCDF = dict()
		dhRecCDF = dict()
		maghRecCDF = dict()
		rhRecCDF = dict()
		for f in self.filters:
			m1hRecCDF[f] = np.zeros_like(mbinsCDF)[1:]
			qhRecCDF[f] = np.zeros_like(qbinsCDF)[1:]
			ehRecCDF[f] = np.zeros_like(ebinsCDF)[1:]
			lphRecCDF[f] = np.zeros_like(lpbinsCDF)[1:]
			dhRecCDF[f] = np.zeros_like(dbinsCDF)[1:]
			maghRecCDF[f] = np.zeros_like(magbinsCDF)[1:]
			rhRecCDF[f] = np.zeros_like(rbinsCDF)[1:]

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

					#swap locations so that m1 is always > m2
					check = data.loc[(data['m2'] > data['m1'])]
					if (len(check.index) > 0):
						for index, row in check.iterrows():
							m1tmp = row['m1']
							data.at[index, 'm1'] = row['m2']
							data.at[index, 'm2'] = m1tmp
							r1tmp = row['r1']
							data.at[index, 'r1'] = row['r2']
							data.at[index, 'r2'] = r1tmp						
							#will want to swap L1 and T1 too if using newer files

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
						data = data.loc[(data['logg1'] > 6) & (data['logg1'] < 12) & (data['logg2'] > 6) & (data['logg2'] < 12)]

					prsa = data.loc[(data['appMagMean_r'] <= 19.5) & (data['appMagMean_r'] > 15.8) & (data['p'] < 1000) & (data['p'] > 0.5)]
					DWD = data.loc[(data['logg1'] > 6) & (data['logg1'] < 12) & (data['logg2'] > 6) & (data['logg2'] < 12)]

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

						m1hAll0CDF, m1bCDF = np.histogram(data["m1"], bins=mbinsCDF)
						qhAll0CDF, qbCDF = np.histogram(data["m2"]/data["m1"], bins=qbinsCDF)
						ehAll0CDF, ebCDF = np.histogram(data["e"], bins=ebinsCDF)
						lphAll0CDF, lpbCDF = np.histogram(np.ma.log10(data["p"].values).filled(-999), bins=lpbinsCDF)
						dhAll0CDF, dbCDF = np.histogram(data["d"], bins=dbinsCDF)
						maghAll0CDF, magbCDF = np.histogram(data["appMagMean_r"], bins=magbinsCDF)
						rhAll0CDF, rbCDF = np.histogram(data["r2"]/data["r1"], bins=rbinsCDF)

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

						m1hAllCDF += m1hAll0CDF/Nall*Nmult
						qhAllCDF += qhAll0CDF/Nall*Nmult
						ehAllCDF += ehAll0CDF/Nall*Nmult
						lphAllCDF += lphAll0CDF/Nall*Nmult
						dhAllCDF += dhAll0CDF/Nall*Nmult
						maghAllCDF += maghAll0CDF/Nall*Nmult
						rhAllCDF += rhAll0CDF/Nall*Nmult

						#Obs
						#I want to account for all filters here too (maybe not necessary; if LSM is != -999 then they are all filled in, I think)...
						obs = data.loc[(data['u_LSS_PERIOD'] != -999) | (data['g_LSS_PERIOD'] != -999) | (data['r_LSS_PERIOD'] != -999) | (data['i_LSS_PERIOD'] != -999) | (data['z_LSS_PERIOD'] != -999) | (data['y_LSS_PERIOD'] != -999) | (data['LSM_PERIOD'] != -999)]
						prsaObs = data.loc[(data['appMagMean_r'] <= 19.5) & (data['appMagMean_r'] > 15.8) & (data['p'] < 1000) & (data['p'] >0.5) & ((data['u_LSS_PERIOD'] != -999) | (data['g_LSS_PERIOD'] != -999) | (data['r_LSS_PERIOD'] != -999) | (data['i_LSS_PERIOD'] != -999) | (data['z_LSS_PERIOD'] != -999) | (data['y_LSS_PERIOD'] != -999) | (data['LSM_PERIOD'] != -999))]
						DWDObs = data.loc[(data['logg1'] > 6) & (data['logg1'] < 12) & (data['logg2'] > 6) & (data['logg2'] < 12) & ((data['u_LSS_PERIOD'] != -999) | (data['g_LSS_PERIOD'] != -999) | (data['r_LSS_PERIOD'] != -999) | (data['i_LSS_PERIOD'] != -999) | (data['z_LSS_PERIOD'] != -999) | (data['y_LSS_PERIOD'] != -999) | (data['LSM_PERIOD'] != -999))]

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

							m1hObs0CDF, m1bCDF = np.histogram(obs["m1"], bins=mbinsCDF)
							qhObs0CDF, qbCDF = np.histogram(obs["m2"]/obs["m1"], bins=qbinsCDF)
							ehObs0CDF, ebCDF = np.histogram(obs["e"], bins=ebinsCDF)
							lphObs0CDF, lpbCDF = np.histogram(np.ma.log10(obs["p"].values).filled(-999), bins=lpbinsCDF)
							dhObs0CDF, dbCDF = np.histogram(obs["d"], bins=dbinsCDF)
							maghObs0CDF, magbCDF = np.histogram(obs["appMagMean_r"], bins=magbinsCDF)
							rhObs0CDF, rbCDF = np.histogram(obs["r2"]/obs["r1"], bins=rbinsCDF)
							m1hObsCDF += m1hObs0CDF/Nall*Nmult
							qhObsCDF += qhObs0CDF/Nall*Nmult
							ehObsCDF += ehObs0CDF/Nall*Nmult
							lphObsCDF += lphObs0CDF/Nall*Nmult
							dhObsCDF += dhObs0CDF/Nall*Nmult
							maghObsCDF += maghObs0CDF/Nall*Nmult
							rhObsCDF += rhObs0CDF/Nall*Nmult

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
								DWDRec = data.loc[(data['logg1'] > 6) & (data['logg1'] < 12) & (data['logg2'] > 6) & (data['logg2'] < 12) & (data['LSM_PERIOD'] != -999) & ( (fullP < self.Pcut) | (halfP < self.Pcut) | (twiceP < self.Pcut))]

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

									m1hRec0CDF, m1bCDF = np.histogram(rec["m1"], bins=mbinsCDF)
									qhRec0CDF, qbCDF = np.histogram(rec["m2"]/rec["m1"], bins=qbinsCDF)
									ehRec0CDF, ebCDF = np.histogram(rec["e"], bins=ebinsCDF)
									lphRec0CDF, lpbCDF = np.histogram(np.ma.log10(rec["p"].values).filled(-999), bins=lpbinsCDF)
									dhRec0CDF, dbCDF = np.histogram(rec["d"], bins=dbinsCDF)
									maghRec0CDF, magbCDF = np.histogram(rec["appMagMean_r"], bins=magbinsCDF)
									rhRec0CDF, rbCDF = np.histogram(rec["r2"]/rec["r1"], bins=rbinsCDF)
									m1hRecCDF[filt] += m1hRec0CDF/Nall*Nmult
									qhRecCDF[filt] += qhRec0CDF/Nall*Nmult
									ehRecCDF[filt] += ehRec0CDF/Nall*Nmult
									lphRecCDF[filt] += lphRec0CDF/Nall*Nmult
									dhRecCDF[filt] += dhRec0CDF/Nall*Nmult
									maghRecCDF[filt] += maghRec0CDF/Nall*Nmult
									rhRecCDF[filt] += rhRec0CDF/Nall*Nmult

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

									m1hRec0CDF, m1bCDF = np.histogram(recCombined["m1"], bins=mbinsCDF)
									qhRec0CDF, qbCDF = np.histogram(recCombined["m2"]/rec["m1"], bins=qbinsCDF)
									ehRec0CDF, ebCDF = np.histogram(recCombined["e"], bins=ebinsCDF)
									lphRec0CDF, lpbCDF = np.histogram(np.ma.log10(recCombined["p"].values).filled(-999), bins=lpbinsCDF)
									dhRec0CDF, dbCDF = np.histogram(recCombined["d"], bins=dbinsCDF)
									maghRec0CDF, magbCDF = np.histogram(recCombined["appMagMean_r"], bins=magbinsCDF)
									rhRec0CDF, rbCDF = np.histogram(recCombined["r2"]/recCombined["r1"], bins=rbinsCDF)
									m1hRecCDF[filt] += m1hRec0CDF/Nall*Nmult
									qhRecCDF[filt] += qhRec0CDF/Nall*Nmult
									ehRecCDF[filt] += ehRec0CDF/Nall*Nmult
									lphRecCDF[filt] += lphRec0CDF/Nall*Nmult
									dhRecCDF[filt] += dhRec0CDF/Nall*Nmult
									maghRecCDF[filt] += maghRec0CDF/Nall*Nmult
									rhRecCDF[filt] += rhRec0CDF/Nall*Nmult
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



				self.outputNumbers['RA'].append(header['OpSimRA'][0])
				self.outputNumbers['Dec'].append(header['OpSimDec'][0])
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



		#bins (inserting zeros at the start and end so that I can more easily plot these with the bin_edges)
		self.outputHists['m1b'] = np.append(m1b, m1b[-1] + mbSize)
		self.outputHists['qb'] = np.append(qb, qb[-1] + qbSize)
		self.outputHists['eb'] = np.append(eb, eb[-1] + ebSize)
		self.outputHists['lpb'] = np.append(lpb, lpb[-1] + lpbSize)
		self.outputHists['db'] = np.append(db, db[-1] + dbSize)
		self.outputHists['magb'] = np.append(magb, magb[-1] + magbSize)
		self.outputHists['rb'] = np.append(rb, rb[-1] + rbSize)
		#All 
		self.outputHists['m1hAll'] = np.append(np.insert(m1hAll,0,0),0)
		self.outputHists['qhAll'] = np.append(np.insert(qhAll,0,0),0)
		self.outputHists['ehAll'] = np.append(np.insert(ehAll,0,0),0)
		self.outputHists['lphAll'] = np.append(np.insert(lphAll,0,0),0)
		self.outputHists['dhAll'] = np.append(np.insert(dhAll,0,0),0)
		self.outputHists['maghAll'] = np.append(np.insert(maghAll,0,0),0)
		self.outputHists['rhAll'] = np.append(np.insert(rhAll,0,0),0)
		#Observable
		self.outputHists['m1hObs'] = np.append(np.insert(m1hObs,0,0),0)
		self.outputHists['qhObs'] = np.append(np.insert(qhObs,0,0),0)
		self.outputHists['ehObs'] = np.append(np.insert(ehObs,0,0),0)
		self.outputHists['lphObs'] = np.append(np.insert(lphObs,0,0),0)
		self.outputHists['dhObs'] = np.append(np.insert(dhObs,0,0),0)
		self.outputHists['maghObs'] = np.append(np.insert(maghObs,0,0),0)
		self.outputHists['rhObs'] = np.append(np.insert(rhObs,0,0),0)
		#Recovered
		self.outputHists['m1hRec'] = dict()
		self.outputHists['qhRec'] = dict()
		self.outputHists['ehRec'] = dict()
		self.outputHists['lphRec'] = dict()
		self.outputHists['dhRec'] = dict()
		self.outputHists['maghRec'] = dict()
		self.outputHists['rhRec'] = dict()
		for f in self.filters:
			self.outputHists['m1hRec'][f] = np.append(np.insert(m1hRec[f],0,0),0)
			self.outputHists['qhRec'][f] = np.append(np.insert(qhRec[f],0,0),0)
			self.outputHists['ehRec'][f] = np.append(np.insert(ehRec[f],0,0),0)
			self.outputHists['lphRec'][f] = np.append(np.insert(lphRec[f],0,0),0)
			self.outputHists['dhRec'][f] = np.append(np.insert(dhRec[f],0,0),0)
			self.outputHists['maghRec'][f] = np.append(np.insert(maghRec[f],0,0),0)
			self.outputHists['rhRec'][f] = np.append(np.insert(rhRec[f],0,0),0)

		self.outputHists['m1bCDF'] = m1bCDF
		self.outputHists['qbCDF'] = qbCDF
		self.outputHists['ebCDF'] = ebCDF
		self.outputHists['lpbCDF'] = lpbCDF
		self.outputHists['dbCDF'] = dbCDF
		self.outputHists['magbCDF'] = magbCDF
		self.outputHists['rbCDF'] = rbCDF
		#All (inserting zeros at the start so that I can more easily plot these with the bin_edges)
		self.outputHists['m1hAllCDF'] = np.insert(m1hAllCDF,0,0)
		self.outputHists['qhAllCDF'] = np.insert(qhAllCDF,0,0)
		self.outputHists['ehAllCDF'] = np.insert(ehAllCDF,0,0)
		self.outputHists['lphAllCDF'] = np.insert(lphAllCDF,0,0)
		self.outputHists['dhAllCDF'] = np.insert(dhAllCDF,0,0)
		self.outputHists['maghAllCDF'] = np.insert(maghAllCDF,0,0)
		self.outputHists['rhAllCDF'] = np.insert(rhAllCDF,0,0)
		#Observable
		self.outputHists['m1hObsCDF'] = np.insert(m1hObsCDF,0,0)
		self.outputHists['qhObsCDF'] = np.insert(qhObsCDF,0,0)
		self.outputHists['ehObsCDF'] = np.insert(ehObsCDF,0,0)
		self.outputHists['lphObsCDF'] = np.insert(lphObsCDF,0,0)
		self.outputHists['dhObsCDF'] = np.insert(dhObsCDF,0,0)
		self.outputHists['maghObsCDF'] = np.insert(maghObsCDF,0,0)
		self.outputHists['rhObsCDF'] = np.insert(rhObsCDF,0,0)
		#Recovered
		self.outputHists['m1hRecCDF'] = dict()
		self.outputHists['qhRecCDF'] = dict()
		self.outputHists['ehRecCDF'] = dict()
		self.outputHists['lphRecCDF'] = dict()
		self.outputHists['dhRecCDF'] = dict()
		self.outputHists['maghRecCDF'] = dict()
		self.outputHists['rhRecCDF'] = dict()
		for f in self.filters:
			self.outputHists['m1hRecCDF'][f] = np.insert(m1hRecCDF[f],0,0)
			self.outputHists['qhRecCDF'][f] = np.insert(qhRecCDF[f],0,0)
			self.outputHists['ehRecCDF'][f] = np.insert(ehRecCDF[f],0,0)
			self.outputHists['lphRecCDF'][f] = np.insert(lphRecCDF[f],0,0)
			self.outputHists['dhRecCDF'][f] = np.insert(dhRecCDF[f],0,0)
			self.outputHists['maghRecCDF'][f] = np.insert(maghRecCDF[f],0,0)
			self.outputHists['rhRecCDF'][f] = np.insert(rhRecCDF[f],0,0)

		if (self.doIndividualPlots):
			self.individualPlots['fmass'] = fmass
			self.individualPlots['fqrat'] = fqrat
			self.individualPlots['fecc'] = fecc
			self.individualPlots['flper'] = flper
			self.individualPlots['fdist'] = fdist
			self.individualPlots['fmag'] = fmag
			self.individualPlots['frad'] = frad

	def makeMollweides(self, d, suffix='', showCbar=True):

		#make the mollweide
		use = d.loc[(d['recFrac'] > 0)]
		coords = SkyCoord(use['RA'], use['Dec'], unit=(units.degree, units.degree),frame='icrs')	
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
		mlw = ax.scatter(np.array(RAwrap).ravel()*np.pi/180., np.array(Decwrap).ravel()*np.pi/180., c=np.array(use['recN']/use['obsN']), cmap='magma_r', s = 10, vmin=0.3, vmax=0.7)
		if (showCbar):
			#cbar = f.colorbar(mlw, shrink=0.7)
			# Now adding the colorbar
			cbaxes = f.add_axes([0.1, 0.9, 0.8, 0.03]) 
			cbar = plt.colorbar(mlw, cax = cbaxes, orientation="horizontal") 
			cbar.set_label(r'$N_\mathrm{Rec.}/N_\mathrm{Obs.}$',fontsize=16)
			cbaxes.xaxis.set_ticks_position('top')
			cbaxes.xaxis.set_label_position('top')

		f.savefig(os.path.join(self.plotsDirectory,'mollweide_pct'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)


		coords = SkyCoord(d['RA'], d['Dec'], unit=(units.degree, units.degree),frame='icrs')	
		lGal = coords.galactic.l.wrap_at(180.*units.degree).degree
		bGal = coords.galactic.b.wrap_at(180.*units.degree).degree
		RAwrap = coords.ra.wrap_at(180.*units.degree).degree
		Decwrap = coords.dec.wrap_at(180.*units.degree).degree

		f, ax = plt.subplots(subplot_kw={'projection': "mollweide"}, figsize=(8,5))
		ax.grid(True)
		#ax.set_xlabel(r"$l$",fontsize=16)
		#ax.set_ylabel(r"$b$",fontsize=16)
		#mlw = ax.scatter(lGal.ravel()*np.pi/180., bGal.ravel()*np.pi/180., c=np.log10(np.array(recN)), cmap='viridis_r', s = 4)
		ax.set_xlabel("RA",fontsize=16)
		ax.set_ylabel("Dec",fontsize=16)
		mlw = ax.scatter(np.array(RAwrap).ravel()*np.pi/180., np.array(Decwrap).ravel()*np.pi/180., c=np.log10(np.array(d['recN'])), cmap='magma_r', s = 10, vmin=0, vmax=4.5)
		if (showCbar):
			#cbar = f.colorbar(mlw, shrink=0.7)
			cbaxes = f.add_axes([0.1, 0.9, 0.8, 0.03]) 
			cbar = plt.colorbar(mlw, cax = cbaxes, orientation="horizontal") 
			cbar.set_label(r'$\log_{10}(N_\mathrm{Rec.})$',fontsize=16)
			cbaxes.xaxis.set_ticks_position('top')
			cbaxes.xaxis.set_label_position('top')
		f.savefig(os.path.join(self.plotsDirectory,'mollweide_N'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)

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


		self.makeMollweides(pd.DataFrame(self.outputNumbers), suffix=suffix)

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
		pickle.dump(self.outputHists, open( os.path.join(self.plotsDirectory,"outputHists.pickle"), "wb"))

	def plotAllObsRecOtherRatio(self, d1, d2):
		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'
		self.plotObsRecOtherRatio(d1, d2, 'm1', 'm1 (Msolar)', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=self.m1xlim)
		self.plotObsRecOtherRatio(d1, d2, 'q', 'q (m2/m1)', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1])
		self.plotObsRecOtherRatio(d1, d2, 'e', 'e', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1])
		self.plotObsRecOtherRatio(d1, d2, 'lp', 'log(P [days])', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5])
		self.plotObsRecOtherRatio(d1, d2, 'd', 'd (kpc)', os.path.join(self.plotsDirectory,'EBLSST_dhist'+suffix), xlim=[0,25])
		self.plotObsRecOtherRatio(d1, d2, 'mag', 'mag', os.path.join(self.plotsDirectory,'EBLSST_maghist'+suffix), xlim=[12, 25])
		self.plotObsRecOtherRatio(d1, d2, 'r', 'r2/r1', os.path.join(self.plotsDirectory,'EBLSST_rhist'+suffix), xlim=[0,3])


		m1xlim = self.m1xlim
		#m1xlim[1] -= 0.01
		f,ax = plt.subplots(3,4,figsize=(20, 12))
		self.plotObsRecOtherRatio(d1, d2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5], ax=ax[:,0], showLegend=False)
		self.plotObsRecOtherRatio(d1, d2, 'e', r'$ecc$', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1], ax=ax[:,1], showLegend=False)
		self.plotObsRecOtherRatio(d1, d2, 'm1', r'$m_1$ $(M_\odot)$', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=m1xlim, ax=ax[:,2], showLegend=False)
		self.plotObsRecOtherRatio(d1, d2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1], ax=ax[:,3], showLegend=True, legendLoc = 'upper left')
		ax[0,0].set_ylabel('CDF', fontsize=16)
		ax[1,0].set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=16)
		#ax[1,0].set_ylabel('PDF', fontsize=16)
		ax[2,0].set_ylabel('Ratio', fontsize=16)
		for i in range(3):
			for j in range(4):
				if (i != 2):
					ax[i,j].set_xticklabels([])
				if (j != 0):
					ax[i,j].set_yticklabels([])

		f.subplots_adjust(hspace=0, wspace=0.07)
		f.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherRatioCombined.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)


	def plotAllObsRecOtherPDF(self, d1, d1C, d2, d2C, d3, d3C):

		suffix = ''

		m1xlim = self.m1xlim
		#m1xlim[1] -= 0.01
		f,ax = plt.subplots(1,4,figsize=(20, 4), sharey=True)
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5], ax=ax[0], showLegend=True, legendLoc = 'upper right')
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'e', r'$ecc$', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1], ax=ax[1], showLegend=False)
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'm1', r'$m_1$ $(M_\odot)$', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=m1xlim, ax=ax[2], showLegend=False)
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1], ax=ax[3], showLegend=False)
		#ax[0].set_ylabel(r'$\frac{N_i}{\sum_i N_i}$', fontsize=16)
		#ax[0].set_ylabel('PDF', fontsize=16)
		ax[0].set_ylabel(r'$N$', fontsize=16)


		f.subplots_adjust(wspace=0.07)
		f.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherPDFCombined.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)

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



