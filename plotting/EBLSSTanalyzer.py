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
#matplotlib.use('Agg')

import cartopy.crs as ccrs


import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sys
sys.path.insert(0, '/Users/ageller/WORK/LSST/onGitHub/EBLSST/code')
from SED import SED

from dust_extinction.parameter_averages import F04

def getrMagBinary(L1, T1, g1, r1, L2, T2, g2, r2, M_H, dist, AV, RV=3.1):
#very slow

	filterFilesRoot = '/Users/ageller/WORK/LSST/onGitHub/EBLSST/input/filters/'
	wavelength = (552. + 691.)/2.

	SED1 = SED()
	SED1.filters = ['r_']
	SED1.filterFilesRoot = filterFilesRoot
	SED1.T = T1*units.K
	SED1.R = r1*units.solRad
	SED1.L = L1*units.solLum
	SED1.logg = g1
	SED1.M_H = M_H
	SED1.EBV = AV/RV #could use this to account for reddening in SED
	SED1.initialize()

	SED2 = SED()
	SED2.filters = ['r_']
	SED2.filterFilesRoot = filterFilesRoot
	SED2.T = T2*units.K
	SED2.R = r2*units.solRad
	SED2.L = L2*units.solLum
	SED2.logg = g2
	SED2.M_H = M_H
	SED2.EBV = AV/RV #could use this to account for reddening in SED
	SED2.initialize()

	#estimate a combined Teff value, as I do in the N-body codes (but where does this comes from?)
	logLb = np.log10(L1 + L2)
	logRb = 0.5*np.log10(r1**2. + r2**2.)
	T12 = 10.**(3.762 + 0.25*logLb - 0.5*logRb)
	#print(L1, L2, T1, T2, T12)


	#one option for getting the extinction
	ext = F04(Rv=RV)

	Lconst1 = SED1.getLconst()
	Lconst2 = SED2.getLconst()

	Ared = ext(wavelength*units.nm)*AV

	Fv1 = SED1.getFvAB(dist*units.kpc, 'r_', Lconst = Lconst1)
	Fv2 = SED2.getFvAB(dist*units.kpc, 'r_', Lconst = Lconst2)
	Fv = Fv1 + Fv2
	return -2.5*np.log10(Fv) + Ared #AB magnitude 


def getrMagSingle(L1, T1, g1, r1, M_H, dist, AV, RV=3.1):
#very slow

	filterFilesRoot = '/Users/ageller/WORK/LSST/onGitHub/EBLSST/input/filters/'
	wavelength = (552. + 691.)/2.

	SED1 = SED()
	SED1.filters = ['r_']
	SED1.filterFilesRoot = filterFilesRoot
	SED1.T = T1*units.K
	SED1.R = r1*units.solRad
	SED1.L = L1*units.solLum
	SED1.logg = g1
	SED1.M_H = M_H
	SED1.EBV = AV/RV #could use this to account for reddening in SED
	SED1.initialize()


	#one option for getting the extinction
	ext = F04(Rv=RV)

	Lconst1 = SED1.getLconst()

	Ared = ext(wavelength*units.nm)*AV
	Fv = SED1.getFvAB(dist*units.kpc, 'r_', Lconst = Lconst1)

	return -2.5*np.log10(Fv) + Ared #AB magnitude 



class EBLSSTanalyzer(object):

	def __init__(self, 
				directory = 'output_files', 
				plotsDirectory ='plots',
				doIndividualPlots = True,
				cluster = False,
				clusterType = None,
				mMean = 0.5,
				filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_', 'all'],
				Pcut = 0.1,
				Nlim = 1,
				onlyDWD = False,
				oneGiant = False,
				twoGiants = False,
				noGiants = False):

		self.directory = directory
		self.plotsDirectory = plotsDirectory
		self.doIndividualPlots = doIndividualPlots
		self.cluster = cluster
		self.clusterType = clusterType
		self.mMean = mMean #assumed mean stellar mass (for cluster analysis)
		self.filters = filters
		self.Pcut = Pcut #cutoff in percent error for "recovered"
		self.Nlim = Nlim #minimum number of lines to consider (for all, obs, rec, etc.)
		self.onlyDWD = onlyDWD
		self.oneGiant = oneGiant
		self.twoGiants = twoGiants
		self.noGiants = noGiants

		self.m1xlim = [0.,3.]

		self.outputNumbers = dict()
		self.outputNumbers['OpSimID'] = []
		self.outputNumbers['RA'] = []
		self.outputNumbers['Dec'] = []
		self.outputNumbers['NstarsReal'] = []
		self.outputNumbers['fb'] = []
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
		# self.outputHists['eachField'] = dict()

		self.individualPlots = dict()

		self.allRec = pd.DataFrame()

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
		y = [0.20, 0.35, 0.44, 0.70, 0.75]
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

		for a in ax:
			a.tick_params(axis='both', which='major', labelsize=14)

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
			ax[0].set_ylabel('CDF', fontsize=18)

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
			ax[1].set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=18)

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
			ax[2].set_ylabel('Ratio', fontsize=18)
		ax[2].set_ylim(0.5e-5,1)
		ax[2].set_yscale('log')
		ax[2].set_xlabel(xtitle, fontsize=18)

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
			ax[0].legend(handles=[lAll, lObs, lRec, lObsAll, lRecAll, lRecObs], loc=legendLoc, fontsize=10.5)


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOtherRatio.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)


	def plotObsRecOther_new(self, d1, d2, key, xtitle, fname,  xlim = None, ax = None, showLegend = True):
		#http://jfly.iam.u-tokyo.ac.jp/color/image/pallete.jpg
		#https://thenode.biologists.com/data-visualization-with-flying-colors/research/
		#https://medium.com/cafe-pixo/inclusive-color-palettes-for-the-web-bbfe8cf2410e
		#https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
		#https://coolors.co/
		c1 = '#000000'#all
		c2 = '#898989' #obs
		c3 = '#FF1B1C'#rec
		w1 = 2
		w2 = 2
		w3 = 3

		saveit = False
		if (ax is None):
			saveit = True
			f,ax = plt.subplots(1,1,figsize=(5, 5), sharex=True)
			ax.tick_params(axis='both', which='major', labelsize=14)


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
		#this is the fraction in each bin
		ax.step(bin_edges, histAll/np.sum(histAll), color=c1, linewidth=w1, label='All')
		ax.step(bin_edges, histObs/np.sum(histObs), color=c2, linewidth=w2, label='Observable')
		ax.step(bin_edges, allhistRec/np.sum(allhistRec), color=c3, linewidth=w3, label='Recoverable')
		ax.step(bin_edgesOD, histAllOD/np.sum(histAll), color=c1, linewidth=w1, linestyle=':')
		ax.step(bin_edgesOD, histObsOD/np.sum(histObs), color=c2, linewidth=w2, linestyle=':')
		ax.step(bin_edgesOD, allhistRecOD/np.sum(allhistRec), color=c3, linewidth=w3, linestyle=':')
		ax.set_ylim(0.5e-5, 1.9)
		ax.set_yscale('log')
		if (saveit):
			ax.set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=18)

		ax.set_xlabel(xtitle, fontsize=18)

		if (xlim is not None):
			ax.set_xlim(xlim[0], xlim[1])

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		if (showLegend):
			lAll = mlines.Line2D([], [], color=c1, linewidth=w1, label='All')
			lObs = mlines.Line2D([], [], color=c2, linewidth=w2, label='Obs.')
			lRec = mlines.Line2D([], [], color=c3, linewidth=w3, label='Rec.')
			lASN = mlines.Line2D([], [], color='#5FC8D0', linewidth=1, label='ASAS-SN')
			ax.legend(handles=[lAll, lObs, lRec, lASN], fontsize=10.5, ncol = 4, bbox_to_anchor=(0.15, 1.1, 1, 0.1))


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOther_new.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)

	def plotObsRecCDF_new(self, d1, d2, key, xtitle, fname,  xlim = None, ax = None, showLegend = True):
		#http://jfly.iam.u-tokyo.ac.jp/color/image/pallete.jpg
		#https://thenode.biologists.com/data-visualization-with-flying-colors/research/
		#https://medium.com/cafe-pixo/inclusive-color-palettes-for-the-web-bbfe8cf2410e
		#https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
		#https://coolors.co/
		c1 = '#000000'#all
		c2 = '#898989' #obs
		c3 = '#FF1B1C'#rec
		w1 = 2
		w2 = 2
		w3 = 3

		saveit = False
		if (ax is None):
			saveit = True
			f,ax = plt.subplots(1,1,figsize=(5, 5), sharex=True)
			ax.tick_params(axis='both', which='major', labelsize=14)

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
		ax.step(bin_edges, cdfAll, color=c1, linewidth=w1, label='All')
		ax.step(bin_edges, cdfObs, color=c2, linewidth=w2, label='Observable')
		ax.step(bin_edges, cdfRec, color=c3, linewidth=w3, label='Recoverable')
		ax.step(bin_edgesOD, cdfAllOD, color=c1, linewidth=w1, linestyle=':')
		ax.step(bin_edgesOD, cdfObsOD, color=c2, linewidth=w2, linestyle=':')
		ax.step(bin_edgesOD, cdfRecOD, color=c3, linewidth=w3, linestyle=':')
		ax.set_ylim(-0.01,1.01)
		if (saveit):
			ax.set_ylabel('CDF', fontsize=18)

		ax.set_xlabel(xtitle, fontsize=18)

		if (xlim is not None):
			ax.set_xlim(xlim[0], xlim[1])

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		if (showLegend):
			lAll = mlines.Line2D([], [], color=c1, linewidth=w1, label='All')
			lObs = mlines.Line2D([], [], color=c2, linewidth=w2, label='Obs.')
			lRec = mlines.Line2D([], [], color=c3, linewidth=w3, label='Rec.')
			lASN = mlines.Line2D([], [], color='#5FC8D0', linewidth=1, label='ASAS-SN')
			ax.legend(handles=[lAll, lObs, lRec, lASN], fontsize=10.5, ncol = 4, bbox_to_anchor=(0.15, 1.1, 1, 0.1))


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOther_new.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)


	def plotObsRecCDFOther_new(self, d1, d2, key, xtitle, fname,  xlim = None, ax = [None], showLegend = True, legendLoc = 'lower right'):
		#http://jfly.iam.u-tokyo.ac.jp/color/image/pallete.jpg
		#https://thenode.biologists.com/data-visualization-with-flying-colors/research/
		#https://medium.com/cafe-pixo/inclusive-color-palettes-for-the-web-bbfe8cf2410e
		#https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
		#https://coolors.co/
		c1 = '#000000'#all
		c2 = '#898989' #obs
		c3 = '#FF1B1C'#rec
		w1 = 2
		w2 = 2
		w3 = 3

		saveit = False
		if (ax[0] is None):
			saveit = True
			f,ax = plt.subplots(2,1,figsize=(5, 8), sharex=True)


		for a in ax:
			a.tick_params(axis='both', which='major', labelsize=14)

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
		ax[0].step(bin_edges, cdfAll, color=c1, linewidth=w1, label='All')
		ax[0].step(bin_edges, cdfObs, color=c2, linewidth=w2, label='Observable')
		ax[0].step(bin_edges, cdfRec, color=c3, linewidth=w3, label='Recoverable')
		ax[0].step(bin_edgesOD, cdfAllOD, color=c1, linewidth=w1, linestyle=':')
		ax[0].step(bin_edgesOD, cdfObsOD, color=c2, linewidth=w2, linestyle=':')
		ax[0].step(bin_edgesOD, cdfRecOD, color=c3, linewidth=w3, linestyle=':')
		ax[0].set_ylim(-0.01,1.01)
		if (saveit):
			ax[0].set_ylabel('CDF', fontsize=18)

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
		#this is the fraction in each bin
		ax[1].step(bin_edges, histAll/np.sum(histAll), color=c1, linewidth=w1, label='All')
		ax[1].step(bin_edges, histObs/np.sum(histObs), color=c2, linewidth=w2, label='Observable')
		ax[1].step(bin_edges, allhistRec/np.sum(allhistRec), color=c3, linewidth=w3, label='Recoverable')
		ax[1].step(bin_edgesOD, histAllOD/np.sum(histAll), color=c1, linewidth=w1, linestyle=':')
		ax[1].step(bin_edgesOD, histObsOD/np.sum(histObs), color=c2, linewidth=w2, linestyle=':')
		ax[1].step(bin_edgesOD, allhistRecOD/np.sum(allhistRec), color=c3, linewidth=w3, linestyle=':')
		ax[1].set_ylim(0.5e-5, 1.9)
		ax[1].set_yscale('log')
		if (saveit):
			ax[1].set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=18)



		ax[-1].set_xlabel(xtitle, fontsize=18)

		if (xlim is not None):
			ax[0].set_xlim(xlim[0], xlim[1])
			ax[1].set_xlim(xlim[0], xlim[1])

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		if (showLegend):
			lAll = mlines.Line2D([], [], color=c1, linewidth=w1, label='All')
			lObs = mlines.Line2D([], [], color=c2, linewidth=w2, label='Obs.')
			lRec = mlines.Line2D([], [], color=c3, linewidth=w3, label='Rec.')
			ax[0].legend(handles=[lAll, lObs, lRec], loc=legendLoc, fontsize=10.5)


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOther_new.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)

	def plotObsRecOtherRatio_new(self, d1, d2, key, xtitle, fname,  xlim = None, ax2 = None, showLegend = True, legendLoc = 'lower right'):
		#http://jfly.iam.u-tokyo.ac.jp/color/image/pallete.jpg
		#https://thenode.biologists.com/data-visualization-with-flying-colors/research/
		#https://medium.com/cafe-pixo/inclusive-color-palettes-for-the-web-bbfe8cf2410e
		#https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
		#https://coolors.co/
		c1 = '#898989' #obs
		c2 = '#000000'#all
		c3 = '#FF1B1C'#rec
		w1 = 2
		w2 = 2
		w3 = 3

		saveit = False
		if (ax2 is None):
			f2,ax2 = plt.subplots(1,1,figsize=(5, 4))
			ax2.tick_params(axis='both', which='major', labelsize=14)

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

		ratio = histObs/histAll
		check = np.isnan(ratio)
		ratio[check]=0.
		ax2.step(bin_edges, ratio, color=c1, linewidth=w1, label='Observable/All')
		#ax2.plot(bin_edges - binHalf, ratio, 'o',color=c1, markersize=5, markeredgecolor=c2)


		ratio = allhistRec/histObs
		check = np.isnan(ratio)
		ratio[check]=0.
		ax2.step(bin_edges, ratio, color=c2, linewidth=w2, label='Recoverable/Observable')
		#ax2.plot(bin_edges - binHalf, ratio, 'o',color=c2, markersize=5, markeredgecolor=c3)

		ratio = allhistRec/histAll
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax2.step(bin_edges, ratio, color=c3, linewidth=w3, label='Recoverable/All')
		#ax2.plot(bin_edges - binHalf, ratio, 'o',color=c1, markersize=5, markeredgecolor=c3)


		ratio = histObsOD/histAllOD
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax2.step(bin_edgesOD, ratio, color=c1, linewidth=w1, linestyle=':')
		#ax2.plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c1, markersize=3.5, markeredgecolor=c2)


		ratio = allhistRecOD/histObsOD
		check = np.isnan(ratio)
		ratio[check]=0.
		ax2.step(bin_edgesOD, ratio, color=c2, linewidth=w2, linestyle=':')
		#ax2.plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c2, markersize=3.5, markeredgecolor=c3)

		ratio = allhistRecOD/histAllOD
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax2.step(bin_edgesOD, ratio, color=c3, linewidth=w3, linestyle=':')
		#ax2.plot(bin_edgesOD - binHalfOD, ratio, 'o',color=c1, markersize=3.5, markeredgecolor=c3)

		if (saveit):
			ax2.set_ylabel('Ratio', fontsize=18)
		ax2.set_ylim(0.5e-5,1)
		ax2.set_yscale('log')
		ax2.set_xlabel(xtitle, fontsize=18)


		if (xlim is not None):
			ax2.set_xlim(xlim[0], xlim[1])

		# ax1.legend()
		# ax2.legend()
		# ax3.legend()
		if (showLegend):
			lObsAll = mlines.Line2D([], [], color=c1, linewidth=w1, label='Obs./All')
			lRecObs = mlines.Line2D([], [], color=c2, linewidth=w2, label='Rec./Obs.')
			lRecAll = mlines.Line2D([], [], color=c3, linewidth=w3, label='Rec./All')
			ax2.legend(handles=[lRecObs, lObsAll, lRecAll], loc=legendLoc, fontsize=10.5)


		if (saveit):
			f2.subplots_adjust(hspace=0)
			f2.savefig(fname+'_ObsRecRatio_new.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f2)

	def plotObsRecOther_clusters_new(self, d1, d2, df1, df2, key, xtitle, fname,  xlim = None, ax = None, showLegend = True, legendLoc = 'lower right'):
		#http://jfly.iam.u-tokyo.ac.jp/color/image/pallete.jpg
		#https://thenode.biologists.com/data-visualization-with-flying-colors/research/
		#https://medium.com/cafe-pixo/inclusive-color-palettes-for-the-web-bbfe8cf2410e
		#https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
		#https://coolors.co/
		c1 = '#000000'#all
		c2 = '#898989' #obs
		c3 = '#FF1B1C'#rec
		w1 = 2
		w2 = 2
		w3 = 3

		saveit = False
		if (ax is None):
			saveit = True
			f,ax = plt.subplots(1,1,figsize=(5, 5), sharex=True)
			ax.tick_params(axis='both', which='major', labelsize=14)


		#I want to divide (or subtract?) the field from the clusters.  This assumes the field and clusters have same bins
		histAll = d1[key+'hAll']
		histObs = d1[key+'hObs']
		allhistRec = d1[key+'hRec']['all']
		bin_edges = d1[key+'b']

		histAllOD = d2[key+'hAll']
		histObsOD = d2[key+'hObs']
		allhistRecOD = d2[key+'hRec']['all']
		bin_edgesOD = d2[key+'b']		

		FhistAll = df1[key+'hAll']
		FhistObs = df1[key+'hObs']
		FallhistRec = df1[key+'hRec']['all']
		Fbin_edges = df1[key+'b']

		FhistAllOD = df2[key+'hAll']
		FhistObsOD = df2[key+'hObs']
		FallhistRecOD = df2[key+'hRec']['all']
		Fbin_edgesOD = df2[key+'b']	

		binHalf = (bin_edges[1] - bin_edges[0])/2.
		binHalfOD = (bin_edgesOD[1] - bin_edgesOD[0])/2.

		#PDF --need to divide by the bin size
		#this is the fraction in each bin
		# ax.step(bin_edges, histAll/np.sum(histAll) - FhistAll/np.sum(FhistAll), color=c1, linewidth=w1, label='All')
		# ax.step(bin_edges, histObs/np.sum(histObs) - FhistObs/np.sum(FhistObs), color=c2, linewidth=w2, label='Observable')
		# ax.step(bin_edges, allhistRec/np.sum(allhistRec) - FallhistRec/np.sum(FallhistRec), color=c3, linewidth=w3, label='Recoverable')
		# ax.step(bin_edgesOD, histAllOD/np.sum(histAll) - FhistAllOD/np.sum(FhistAll), color=c1, linewidth=w1, linestyle=':')
		# ax.step(bin_edgesOD, histObsOD/np.sum(histObs) - FhistObsOD/np.sum(FhistObs), color=c2, linewidth=w2, linestyle=':')
		# ax.step(bin_edgesOD, allhistRecOD/np.sum(allhistRec) - FallhistRecOD/np.sum(FallhistRec), color=c3, linewidth=w3, linestyle=':')

		ratio = histAll/np.sum(histAll)/(FhistAll/np.sum(FhistAll))
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax.step(bin_edges, ratio, color=c1, linewidth=w1, label='All')

		ratio = histObs/np.sum(histObs)/(FhistObs/np.sum(FhistObs))
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax.step(bin_edges, ratio, color=c2, linewidth=w2, label='Observable')


		ratio = allhistRec/np.sum(allhistRec)/(FallhistRec/np.sum(FallhistRec))
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax.step(bin_edges, ratio, color=c3, linewidth=w3, label='Recoverable')

		ratio = histAllOD/np.sum(histAll)/(FhistAllOD/np.sum(FhistAll))
		check = np.isnan(ratio)
		ratio[check]=0.	
		ax.step(bin_edgesOD, ratio, color=c1, linewidth=w1, linestyle=':')

		ratio = histObsOD/np.sum(histObs)/(FhistObsOD/np.sum(FhistObs))
		check = np.isnan(ratio)
		ratio[check]=0.			
		ax.step(bin_edgesOD, ratio, color=c2, linewidth=w2, linestyle=':')

		ratio = allhistRecOD/np.sum(allhistRec)/(FallhistRecOD/np.sum(FallhistRec))
		check = np.isnan(ratio)
		ratio[check]=0.		
		ax.step(bin_edgesOD, ratio, color=c3, linewidth=w3, linestyle=':')

		ax.set_ylim(0,5)
		#ax.set_ylim(0.5e-5, 1.9)
		#ax.set_yscale('log')
		if (saveit):
			ax.set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=18)


		ax.set_xlabel(xtitle, fontsize=18)

		if (xlim is not None):
			ax.set_xlim(xlim[0], xlim[1])


		if (showLegend):
			lAll = mlines.Line2D([], [], color=c1, linewidth=w1, label='All')
			lObs = mlines.Line2D([], [], color=c2, linewidth=w2, label='Obs.')
			lRec = mlines.Line2D([], [], color=c3, linewidth=w3, label='Rec.')
			ax.legend(handles=[lAll, lObs, lRec], loc=legendLoc, fontsize=10.5)


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOther_clusters_new.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)

	def plotObsRecOtherPDF(self, d1, d1C, d2, d2C, d3, d3C, key, xtitle, fname,  xlim = None, ax1 = None, ax2 = None, showLegend = True, legendLoc = 'lower right',includeASASSN=False, c1=None, c2=None, c3=None, c4=None):
		if (c1 is None):
			c1 = '#0294A5'  #turqoise
		if (c2 is None):
			c2 = '#d95f02' #orange from color brewer
		if (c3 is None):
			c3 = '#00353E' #slate
		if (c4 is None):
			c4 = '#508201' #olive

		saveit = False
		if (ax1 is None):
			saveit = True
			f,ax1 = plt.subplots(figsize=(5, 4))

		ax1.tick_params(axis='both', which='major', labelsize=14)
		if (ax2 is not None):
			ax2.tick_params(axis='both', which='major', labelsize=14)

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

		ax1.step(bin_edges1,  allhistRec1, color=c3, label='Field')
		ax1.step(bin_edges1C, allhistRec1C, color=c3,linestyle=':')
		ax1.step(bin_edges2,  allhistRec2, color=c2, label='GCs')
		ax1.step(bin_edges2C, allhistRec2C, color=c2,linestyle=':')
		ax1.step(bin_edges3,  allhistRec3, color=c1, label='OCs')
		ax1.step(bin_edges3C, allhistRec3C, color=c1,linestyle=':')


		#ax.set_ylim(0.5e-5, 1.9)
		ax1.set_ylim(1, 4e6)
		ax1.set_yscale('log')

		if (ax2 is not None):
			ratio1 = allhistRec1/histObs1
			check = np.isnan(ratio1)
			ratio1[check]=0.
			ax2.step(bin_edges1, ratio1, color=c3)
			ratio1C = allhistRec1C/histObs1C
			check = np.isnan(ratio1C)
			ratio1C[check]=0.
			ax2.step(bin_edges1C, ratio1C, color=c3, linestyle=':')

			ratio2 = allhistRec2/histObs2
			check = np.isnan(ratio2)
			ratio2[check]=0.
			ax2.step(bin_edges2, ratio2, color=c2)
			ratio2C = allhistRec2C/histObs2C
			check = np.isnan(ratio2C)
			ratio2C[check]=0.
			ax2.step(bin_edges2C, ratio2C, color=c2, linestyle=':')

			ratio3 = allhistRec3/histObs3
			check = np.isnan(ratio3)
			ratio3[check]=0.
			ax2.step(bin_edges3, ratio3, color=c1)
			ratio3C = allhistRec3C/histObs3C
			check = np.isnan(ratio3C)
			ratio3C[check]=0.
			ax2.step(bin_edges3C, ratio3C, color=c1, linestyle=':')

			ax2.set_ylim(1e-3,5)
			ax2.set_yscale('log')
			ax2.set_xlabel(xtitle, fontsize=18)


		if (saveit):
			#ax.set_ylabel(r'$N_i/\sum_i N_i$', fontsize=16)
			ax1.set_ylabel(r'$N$', fontsize=18)


		if (xlim is not None):
			ax1.set_xlim(xlim[0], xlim[1])
			if (ax2 is not None):
				ax2.set_xlim(xlim[0], xlim[1])

		if (showLegend):
			lAll = mlines.Line2D([], [], color=c3, label='Field')
			lObs = mlines.Line2D([], [], color=c2, label='GCs')
			lRec = mlines.Line2D([], [], color=c1, label='OCs')
			if (includeASASSN):
				lASN = mlines.Line2D([], [], color='lightgray', label='ASAS-SN')
			ax1.legend(handles=[lAll, lObs, lRec,lASN], loc=legendLoc, fontsize=12)


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_ObsRecOtherPDF.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)

	def plotRecFilterPDF(self, d1, d1C, key, xtitle, fname,  xlim = None, ax = None, showLegend = True, legendLoc = 'lower right'):

		colors = ['purple', 'green', 'red', 'darkorange', 'gold', 'yellow', 'black']

		saveit = False
		if (ax is None):
			saveit = True
			f,ax = plt.subplots(figsize=(5, 4))

		bin_edges1 = d1[key+'b']
		bin_edges1C = d1C[key+'b']

		outrec = []
		for f,c in zip(self.filters, colors):
			rec = d1[key+'hRec'][f]
			recC = d1C[key+'hRec'][f]
			#ax.step(bin_edges1, rec, color=c)
			#ax.step(bin_edges1, recC, color=c, linestyle=':', label=f)
			ax.plot(bin_edges1, rec, color=c)
			ax.plot(bin_edges1, recC, color=c, linestyle=':', label=f)
			outrec.append(sum(rec))
			print(f,sum(rec))

		ax.set_ylim(1, 4e6)
		ax.set_yscale('log')


		if (saveit):
			#ax.set_ylabel(r'$N_i/\sum_i N_i$', fontsize=16)
			ax.set_ylabel(r'$N$', fontsize=16)


		if (xlim is not None):
			ax.set_xlim(xlim[0], xlim[1])

		if (showLegend):
			lab = []
			for f,c in zip(self.filters, colors):
				lf = mlines.Line2D([], [], color=c, label=f)
				lab.append(lf)
			ax.legend(handles=lab, loc=legendLoc)


		if (saveit):
			f.subplots_adjust(hspace=0)
			f.savefig(fname+'_recFilter.pdf',format='pdf', bbox_inches = 'tight')
			plt.close(f)

		return outrec

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
		mbSize = 0.4
		mbSizeSmall = 0.1
		qbSize = 0.05
		ebSize = 0.05
		lpbSize = 0.25
		dbSize = 1.
		magbSize = 1.
		rbSize = 0.2
		mbins = np.arange(0,10+mbSize, mbSize, dtype='float')
		mbinsSmall = np.arange(0,10+mbSizeSmall, mbSizeSmall, dtype='float')
		qbins = np.arange(0,1+qbSize, qbSize, dtype='float')
		ebins = np.arange(0, 1+ebSize, ebSize, dtype='float')
		lpbins = np.arange(-3, 10+lpbSize, lpbSize, dtype='float')
		dbins = np.arange(0, 40+dbSize, dbSize, dtype='float')
		magbins = np.arange(11, 25+magbSize, magbSize, dtype='float')
		rbins = np.arange(0, 5+rbSize, rbSize, dtype='float')

		CDFfac = 1000.
		mbinsCDF = np.arange(0,10+mbSize/CDFfac, mbSize/CDFfac, dtype='float')
		mbinsCDFSmall = np.arange(0,10+mbSizeSmall/CDFfac, mbSizeSmall/CDFfac, dtype='float')
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
		m1hAllSmall = np.zeros_like(mbinsSmall)[1:]
		qhAll = np.zeros_like(qbins)[1:]
		ehAll = np.zeros_like(ebins)[1:]
		lphAll = np.zeros_like(lpbins)[1:]
		dhAll = np.zeros_like(dbins)[1:]
		maghAll = np.zeros_like(magbins)[1:]
		rhAll = np.zeros_like(rbins)[1:]
		#Observable
		m1hObs = np.zeros_like(mbins)[1:]
		m1hObsSmall = np.zeros_like(mbinsSmall)[1:]
		qhObs = np.zeros_like(qbins)[1:]
		ehObs = np.zeros_like(ebins)[1:]
		lphObs = np.zeros_like(lpbins)[1:]
		dhObs = np.zeros_like(dbins)[1:]
		maghObs = np.zeros_like(magbins)[1:]
		rhObs = np.zeros_like(rbins)[1:]
		#Recovered
		m1hRec = dict()
		m1hRecSmall = dict()
		qhRec = dict()
		ehRec = dict()
		lphRec = dict()
		dhRec = dict()
		maghRec = dict()
		rhRec = dict()
		for f in self.filters:
			m1hRec[f] = np.zeros_like(mbins)[1:]
			m1hRecSmall[f] = np.zeros_like(mbinsSmall)[1:]
			qhRec[f] = np.zeros_like(qbins)[1:]
			ehRec[f] = np.zeros_like(ebins)[1:]
			lphRec[f] = np.zeros_like(lpbins)[1:]
			dhRec[f] = np.zeros_like(dbins)[1:]
			maghRec[f] = np.zeros_like(magbins)[1:]
			rhRec[f] = np.zeros_like(rbins)[1:]

		#blanks for the CDFs
		#All
		m1hAllCDF = np.zeros_like(mbinsCDF)[1:]
		m1hAllCDFSmall = np.zeros_like(mbinsCDFSmall)[1:]
		qhAllCDF = np.zeros_like(qbinsCDF)[1:]
		ehAllCDF = np.zeros_like(ebinsCDF)[1:]
		lphAllCDF = np.zeros_like(lpbinsCDF)[1:]
		dhAllCDF = np.zeros_like(dbinsCDF)[1:]
		maghAllCDF = np.zeros_like(magbinsCDF)[1:]
		rhAllCDF = np.zeros_like(rbinsCDF)[1:]
		#Observable
		m1hObsCDF = np.zeros_like(mbinsCDF)[1:]
		m1hObsCDFSmall = np.zeros_like(mbinsCDFSmall)[1:]
		qhObsCDF = np.zeros_like(qbinsCDF)[1:]
		ehObsCDF = np.zeros_like(ebinsCDF)[1:]
		lphObsCDF = np.zeros_like(lpbinsCDF)[1:]
		dhObsCDF = np.zeros_like(dbinsCDF)[1:]
		maghObsCDF = np.zeros_like(magbinsCDF)[1:]
		rhObsCDF = np.zeros_like(rbinsCDF)[1:]
		#Recovered
		m1hRecCDF = dict()
		m1hRecCDFSmall = dict()
		qhRecCDF = dict()
		ehRecCDF = dict()
		lphRecCDF = dict()
		dhRecCDF = dict()
		maghRecCDF = dict()
		rhRecCDF = dict()
		for f in self.filters:
			m1hRecCDF[f] = np.zeros_like(mbinsCDF)[1:]
			m1hRecCDFSmall[f] = np.zeros_like(mbinsCDFSmall)[1:]
			qhRecCDF[f] = np.zeros_like(qbinsCDF)[1:]
			ehRecCDF[f] = np.zeros_like(ebinsCDF)[1:]
			lphRecCDF[f] = np.zeros_like(lpbinsCDF)[1:]
			dhRecCDF[f] = np.zeros_like(dbinsCDF)[1:]
			maghRecCDF[f] = np.zeros_like(magbinsCDF)[1:]
			rhRecCDF[f] = np.zeros_like(rbinsCDF)[1:]

		#Read in all the data and make the histograms
		files = os.listdir(self.directory)
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
					p = f.find('__output_file.csv')
					clusterID = f[0:p]
				else:
					Nmult = header['NstarsTRILEGAL'][0]
				NstarsReal = Nmult

				#Nmult = 1.


				#to normalize for the difference in period, since I limit this for the field for computational efficiency (but not for clusters, which I allow to extend to the hard-soft boundary)
				intNorm = 1.
				#instead I will limit the binary frequency, like I do with the clusters
				# if (not self.cluster):
				# 	intAll = 1.
				# 	intCut = self.RagNormal(np.log10(3650), cdf = True)
				# 	intNorm = intCut/intAll


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
						# g1tmp = row['logg1']
						# data.at[index, 'logg1'] = row['logg2']
						# data.at[index, 'logg2'] = g1tmp							
						# #will want to swap L1 and T1 too if using newer files


				#calculate logg values
				data['r1'].replace(0., 1e-10, inplace = True)
				data['r2'].replace(0., 1e-10, inplace = True)
				data['m1'].replace(0., 1e-10, inplace = True)
				data['m2'].replace(0., 1e-10, inplace = True)
				logg1 = np.log10((constants.G*data['m1'].values*units.solMass/(data['r1'].values*units.solRad)**2.).decompose().to(units.cm/units.s**2.).value)
				logg2 = np.log10((constants.G*data['m2'].values*units.solMass/(data['r2'].values*units.solRad)**2.).decompose().to(units.cm/units.s**2.).value)
				data['logg1'] = logg1
				data['logg2'] = logg2

				#get the approximate magnitudes for binary from the isochrone for clusters, so that I can limit the sample
				if (self.cluster):
					#print(clusterID, data['appMagMean_r'])
					#read in the isochrone file that I have already downloaded
					isoFile = '/Users/ageller/WORK/LSST/onGitHub/EBLSST/input/isochrones/'+self.clusterType+'s/'+clusterID+'.csv'
					isodf = pd.read_csv(isoFile, comment='#')
					isodf.sort_values('logL', inplace=True)
					#get the reddening
					ext = F04(Rv=3.1)
					wavelength = (552. + 691.)/2.
					Ared = ext(wavelength*units.nm)*data['Av'][0] #all stars in the cluster will have same Av value
					rMag = data['appMagMean_r'].values
					for index, row in data.iterrows():
						if (row['appMagMean_r'] == -999.):
							Mag = np.interp(np.log10(row['L1'] + row['L2']), isodf['logL'], isodf['rmag'])
							rMag[index] = Mag + 5*np.log10(row['d']*100) + Ared #d is in kpc, and I need d(pc)/10pc = d(kpc)*1000/10 = d(kpd)*100
					data['appMagMean_r'] = rMag
					#print(data['appMagMean_r'])

				Nall = len(data.index)/intNorm  #saving this in case we want to limit the entire analysis to DWDs, but still want the full sample size for the cumulative numbers

##        self.magLims = np.array([15.8, 25.]) #lower and upper limits on the magnitude detection assumed for LSST: 15.8 = rband saturation from Science Book page 57, before Section 3.3; 24.5 is the desired detection limit

				#limit to the magnitude range of LSST
				data = data.loc[(data['appMagMean_r'] <= 25) & (data['appMagMean_r'] > 15.8)].reset_index()
				print('   file length, N in mag limits', Nall, len(data.index))
				if (len(data) > 0):
					if (data['m1'][0] != -1): #these are files that were not even run

						#I will use this for the total number of objects in the sample (as printed near the bottom)
						NallMag = len(data.index)






						if (self.onlyDWD):
							data = data.loc[(data['logg1'] > 6) & (data['logg1'] < 12) & (data['logg2'] > 6) & (data['logg2'] < 12)]

						if (self.oneGiant):
							data = data.loc[((data['Teff1'] < 5500) & (data['L1'] > 10)) | (data['Teff2'] < 5500) & (data['L2'] > 10)]
						
						if (self.twoGiants):
							data = data.loc[((data['Teff1'] < 5500) & (data['L1'] > 10)) & (data['Teff2'] < 5500) & (data['L2'] > 10)]

						if (self.noGiants):
							data = data.loc[((data['Teff1'] > 5500) | (data['L1'] < 10)) & (data['Teff2'] > 5500) | (data['L2'] < 10)]




						prsa = data.loc[(data['appMagMean_r'] <= 19.5) & (data['appMagMean_r'] > 15.8) & (data['p'] < 1000) & (data['p'] > 0.5)]
						DWD = data.loc[(data['logg1'] > 6) & (data['logg1'] < 12) & (data['logg2'] > 6) & (data['logg2'] < 12)]

	###is this correct? (and the only place I need to normalize?) -- I think yes (the observed binary distribution should be cut at a period of the survey duration)
						NallPrsa = len(prsa.index)/intNorm
						NallDWD = len(DWD.index)/intNorm


	#should I limit this to the LSST magnitude limits?, but I don't think I can because I don't calculate the magnitude for each binary. 

						if (len(data.index) >= self.Nlim):
							#create histograms
							#All
							m1hAll0, m1b = np.histogram(data["m1"], bins=mbins)
							m1hAll0Small, m1bSmall = np.histogram(data["m1"], bins=mbinsSmall)
							qhAll0, qb = np.histogram(data["m2"]/data["m1"], bins=qbins)
							ehAll0, eb = np.histogram(data["e"], bins=ebins)
							lphAll0, lpb = np.histogram(np.ma.log10(data["p"].values).filled(-999), bins=lpbins)
							dhAll0, db = np.histogram(data["d"], bins=dbins)
							maghAll0, magb = np.histogram(data["appMagMean_r"], bins=magbins)
							rhAll0, rb = np.histogram(data["r2"]/data["r1"], bins=rbins)

							m1hAll0CDF, m1bCDF = np.histogram(data["m1"], bins=mbinsCDF)
							m1hAll0CDFSmall, m1bCDFSmall = np.histogram(data["m1"], bins=mbinsCDFSmall)
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
							dm1 = np.diff(m1bSmall)
							m1val = m1bSmall[:-1] + dm1/2.
							fb = np.sum(m1hAll0Small/np.sum(m1hAll0Small)*fbFit(m1val))
							Pcut = 3650.
							if (self.cluster):
								#account for the hard-soft boundary
								Pcut = min(3650., self.getPhs(header['clusterVdisp'].iloc[0]*units.km/units.s).to(units.day).value)
							fb *= self.RagNormal(np.log10(Pcut), cdf = True)
							print("   fb, Nbins, log10(Pcut) = ", fb, len(data.index), np.log10(Pcut))

							Nmult *= fb

										
							m1hAll += m1hAll0/Nall*Nmult
							m1hAllSmall += m1hAll0Small/Nall*Nmult
							qhAll += qhAll0/Nall*Nmult
							ehAll += ehAll0/Nall*Nmult
							lphAll += lphAll0/Nall*Nmult
							dhAll += dhAll0/Nall*Nmult
							maghAll += maghAll0/Nall*Nmult
							rhAll += rhAll0/Nall*Nmult

							m1hAllCDF += m1hAll0CDF/Nall*Nmult
							m1hAllCDFSmall += m1hAll0CDFSmall/Nall*Nmult
							qhAllCDF += qhAll0CDF/Nall*Nmult
							ehAllCDF += ehAll0CDF/Nall*Nmult
							lphAllCDF += lphAll0CDF/Nall*Nmult
							dhAllCDF += dhAll0CDF/Nall*Nmult
							maghAllCDF += maghAll0CDF/Nall*Nmult
							rhAllCDF += rhAll0CDF/Nall*Nmult


							# #save the individual histograms and CDFs
							# OpSimID = header['OpSimID'][0]
							# self.outputHists['eachField'][OpSimID] = dict()
							# #Histograms
							# #x
							# # self.outputHists['eachField'][OpSimID]['m1b'] = np.append(m1b, m1b[-1] + mbSize)
							# # self.outputHists['eachField'][OpSimID]['m1Smallb'] = np.append(m1bSmall, m1bSmall[-1] + mbSizeSmall)
							# # self.outputHists['eachField'][OpSimID]['qb'] = np.append(qb, qb[-1] + qbSize)
							# # self.outputHists['eachField'][OpSimID]['eb'] = np.append(eb, eb[-1] + ebSize)
							# # self.outputHists['eachField'][OpSimID]['lpb'] = np.append(lpb, lpb[-1] + lpbSize)
							# # self.outputHists['eachField'][OpSimID]['db'] = np.append(db, db[-1] + dbSize)
							# # self.outputHists['eachField'][OpSimID]['magb'] = np.append(magb, magb[-1] + magbSize)
							# # self.outputHists['eachField'][OpSimID]['rb'] = np.append(rb, rb[-1] + rbSize)
							# #y
							# self.outputHists['eachField'][OpSimID]['m1hAll'] = np.append(np.insert(m1hAll0/Nall*Nmult,0,0),0)
							# self.outputHists['eachField'][OpSimID]['m1SmallhAll'] = np.append(np.insert(m1hAll0Small/Nall,0,0),0)
							# self.outputHists['eachField'][OpSimID]['qhAll'] = np.append(np.insert(qhAll0/Nall*Nmult,0,0),0)
							# self.outputHists['eachField'][OpSimID]['ehAll'] = np.append(np.insert(ehAll0/Nall*Nmult,0,0),0)
							# self.outputHists['eachField'][OpSimID]['lphAll'] = np.append(np.insert(lphAll0/Nall*Nmult,0,0),0)
							# self.outputHists['eachField'][OpSimID]['dhAll'] = np.append(np.insert(dhAll0/Nall*Nmult,0,0),0)
							# self.outputHists['eachField'][OpSimID]['maghAll'] = np.append(np.insert(maghAll0/Nall*Nmult,0,0),0)
							# self.outputHists['eachField'][OpSimID]['rhAll'] = np.append(np.insert(rhAll0/Nall*Nmult,0,0),0)
							# #CDFs
							# #x
							# # self.outputHists['eachField'][OpSimID]['m1bCDF'] = m1bCDF
							# # self.outputHists['eachField'][OpSimID]['m1SmallbCDF'] = m1bCDFSmall
							# # self.outputHists['eachField'][OpSimID]['qbCDF'] = qbCDF
							# # self.outputHists['eachField'][OpSimID]['ebCDF'] = ebCDF
							# # self.outputHists['eachField'][OpSimID]['lpbCDF'] = lpbCDF
							# # self.outputHists['eachField'][OpSimID]['dbCDF'] = dbCDF
							# # self.outputHists['eachField'][OpSimID]['magbCDF'] = magbCDF
							# # self.outputHists['eachField'][OpSimID]['rbCDF'] = rbCDF
							# #y
							# self.outputHists['eachField'][OpSimID]['m1hAllCDF'] = np.insert(m1hAll0CDF/Nall*Nmult,0,0)
							# self.outputHists['eachField'][OpSimID]['m1SmallhAllCDF'] = np.insert(m1hAll0CDFSmall/Nall*Nmult,0,0)
							# self.outputHists['eachField'][OpSimID]['qhAllCDF'] = np.insert(qhAll0CDF/Nall*Nmult,0,0)
							# self.outputHists['eachField'][OpSimID]['ehAllCDF'] = np.insert(ehAll0CDF/Nall*Nmult,0,0)
							# self.outputHists['eachField'][OpSimID]['lphAllCDF'] = np.insert(lphAll0CDF/Nall*Nmult,0,0)
							# self.outputHists['eachField'][OpSimID]['dhAllCDF'] = np.insert(dhAll0CDF/Nall*Nmult,0,0)
							# self.outputHists['eachField'][OpSimID]['maghAllCDF'] = np.insert(maghAll0CDF/Nall*Nmult,0,0)
							# self.outputHists['eachField'][OpSimID]['rhAllCDF'] = np.insert(rhAll0CDF/Nall*Nmult,0,0)

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
								m1hObs0Small, m1bSmall = np.histogram(obs["m1"], bins=mbinsSmall)
								qhObs0, qb = np.histogram(obs["m2"]/obs["m1"], bins=qbins)
								ehObs0, eb = np.histogram(obs["e"], bins=ebins)
								lphObs0, lpb = np.histogram(np.ma.log10(obs["p"].values).filled(-999), bins=lpbins)
								dhObs0, db = np.histogram(obs["d"], bins=dbins)
								maghObs0, magb = np.histogram(obs["appMagMean_r"], bins=magbins)
								rhObs0, rb = np.histogram(obs["r2"]/obs["r1"], bins=rbins)
								m1hObs += m1hObs0/Nall*Nmult
								m1hObsSmall += m1hObs0Small/Nall*Nmult
								qhObs += qhObs0/Nall*Nmult
								ehObs += ehObs0/Nall*Nmult
								lphObs += lphObs0/Nall*Nmult
								dhObs += dhObs0/Nall*Nmult
								maghObs += maghObs0/Nall*Nmult
								rhObs += rhObs0/Nall*Nmult

								m1hObs0CDF, m1bCDF = np.histogram(obs["m1"], bins=mbinsCDF)
								m1hObs0CDFSmall, m1bCDFSmall = np.histogram(obs["m1"], bins=mbinsCDFSmall)
								qhObs0CDF, qbCDF = np.histogram(obs["m2"]/obs["m1"], bins=qbinsCDF)
								ehObs0CDF, ebCDF = np.histogram(obs["e"], bins=ebinsCDF)
								lphObs0CDF, lpbCDF = np.histogram(np.ma.log10(obs["p"].values).filled(-999), bins=lpbinsCDF)
								dhObs0CDF, dbCDF = np.histogram(obs["d"], bins=dbinsCDF)
								maghObs0CDF, magbCDF = np.histogram(obs["appMagMean_r"], bins=magbinsCDF)
								rhObs0CDF, rbCDF = np.histogram(obs["r2"]/obs["r1"], bins=rbinsCDF)
								m1hObsCDF += m1hObs0CDF/Nall*Nmult
								m1hObsCDFSmall += m1hObs0CDFSmall/Nall*Nmult
								qhObsCDF += qhObs0CDF/Nall*Nmult
								ehObsCDF += ehObs0CDF/Nall*Nmult
								lphObsCDF += lphObs0CDF/Nall*Nmult
								dhObsCDF += dhObs0CDF/Nall*Nmult
								maghObsCDF += maghObs0CDF/Nall*Nmult
								rhObsCDF += rhObs0CDF/Nall*Nmult

								# #save the individual histograms and CDFs
								# #Histograms
								# #y
								# self.outputHists['eachField'][OpSimID]['m1hObs'] = np.append(np.insert(m1hObs0/Nall*Nmult,0,0),0)
								# self.outputHists['eachField'][OpSimID]['m1SmallhObs'] = np.append(np.insert(m1hObs0Small/Nall,0,0),0)
								# self.outputHists['eachField'][OpSimID]['qhObs'] = np.append(np.insert(qhObs0/Nall*Nmult,0,0),0)
								# self.outputHists['eachField'][OpSimID]['ehObs'] = np.append(np.insert(ehObs0/Nall*Nmult,0,0),0)
								# self.outputHists['eachField'][OpSimID]['lphObs'] = np.append(np.insert(lphObs0/Nall*Nmult,0,0),0)
								# self.outputHists['eachField'][OpSimID]['dhObs'] = np.append(np.insert(dhObs0/Nall*Nmult,0,0),0)
								# self.outputHists['eachField'][OpSimID]['maghObs'] = np.append(np.insert(maghObs0/Nall*Nmult,0,0),0)
								# self.outputHists['eachField'][OpSimID]['rhObs'] = np.append(np.insert(rhObs0/Nall*Nmult,0,0),0)
								# #CDFs
								# #y
								# self.outputHists['eachField'][OpSimID]['m1hObsCDF'] = np.insert(m1hObs0CDF/Nall*Nmult,0,0)
								# self.outputHists['eachField'][OpSimID]['m1SmallhObsCDF'] = np.insert(m1hObs0CDFSmall/Nall*Nmult,0,0)
								# self.outputHists['eachField'][OpSimID]['qhObsCDF'] = np.insert(qhObs0CDF/Nall*Nmult,0,0)
								# self.outputHists['eachField'][OpSimID]['ehObsCDF'] = np.insert(ehObs0CDF/Nall*Nmult,0,0)
								# self.outputHists['eachField'][OpSimID]['lphObsCDF'] = np.insert(lphObs0CDF/Nall*Nmult,0,0)
								# self.outputHists['eachField'][OpSimID]['dhObsCDF'] = np.insert(dhObs0CDF/Nall*Nmult,0,0)
								# self.outputHists['eachField'][OpSimID]['maghObsCDF'] = np.insert(maghObs0CDF/Nall*Nmult,0,0)
								# self.outputHists['eachField'][OpSimID]['rhObsCDF'] = np.insert(rhObs0CDF/Nall*Nmult,0,0)

								#Rec
								recCombined = pd.DataFrame()
								prsaRecCombined = pd.DataFrame()
								DWDRecCombined = pd.DataFrame()

								# #histograms
								# self.outputHists['eachField'][OpSimID]['m1hRec'] = dict()
								# self.outputHists['eachField'][OpSimID]['m1SmallhRec'] = dict()
								# self.outputHists['eachField'][OpSimID]['qhRec'] = dict()
								# self.outputHists['eachField'][OpSimID]['ehRec'] = dict()
								# self.outputHists['eachField'][OpSimID]['lphRec'] = dict()
								# self.outputHists['eachField'][OpSimID]['dhRec'] = dict()
								# self.outputHists['eachField'][OpSimID]['maghRec'] = dict()
								# self.outputHists['eachField'][OpSimID]['rhRec'] = dict()
								# #CDFs
								# self.outputHists['eachField'][OpSimID]['m1hRecCDF'] = dict()
								# self.outputHists['eachField'][OpSimID]['m1SmallhRecCDF'] = dict()
								# self.outputHists['eachField'][OpSimID]['qhRecCDF'] = dict()
								# self.outputHists['eachField'][OpSimID]['ehRecCDF'] = dict()
								# self.outputHists['eachField'][OpSimID]['lphRecCDF'] = dict()
								# self.outputHists['eachField'][OpSimID]['dhRecCDF'] = dict()
								# self.outputHists['eachField'][OpSimID]['maghRecCDF'] = dict()
								# self.outputHists['eachField'][OpSimID]['rhRecCDF'] = dict()

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
										m1hRec0Small, m1bSmall = np.histogram(rec["m1"], bins=mbinsSmall)
										qhRec0, qb = np.histogram(rec["m2"]/rec["m1"], bins=qbins)
										ehRec0, eb = np.histogram(rec["e"], bins=ebins)
										lphRec0, lpb = np.histogram(np.ma.log10(rec["p"].values).filled(-999), bins=lpbins)
										dhRec0, db = np.histogram(rec["d"], bins=dbins)
										maghRec0, magb = np.histogram(rec["appMagMean_r"], bins=magbins)
										rhRec0, rb = np.histogram(rec["r2"]/rec["r1"], bins=rbins)
										m1hRec[filt] += m1hRec0/Nall*Nmult
										m1hRecSmall[filt] += m1hRec0Small/Nall*Nmult
										qhRec[filt] += qhRec0/Nall*Nmult
										ehRec[filt] += ehRec0/Nall*Nmult
										lphRec[filt] += lphRec0/Nall*Nmult
										dhRec[filt] += dhRec0/Nall*Nmult
										maghRec[filt] += maghRec0/Nall*Nmult
										rhRec[filt] += rhRec0/Nall*Nmult

										m1hRec0CDF, m1bCDF = np.histogram(rec["m1"], bins=mbinsCDF)
										m1hRec0CDFSmall, m1bCDFSmall = np.histogram(rec["m1"], bins=mbinsCDFSmall)
										qhRec0CDF, qbCDF = np.histogram(rec["m2"]/rec["m1"], bins=qbinsCDF)
										ehRec0CDF, ebCDF = np.histogram(rec["e"], bins=ebinsCDF)
										lphRec0CDF, lpbCDF = np.histogram(np.ma.log10(rec["p"].values).filled(-999), bins=lpbinsCDF)
										dhRec0CDF, dbCDF = np.histogram(rec["d"], bins=dbinsCDF)
										maghRec0CDF, magbCDF = np.histogram(rec["appMagMean_r"], bins=magbinsCDF)
										rhRec0CDF, rbCDF = np.histogram(rec["r2"]/rec["r1"], bins=rbinsCDF)
										m1hRecCDF[filt] += m1hRec0CDF/Nall*Nmult
										m1hRecCDFSmall[filt] += m1hRec0CDFSmall/Nall*Nmult
										qhRecCDF[filt] += qhRec0CDF/Nall*Nmult
										ehRecCDF[filt] += ehRec0CDF/Nall*Nmult
										lphRecCDF[filt] += lphRec0CDF/Nall*Nmult
										dhRecCDF[filt] += dhRec0CDF/Nall*Nmult
										maghRecCDF[filt] += maghRec0CDF/Nall*Nmult
										rhRecCDF[filt] += rhRec0CDF/Nall*Nmult

										# #Histograms
										# #y
										# self.outputHists['eachField'][OpSimID]['m1hRec'][filt] = np.append(np.insert(m1hRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['m1SmallhRec'][filt] = np.append(np.insert(m1hRec0Small/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['qhRec'][filt] = np.append(np.insert(qhRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['ehRec'][filt] = np.append(np.insert(ehRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['lphRec'][filt] = np.append(np.insert(lphRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['dhRec'][filt] = np.append(np.insert(dhRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['maghRec'][filt] = np.append(np.insert(maghRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['rhRec'][filt] = np.append(np.insert(rhRec0/Nall*Nmult,0,0),0)
										# #CDFs
										# #y
										# self.outputHists['eachField'][OpSimID]['m1hRecCDF'][filt] = np.append(np.insert(m1hRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['m1SmallhRecCDF'][filt] = np.append(np.insert(m1hRec0CDFSmall/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['qhRecCDF'][filt] = np.append(np.insert(qhRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['ehRecCDF'][filt] = np.append(np.insert(ehRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['lphRecCDF'][filt] = np.append(np.insert(lphRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['dhRecCDF'][filt] = np.append(np.insert(dhRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['maghRecCDF'][filt] = np.append(np.insert(maghRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['rhRecCDF'][filt] = np.append(np.insert(rhRec0CDF/Nall*Nmult,0,0),0)

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
										m1hRec0Small, m1bSmall = np.histogram(recCombined["m1"], bins=mbinsSmall)
										qhRec0, qb = np.histogram(recCombined["m2"]/rec["m1"], bins=qbins)
										ehRec0, eb = np.histogram(recCombined["e"], bins=ebins)
										lphRec0, lpb = np.histogram(np.ma.log10(recCombined["p"].values).filled(-999), bins=lpbins)
										dhRec0, db = np.histogram(recCombined["d"], bins=dbins)
										maghRec0, magb = np.histogram(recCombined["appMagMean_r"], bins=magbins)
										rhRec0, rb = np.histogram(recCombined["r2"]/recCombined["r1"], bins=rbins)
										m1hRec[filt] += m1hRec0/Nall*Nmult
										m1hRecSmall[filt] += m1hRec0Small/Nall*Nmult
										qhRec[filt] += qhRec0/Nall*Nmult
										ehRec[filt] += ehRec0/Nall*Nmult
										lphRec[filt] += lphRec0/Nall*Nmult
										dhRec[filt] += dhRec0/Nall*Nmult
										maghRec[filt] += maghRec0/Nall*Nmult
										rhRec[filt] += rhRec0/Nall*Nmult

										m1hRec0CDF, m1bCDF = np.histogram(recCombined["m1"], bins=mbinsCDF)
										m1hRec0CDFSmall, m1bCDFSmall = np.histogram(recCombined["m1"], bins=mbinsCDFSmall)
										qhRec0CDF, qbCDF = np.histogram(recCombined["m2"]/rec["m1"], bins=qbinsCDF)
										ehRec0CDF, ebCDF = np.histogram(recCombined["e"], bins=ebinsCDF)
										lphRec0CDF, lpbCDF = np.histogram(np.ma.log10(recCombined["p"].values).filled(-999), bins=lpbinsCDF)
										dhRec0CDF, dbCDF = np.histogram(recCombined["d"], bins=dbinsCDF)
										maghRec0CDF, magbCDF = np.histogram(recCombined["appMagMean_r"], bins=magbinsCDF)
										rhRec0CDF, rbCDF = np.histogram(recCombined["r2"]/recCombined["r1"], bins=rbinsCDF)
										m1hRecCDF[filt] += m1hRec0CDF/Nall*Nmult
										m1hRecCDFSmall[filt] += m1hRec0CDFSmall/Nall*Nmult
										qhRecCDF[filt] += qhRec0CDF/Nall*Nmult
										ehRecCDF[filt] += ehRec0CDF/Nall*Nmult
										lphRecCDF[filt] += lphRec0CDF/Nall*Nmult
										dhRecCDF[filt] += dhRec0CDF/Nall*Nmult
										maghRecCDF[filt] += maghRec0CDF/Nall*Nmult
										rhRecCDF[filt] += rhRec0CDF/Nall*Nmult

										# #Histograms
										# #y
										# self.outputHists['eachField'][OpSimID]['m1hRec'][filt] = np.append(np.insert(m1hRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['m1SmallhRec'][filt] = np.append(np.insert(m1hRec0Small/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['qhRec'][filt] = np.append(np.insert(qhRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['ehRec'][filt] = np.append(np.insert(ehRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['lphRec'][filt] = np.append(np.insert(lphRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['dhRec'][filt] = np.append(np.insert(dhRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['maghRec'][filt] = np.append(np.insert(maghRec0/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['rhRec'][filt] = np.append(np.insert(rhRec0/Nall*Nmult,0,0),0)
										# #CDFs
										# #y
										# self.outputHists['eachField'][OpSimID]['m1hRecCDF'][filt] = np.append(np.insert(m1hRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['m1SmallhRecCDF'][filt] = np.append(np.insert(m1hRec0CDFSmall/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['qhRecCDF'][filt] = np.append(np.insert(qhRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['ehRecCDF'][filt] = np.append(np.insert(ehRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['lphRecCDF'][filt] = np.append(np.insert(lphRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['dhRecCDF'][filt] = np.append(np.insert(dhRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['maghRecCDF'][filt] = np.append(np.insert(maghRec0CDF/Nall*Nmult,0,0),0)
										# self.outputHists['eachField'][OpSimID]['rhRecCDF'][filt] = np.append(np.insert(rhRec0CDF/Nall*Nmult,0,0),0)


								self.allRec = self.allRec.append(recCombined)

					rF = Nrec/Nall
					rN = Nrec/Nall*Nmult
					raN = NallMag/Nall*Nmult
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



				self.outputNumbers['OpSimID'].append(header['OpSimID'][0])
				self.outputNumbers['RA'].append(header['OpSimRA'][0])
				self.outputNumbers['Dec'].append(header['OpSimDec'][0])
				self.outputNumbers['NstarsReal'].append(NstarsReal)
				self.outputNumbers['fb'].append(fb)
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
		self.outputHists['m1Smallb'] = np.append(m1bSmall, m1bSmall[-1] + mbSizeSmall)
		self.outputHists['qb'] = np.append(qb, qb[-1] + qbSize)
		self.outputHists['eb'] = np.append(eb, eb[-1] + ebSize)
		self.outputHists['lpb'] = np.append(lpb, lpb[-1] + lpbSize)
		self.outputHists['db'] = np.append(db, db[-1] + dbSize)
		self.outputHists['magb'] = np.append(magb, magb[-1] + magbSize)
		self.outputHists['rb'] = np.append(rb, rb[-1] + rbSize)
		#All 
		self.outputHists['m1hAll'] = np.append(np.insert(m1hAll,0,0),0)
		self.outputHists['m1SmallhAll'] = np.append(np.insert(m1hAllSmall,0,0),0)
		self.outputHists['qhAll'] = np.append(np.insert(qhAll,0,0),0)
		self.outputHists['ehAll'] = np.append(np.insert(ehAll,0,0),0)
		self.outputHists['lphAll'] = np.append(np.insert(lphAll,0,0),0)
		self.outputHists['dhAll'] = np.append(np.insert(dhAll,0,0),0)
		self.outputHists['maghAll'] = np.append(np.insert(maghAll,0,0),0)
		self.outputHists['rhAll'] = np.append(np.insert(rhAll,0,0),0)
		#Observable
		self.outputHists['m1hObs'] = np.append(np.insert(m1hObs,0,0),0)
		self.outputHists['m1SmallhObs'] = np.append(np.insert(m1hObsSmall,0,0),0)
		self.outputHists['qhObs'] = np.append(np.insert(qhObs,0,0),0)
		self.outputHists['ehObs'] = np.append(np.insert(ehObs,0,0),0)
		self.outputHists['lphObs'] = np.append(np.insert(lphObs,0,0),0)
		self.outputHists['dhObs'] = np.append(np.insert(dhObs,0,0),0)
		self.outputHists['maghObs'] = np.append(np.insert(maghObs,0,0),0)
		self.outputHists['rhObs'] = np.append(np.insert(rhObs,0,0),0)
		#Recovered
		self.outputHists['m1hRec'] = dict()
		self.outputHists['m1SmallhRec'] = dict()
		self.outputHists['qhRec'] = dict()
		self.outputHists['ehRec'] = dict()
		self.outputHists['lphRec'] = dict()
		self.outputHists['dhRec'] = dict()
		self.outputHists['maghRec'] = dict()
		self.outputHists['rhRec'] = dict()
		for f in self.filters:
			self.outputHists['m1hRec'][f] = np.append(np.insert(m1hRec[f],0,0),0)
			self.outputHists['m1SmallhRec'][f] = np.append(np.insert(m1hRecSmall[f],0,0),0)
			self.outputHists['qhRec'][f] = np.append(np.insert(qhRec[f],0,0),0)
			self.outputHists['ehRec'][f] = np.append(np.insert(ehRec[f],0,0),0)
			self.outputHists['lphRec'][f] = np.append(np.insert(lphRec[f],0,0),0)
			self.outputHists['dhRec'][f] = np.append(np.insert(dhRec[f],0,0),0)
			self.outputHists['maghRec'][f] = np.append(np.insert(maghRec[f],0,0),0)
			self.outputHists['rhRec'][f] = np.append(np.insert(rhRec[f],0,0),0)

		self.outputHists['m1bCDF'] = m1bCDF
		self.outputHists['m1SmallbCDF'] = m1bCDFSmall
		self.outputHists['qbCDF'] = qbCDF
		self.outputHists['ebCDF'] = ebCDF
		self.outputHists['lpbCDF'] = lpbCDF
		self.outputHists['dbCDF'] = dbCDF
		self.outputHists['magbCDF'] = magbCDF
		self.outputHists['rbCDF'] = rbCDF
		#All (inserting zeros at the start so that I can more easily plot these with the bin_edges)
		self.outputHists['m1hAllCDF'] = np.insert(m1hAllCDF,0,0)
		self.outputHists['m1SmallhAllCDF'] = np.insert(m1hAllCDFSmall,0,0)
		self.outputHists['qhAllCDF'] = np.insert(qhAllCDF,0,0)
		self.outputHists['ehAllCDF'] = np.insert(ehAllCDF,0,0)
		self.outputHists['lphAllCDF'] = np.insert(lphAllCDF,0,0)
		self.outputHists['dhAllCDF'] = np.insert(dhAllCDF,0,0)
		self.outputHists['maghAllCDF'] = np.insert(maghAllCDF,0,0)
		self.outputHists['rhAllCDF'] = np.insert(rhAllCDF,0,0)
		#Observable
		self.outputHists['m1hObsCDF'] = np.insert(m1hObsCDF,0,0)
		self.outputHists['m1SmallhObsCDF'] = np.insert(m1hObsCDFSmall,0,0)
		self.outputHists['qhObsCDF'] = np.insert(qhObsCDF,0,0)
		self.outputHists['ehObsCDF'] = np.insert(ehObsCDF,0,0)
		self.outputHists['lphObsCDF'] = np.insert(lphObsCDF,0,0)
		self.outputHists['dhObsCDF'] = np.insert(dhObsCDF,0,0)
		self.outputHists['maghObsCDF'] = np.insert(maghObsCDF,0,0)
		self.outputHists['rhObsCDF'] = np.insert(rhObsCDF,0,0)
		#Recovered
		self.outputHists['m1hRecCDF'] = dict()
		self.outputHists['m1SmallhRecCDF'] = dict()
		self.outputHists['qhRecCDF'] = dict()
		self.outputHists['ehRecCDF'] = dict()
		self.outputHists['lphRecCDF'] = dict()
		self.outputHists['dhRecCDF'] = dict()
		self.outputHists['maghRecCDF'] = dict()
		self.outputHists['rhRecCDF'] = dict()
		for f in self.filters:
			self.outputHists['m1hRecCDF'][f] = np.insert(m1hRecCDF[f],0,0)
			self.outputHists['m1SmallhRecCDF'][f] = np.insert(m1hRecCDFSmall[f],0,0)
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

		#I want to show the RA with the origin at the side, so that the plots look like the figures here http://astro-lsst-01.astro.washington.edu:8080/allMetricResults?runId=1
		#this requires me to use cartopy, which doesn't label the axes, so I have to go through a LOT of trouble to add axes labels!
		#but this appears to work well enough, despite a few "magic numbers"
		def makeMollweideAxes():    
			#set up the projections
			proj = ccrs.Mollweide(central_longitude=180)
			data_proj = ccrs.PlateCarree()#ccrs.Geodetic()

			#create the plot
			f,ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection':proj})
			ax.set_global()

			#for the labels
			xlocs = np.linspace(-180, 180, 13)
			ylocs = np.linspace(-90, 90, 13)
			ax.gridlines(xlocs=xlocs, ylocs=ylocs)

			#labels
			plt.gcf().text(0.51, 0.15, 'RA', fontsize=16, horizontalalignment='center')
			plt.gcf().text(0.045, 0.5, 'Dec', fontsize=16, rotation=90, verticalalignment='center')
			for x in xlocs[1:-1]:
				l = r'$'+str(int(x))+'^\degree$'
				#plt.gcf().text(x/360. + 0.04 , 0.51, l, fontsize=12, horizontalalignment='center')
				ax.text(x, 1, l, fontsize=12, horizontalalignment='center', transform=data_proj)

			#it seems like there should be a better way to do this!
			bbox = ax.dataLim
			for y in ylocs[1:-1]:
				loc = proj.transform_point(0, y,data_proj)
				xval = (loc[0]*0.75 - bbox.x0)/(bbox.x1 - bbox.x0)   
				yval = (loc[1]*0.65 - bbox.y0)/(bbox.y1 - bbox.y0)   
				tval = y
				if (tval < 0):
					tval += 360 
				l = r'$'+str(int(tval))+'^\degree$'
				xoff = -0.015
				if (y < 0):
					xoff = -0.03
				plt.gcf().text(xval + xoff, yval, l, verticalalignment='center', horizontalalignment='center')

			return f, ax, data_proj


		#make the mollweide
		use = d.loc[(d['recFrac'] > 0)]
		coords = SkyCoord(use['RA'], use['Dec'], unit=(units.degree, units.degree),frame='icrs')	
		RAwrap = coords.ra.wrap_at(360.*units.degree).degree
		Decwrap = coords.dec.wrap_at(90.*units.degree).degree

		f, ax, data_proj = makeMollweideAxes()

		mlw = ax.scatter(np.array(RAwrap).ravel(), np.array(Decwrap).ravel(), c=np.array(use['recN']/use['obsN']), cmap='magma_r', s = 10, vmin=0.3, vmax=0.7, transform=data_proj)
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
		RAwrap = coords.ra.wrap_at(360.*units.degree).degree
		Decwrap = coords.dec.wrap_at(90.*units.degree).degree

		f, ax, data_proj = makeMollweideAxes()
		mlw = ax.scatter(np.array(RAwrap).ravel(), np.array(Decwrap).ravel(), c=np.log10(np.array(d['recN'])), cmap='magma_r', s = 10, vmin=0, vmax=4.5, transform=data_proj)
		if (showCbar):
			#cbar = f.colorbar(mlw, shrink=0.7)
			cbaxes = f.add_axes([0.1, 0.9, 0.8, 0.03]) 
			cbar = plt.colorbar(mlw, cax = cbaxes, orientation="horizontal") 
			cbar.set_label(r'$\log_{10}(N_\mathrm{Rec.})$',fontsize=16)
			cbaxes.xaxis.set_ticks_position('top')
			cbaxes.xaxis.set_label_position('top')
		f.savefig(os.path.join(self.plotsDirectory,'mollweide_N'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)


	def makeMollweidesDiff(self, d1, d2, suffix='', showCbar=True):

		#I want to show the RA with the origin at the side, so that the plots look like the figures here http://astro-lsst-01.astro.washington.edu:8080/allMetricResults?runId=1
		#this requires me to use cartopy, which doesn't label the axes, so I have to go through a LOT of trouble to add axes labels!
		#but this appears to work well enough, despite a few "magic numbers"
		def makeMollweideAxes():    
			#set up the projections
			proj = ccrs.Mollweide(central_longitude=180)
			data_proj = ccrs.PlateCarree()#ccrs.Geodetic()

			#create the plot
			f,ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection':proj})
			ax.set_global()

			#for the labels
			xlocs = np.linspace(-180, 180, 13)
			ylocs = np.linspace(-90, 90, 13)
			ax.gridlines(xlocs=xlocs, ylocs=ylocs)

			#labels
			plt.gcf().text(0.51, 0.15, 'RA', fontsize=16, horizontalalignment='center')
			plt.gcf().text(0.045, 0.5, 'Dec', fontsize=16, rotation=90, verticalalignment='center')
			for x in xlocs[1:-1]:
				l = r'$'+str(int(x))+'^\degree$'
				#plt.gcf().text(x/360. + 0.04 , 0.51, l, fontsize=12, horizontalalignment='center')
				ax.text(x, 1, l, fontsize=12, horizontalalignment='center', transform=data_proj)

			#it seems like there should be a better way to do this!
			bbox = ax.dataLim
			for y in ylocs[1:-1]:
				loc = proj.transform_point(0, y,data_proj)
				xval = (loc[0]*0.75 - bbox.x0)/(bbox.x1 - bbox.x0)   
				yval = (loc[1]*0.65 - bbox.y0)/(bbox.y1 - bbox.y0)   
				tval = y
				if (tval < 0):
					tval += 360 
				l = r'$'+str(int(tval))+'^\degree$'
				xoff = -0.015
				if (y < 0):
					xoff = -0.03
				plt.gcf().text(xval + xoff, yval, l, verticalalignment='center', horizontalalignment='center')

			return f, ax, data_proj


		#make the mollweide
		coords = SkyCoord(d1['RA'], d1['Dec'], unit=(units.degree, units.degree),frame='icrs')	
		RAwrap = coords.ra.wrap_at(360.*units.degree).degree
		Decwrap = coords.dec.wrap_at(90.*units.degree).degree

		f, ax, data_proj = makeMollweideAxes()
		r1 = np.array(d1['recN']/d1['obsN'])
		check = np.isnan(r1)
		r1[check] = 0.
		r2 = np.array(d2['recN']/d2['obsN'])
		check = np.isnan(r2)
		r2[check] = 0.
		mlw = ax.scatter(np.array(RAwrap).ravel(), np.array(Decwrap).ravel(), c=r1-r2, cmap='seismic_r', s = 10, vmin=-1, vmax=1, transform=data_proj)
		if (showCbar):
			#cbar = f.colorbar(mlw, shrink=0.7)
			# Now adding the colorbar
			cbaxes = f.add_axes([0.1, 0.9, 0.8, 0.03]) 
			cbar = plt.colorbar(mlw, cax = cbaxes, orientation="horizontal") 
			cbar.set_label(r'$\left(N_\mathrm{Rec.}/N_\mathrm{Obs.}\right)_\mathrm{colossus} - \left(N_\mathrm{Rec.}/N_\mathrm{Obs.}\right)_\mathrm{baseline}$',fontsize=16)
			cbaxes.xaxis.set_ticks_position('top')
			cbaxes.xaxis.set_label_position('top')

		f.savefig(os.path.join(self.plotsDirectory,'mollweide_pct'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)

		coords = SkyCoord(d1['RA'], d1['Dec'], unit=(units.degree, units.degree),frame='icrs')	
		RAwrap = coords.ra.wrap_at(360.*units.degree).degree
		Decwrap = coords.dec.wrap_at(90.*units.degree).degree

		f, ax, data_proj = makeMollweideAxes()
		mlw = ax.scatter(np.array(RAwrap).ravel(), np.array(Decwrap).ravel(), c=(np.array(d1['recN']) - np.array(d2['recN'])), cmap='seismic_r', s = 10, vmin=-2000, vmax=2000, transform=data_proj)
		if (showCbar):
			#cbar = f.colorbar(mlw, shrink=0.7)
			cbaxes = f.add_axes([0.1, 0.9, 0.8, 0.03]) 
			cbar = plt.colorbar(mlw, cax = cbaxes, orientation="horizontal", extend='both') 
			cbar.set_label(r'$N_\mathrm{Rec.,colossus} - N_\mathrm{Rec.,baseline}$',fontsize=16)
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
		if (self.oneGiant):
			suffix = '_oneGiant'
		if (self.twoGiants):
			suffix = '_twoGiants'
		if (self.noGiants):
			suffix = '_noGiants'


		#print summary stats
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
		df.to_csv(os.path.join(self.plotsDirectory,'numbers'+suffix+'.csv'), index=False)
		pickle.dump(self.outputHists, open( os.path.join(self.plotsDirectory,'outputHists'+suffix+'.pickle'), "wb"))

		pickle.dump(self.allRec, open( os.path.join(self.plotsDirectory,'allRec'+suffix+'.pickle'), "wb"))

		#now make all the plots

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




	def plotAllObsRecOtherRatio(self, d1, d2, m1key='m1'):
		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'
		if (self.oneGiant):
			suffix = '_oneGiant'
		if (self.twoGiants):
			suffix = '_twoGiants'
		if (self.noGiants):
			suffix = '_noGiants'	

		self.plotObsRecOtherRatio(d1, d2, m1key, 'm1 (Msolar)', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=self.m1xlim)
		self.plotObsRecOtherRatio(d1, d2, 'q', 'q (m2/m1)', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1])
		self.plotObsRecOtherRatio(d1, d2, 'e', 'e', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1])
		self.plotObsRecOtherRatio(d1, d2, 'lp', 'log(P [days])', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5])
		self.plotObsRecOtherRatio(d1, d2, 'd', 'd (kpc)', os.path.join(self.plotsDirectory,'EBLSST_dhist'+suffix), xlim=[0,25])
		self.plotObsRecOtherRatio(d1, d2, 'mag', 'mag', os.path.join(self.plotsDirectory,'EBLSST_maghist'+suffix), xlim=[12, 25])
		self.plotObsRecOtherRatio(d1, d2, 'r', 'r2/r1', os.path.join(self.plotsDirectory,'EBLSST_rhist'+suffix), xlim=[0,3])


		m1xlim = self.m1xlim
		#m1xlim[1] -= 0.01
		#f,ax = plt.subplots(3,5,figsize=(24, 12))
		f,ax = plt.subplots(3,4,figsize=(20, 12))
		self.plotObsRecOtherRatio(d1, d2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5], ax=ax[:,0], showLegend=True, legendLoc = 'lower right')
		self.plotObsRecOtherRatio(d1, d2, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1], ax=ax[:,1], showLegend=False)
		self.plotObsRecOtherRatio(d1, d2, m1key, r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=m1xlim, ax=ax[:,2], showLegend=False)
		self.plotObsRecOtherRatio(d1, d2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1], ax=ax[:,3], showLegend=False)
		#self.plotObsRecOtherRatio(d1, d2, 'mag', r'$r\_$ [mag]', os.path.join(self.plotsDirectory,'EBLSST_maghist'+suffix), xlim=[12,25], ax=ax[:,4], showLegend=False)
		ax[0,0].set_ylabel('CDF', fontsize=18)
		ax[1,0].set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=18)
		#ax[1,0].set_ylabel('PDF', fontsize=16)
		ax[2,0].set_ylabel('Ratio', fontsize=18)
		for i in range(3):
			for j in range(4):
				if (i != 2):
					ax[i,j].set_xticklabels([])
				if (j != 0):
					ax[i,j].set_yticklabels([])

		f.subplots_adjust(hspace=0, wspace=0.1)
		f.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherRatioCombined'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)


	def plotAllObsRecOtherRatio_new(self, d1, d2, m1key='m1'):
		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'
		if (self.oneGiant):
			suffix = '_oneGiant'
		if (self.twoGiants):
			suffix = '_twoGiants'
		if (self.noGiants):
			suffix = '_noGiants'

		m1xlim = self.m1xlim
		f,ax = plt.subplots(2,4,figsize=(20, 8))
		self.plotObsRecCDFOther_new(d1, d2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,5], ax=ax[:,0], showLegend=True, legendLoc = 'lower right')
		self.plotObsRecCDFOther_new(d1, d2, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1], ax=ax[:,1], showLegend=False)
		self.plotObsRecCDFOther_new(d1, d2, m1key, r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=m1xlim, ax=ax[:,2], showLegend=False)
		self.plotObsRecCDFOther_new(d1, d2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax=ax[:,3], showLegend=False)
		ax[0,0].set_ylabel('CDF', fontsize=18)
		#ax[1,0].set_ylabel(r'$N_\mathrm{norm} = N/\sum N_\mathrm{baseline}$', fontsize=18)
		ax[1,0].set_ylabel(r'$N/\sum N_\mathrm{baseline}$', fontsize=18)
		for i in range(2):
			for j in range(4):
				if (i != 2):
					ax[i,j].set_xticklabels([])
				if (j != 0):
					ax[i,j].set_yticklabels([])
	

		f.subplots_adjust(hspace=0, wspace=0.1)
		f.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherCombined_new'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)

		f2,ax2 = plt.subplots(1,4,figsize=(20, 4))
		self.plotObsRecOtherRatio_new(d1, d2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,5], ax2=ax2[0],showLegend=True, legendLoc = 'upper right')
		self.plotObsRecOtherRatio_new(d1, d2, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1],  ax2=ax2[1], showLegend=False)
		self.plotObsRecOtherRatio_new(d1, d2, m1key, r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=m1xlim, ax2=ax2[2], showLegend=False)
		self.plotObsRecOtherRatio_new(d1, d2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax2=ax2[3],showLegend=False)
		ax2[0].set_ylabel('Ratio', fontsize=18)
		for j in range(4):
			if (j != 0):
				ax2[j].set_yticklabels([])
		f2.subplots_adjust(hspace=0, wspace=0.1)
		f2.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherRatioCombined_new'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f2)


	def plotAllObsRecOtherRatioCombined_new(self, dF1, dF2, dGC1, dGC2, dOC1, dOC2):
		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'
		if (self.oneGiant):
			suffix = '_oneGiant'
		if (self.twoGiants):
			suffix = '_twoGiants'
		if (self.noGiants):
			suffix = '_noGiants'

		f,ax = plt.subplots(3,4,figsize=(20, 12))

		#histograms
		#ASAS-SN DATA
		lpbins = dF1['lpb']
		df = pd.read_csv('/Users/ageller/WORK/EBdata/asassn-catalog.csv')
		EB = df.loc[(df['Type'] == 'EA')]# | (df['Type'] == 'EB') | (df['Type'] == 'EW') ]
		h, b = np.histogram(np.log10(EB['period']), bins=lpbins)
		ax[0][0].step(b[0:-1],h/np.sum(h), color='#5FC8D0', linewidth=1)

		self.plotObsRecOther_new(dF1, dF2, 'lp', '', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,4], ax=ax[0,0],showLegend=True)
		self.plotObsRecOther_new(dF1, dF2, 'e', '', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1],  ax=ax[0,1], showLegend=False)
		self.plotObsRecOther_new(dF1, dF2, 'm1', '', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=[0,10], ax=ax[0,2], showLegend=False)
		self.plotObsRecOther_new(dF1, dF2, 'q', '', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax=ax[0,3],showLegend=False)

		self.plotObsRecOther_new(dGC1, dGC2, 'lp', '', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,4], ax=ax[1,0],showLegend=False)
		self.plotObsRecOther_new(dGC1, dGC2, 'e', '', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1],  ax=ax[1,1], showLegend=False)
		self.plotObsRecOther_new(dGC1, dGC2, 'm1Small', '', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=[0,3], ax=ax[1,2], showLegend=False)
		self.plotObsRecOther_new(dGC1, dGC2, 'q', '', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax=ax[1,3],showLegend=False)

		self.plotObsRecOther_new(dOC1, dOC2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,4], ax=ax[2,0],showLegend=False)
		self.plotObsRecOther_new(dOC1, dOC2, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1],  ax=ax[2,1], showLegend=False)
		self.plotObsRecOther_new(dOC1, dOC2, 'm1', r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=[0,10], ax=ax[2,2], showLegend=False)
		self.plotObsRecOther_new(dOC1, dOC2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax=ax[2,3],showLegend=False)

		# ax[0,0].set_ylabel(r'$N_\mathrm{field}/\sum N_\mathrm{baseline,field}$', fontsize=18)
		# ax[1,0].set_ylabel(r'$N_\mathrm{GC}/\sum N_\mathrm{baseline,GC}$', fontsize=18)
		# ax[2,0].set_ylabel(r'$N_\mathrm{OC}/\sum N_\mathrm{baseline,OC}$', fontsize=18)
		ax[0,0].set_ylabel(r'$N_\mathrm{norm, field}$', fontsize=18)
		ax[1,0].set_ylabel(r'$N_\mathrm{norm,GC}$', fontsize=18)
		ax[2,0].set_ylabel(r'$N_\mathrm{norm,OC}$', fontsize=18)
		for i in range(3):
			for j in range(4):
				if (j != 0):
					ax[i,j].set_yticklabels([])


		f.subplots_adjust(hspace=0.3, wspace=0.1)
		f.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherCombined_new'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)

		#CDFs
		f1,ax1 = plt.subplots(3,4,figsize=(20, 12))

		#ASAS-SN DATA
		lpbSize = 0.25
		CDFfac = 1000.
		lpbinsCDF = np.arange(-3, 10+lpbSize/CDFfac, lpbSize/CDFfac, dtype='float')
		h, b = np.histogram(np.log10(EB['period']), bins=lpbinsCDF)
		cdf = []
		for i in range(len(h)):
			cdf.append(np.sum(h[:i])/np.sum(h))
		ax1[0][0].step(b[0:-1],cdf, color='#5FC8D0', linewidth=1)

		#ax1[0][0].hist(np.log10(EB['period']), bins=lpbinsCDF, color='#5FC8D0', linewidth=2, cumulative=True, histtype='step',density=True)

		self.plotObsRecCDF_new(dF1, dF2, 'lp', '', os.path.join(self.plotsDirectory,'EBLSST_lphistCDF_new'+suffix), xlim=[-2,4], ax=ax1[0,0],showLegend=True)
		self.plotObsRecCDF_new(dF1, dF2, 'e', '', os.path.join(self.plotsDirectory,'EBLSST_ehistCDF_new'+suffix), xlim=[0,1],  ax=ax1[0,1], showLegend=False)
		self.plotObsRecCDF_new(dF1, dF2, 'm1', '', os.path.join(self.plotsDirectory,'EBLSST_m1histCDF_new'+suffix), xlim=[0,10], ax=ax1[0,2], showLegend=False)
		self.plotObsRecCDF_new(dF1, dF2, 'q', '', os.path.join(self.plotsDirectory,'EBLSST_qhistCDF_new'+suffix), xlim=[0,1], ax=ax1[0,3],showLegend=False)

		self.plotObsRecCDF_new(dGC1, dGC2, 'lp', '', os.path.join(self.plotsDirectory,'EBLSST_lphistCDF_new'+suffix), xlim=[-2,4], ax=ax1[1,0],showLegend=False)
		self.plotObsRecCDF_new(dGC1, dGC2, 'e', '', os.path.join(self.plotsDirectory,'EBLSST_ehistCDF_new'+suffix), xlim=[0,1],  ax=ax1[1,1], showLegend=False)
		self.plotObsRecCDF_new(dGC1, dGC2, 'm1Small', '', os.path.join(self.plotsDirectory,'EBLSST_m1histCDF_new'+suffix), xlim=[0,3], ax=ax1[1,2], showLegend=False)
		self.plotObsRecCDF_new(dGC1, dGC2, 'q', '', os.path.join(self.plotsDirectory,'EBLSST_qhistCDF_new'+suffix), xlim=[0,1], ax=ax1[1,3],showLegend=False)

		self.plotObsRecCDF_new(dOC1, dOC2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphistCDF_new'+suffix), xlim=[-2,4], ax=ax1[2,0],showLegend=False)
		self.plotObsRecCDF_new(dOC1, dOC2, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehistCDF_new'+suffix), xlim=[0,1],  ax=ax1[2,1], showLegend=False)
		self.plotObsRecCDF_new(dOC1, dOC2, 'm1', r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1histCDF_new'+suffix), xlim=[0,10], ax=ax1[2,2], showLegend=False)
		self.plotObsRecCDF_new(dOC1, dOC2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhistCDF_new'+suffix), xlim=[0,1], ax=ax1[2,3],showLegend=False)

		# ax[0,0].set_ylabel(r'$N_\mathrm{field}/\sum N_\mathrm{baseline,field}$', fontsize=18)
		# ax[1,0].set_ylabel(r'$N_\mathrm{GC}/\sum N_\mathrm{baseline,GC}$', fontsize=18)
		# ax[2,0].set_ylabel(r'$N_\mathrm{OC}/\sum N_\mathrm{baseline,OC}$', fontsize=18)
		ax1[0,0].set_ylabel(r'CDF$_\mathrm{field}$', fontsize=18)
		ax1[1,0].set_ylabel(r'CDF$_\mathrm{GC}$', fontsize=18)
		ax1[2,0].set_ylabel(r'CDF$_\mathrm{OC}$', fontsize=18)
		for i in range(3):
			for j in range(4):
				if (j != 0):
					ax1[i,j].set_yticklabels([])


		f1.subplots_adjust(hspace=0.3, wspace=0.1)
		f1.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecCDFCombined_new'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f1)

		#ratios
		f2,ax2 = plt.subplots(3,4,figsize=(20, 12))
		self.plotObsRecOtherRatio_new(dF1, dF2, 'lp', '', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,4], ax2=ax2[0,0],showLegend=True, legendLoc = 'upper right')
		self.plotObsRecOtherRatio_new(dF1, dF2, 'e', '', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1],  ax2=ax2[0,1], showLegend=False)
		self.plotObsRecOtherRatio_new(dF1, dF2, 'm1', '', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=[0,10], ax2=ax2[0,2], showLegend=False)
		self.plotObsRecOtherRatio_new(dF1, dF2, 'q', '', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax2=ax2[0,3],showLegend=False)

		self.plotObsRecOtherRatio_new(dGC1, dGC2, 'lp', '', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,4], ax2=ax2[1,0],showLegend=False)
		self.plotObsRecOtherRatio_new(dGC1, dGC2, 'e', '', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1],  ax2=ax2[1,1], showLegend=False)
		self.plotObsRecOtherRatio_new(dGC1, dGC2, 'm1Small', '', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=[0,3], ax2=ax2[1,2], showLegend=False)
		self.plotObsRecOtherRatio_new(dGC1, dGC2, 'q', '', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax2=ax2[1,3],showLegend=False)

		self.plotObsRecOtherRatio_new(dOC1, dOC2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,4], ax2=ax2[2,0],showLegend=False)
		self.plotObsRecOtherRatio_new(dOC1, dOC2, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1],  ax2=ax2[2,1], showLegend=False)
		self.plotObsRecOtherRatio_new(dOC1, dOC2, 'm1', r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=[0,10], ax2=ax2[2,2], showLegend=False)
		self.plotObsRecOtherRatio_new(dOC1, dOC2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax2=ax2[2,3],showLegend=False)

		ax2[0,0].set_ylabel(r'Ratio$_\mathrm{field}$', fontsize=18)
		ax2[1,0].set_ylabel(r'Ratio$_\mathrm{GC}$', fontsize=18)
		ax2[2,0].set_ylabel(r'Ratio$_\mathrm{OC}$', fontsize=18)
		for i in range(3):
			for j in range(4):
				if (j != 0):
					ax2[i,j].set_yticklabels([])

		f2.subplots_adjust(hspace=0.3, wspace=0.1)
		f2.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherRatioCombined_new'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f2)


	def plotAllObsRecOther_clusters_new(self, d1, d2, df1, df2, m1key='m1',tkey='GC'):
		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'
		if (self.oneGiant):
			suffix = '_oneGiant'
		if (self.twoGiants):
			suffix = '_twoGiants'
		if (self.noGiants):
			suffix = '_noGiants'	


		m1xlim = self.m1xlim
		f,ax = plt.subplots(1,4,figsize=(20, 4))
		self.plotObsRecOther_clusters_new(d1, d2, df1, df2, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist_new'+suffix), xlim=[-2,5], ax=ax[0],showLegend=True, legendLoc = 'upper right')
		self.plotObsRecOther_clusters_new(d1, d2, df1, df2, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehist_new'+suffix), xlim=[0,1], ax=ax[1], showLegend=False)
		self.plotObsRecOther_clusters_new(d1, d2, df1, df2, m1key, r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1hist_new'+suffix), xlim=m1xlim, ax=ax[2], showLegend=False)
		self.plotObsRecOther_clusters_new(d1, d2, df1, df2, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist_new'+suffix), xlim=[0,1], ax=ax[3],showLegend=False)
		ax[0].set_ylabel(r'$N_\mathrm{norm,'+tkey+'} / N_\mathrm{norm,field}$', fontsize=18)
		for j in range(4):
			if (j != 0):
				ax[j].set_yticklabels([])


		f.subplots_adjust(hspace=0, wspace=0.1)
		f.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherCombined_clusters_new'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
		plt.close(f)


	def plotAllObsRecOtherPDF(self, d1, d1C, d2, d2C, d3, d3C, c1=None, c2=None, c3=None, includeASASSN=True):

		suffix = ''
		if (self.onlyDWD):
			suffix = '_DWD'
		if (self.oneGiant):
			suffix = '_oneGiant'
		if (self.twoGiants):
			suffix = '_twoGiants'
		if (self.noGiants):
			suffix = '_noGiants'

		m1xlim = self.m1xlim
		#m1xlim[1] -= 0.01
		#f,ax = plt.subplots(1,5,figsize=(25, 4), sharey=True)
		f,ax = plt.subplots(2,4,figsize=(20, 8))
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'lp', r'$\log_{10}(P$ [days]$)$', os.path.join(self.plotsDirectory,'EBLSST_lphist'+suffix), xlim=[-2,5], ax1=ax[0][0], ax2=ax[1][0], showLegend=True, legendLoc = 'upper right',includeASASSN=includeASASSN, c1=c1, c2=c2, c3=c3)
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'e', r'$e$', os.path.join(self.plotsDirectory,'EBLSST_ehist'+suffix), xlim=[0,1], ax1=ax[0][1], ax2=ax[1][1], showLegend=False, c1=c1, c2=c2, c3=c3)
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'm1', r'$m_1$ [M$_\odot$]', os.path.join(self.plotsDirectory,'EBLSST_m1hist'+suffix), xlim=m1xlim, ax1=ax[0][2], ax2=ax[1][2], showLegend=False,  c1=c1, c2=c2, c3=c3)
		self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'q', r'$q$ $(m_2/m_1)$', os.path.join(self.plotsDirectory,'EBLSST_qhist'+suffix), xlim=[0,1], ax1=ax[0][3], ax2=ax[1][3], showLegend=False,  c1=c1, c2=c2, c3=c3)
		#self.plotObsRecOtherPDF(d1, d1C, d2, d2C, d3, d3C, 'mag', r'$r\_$ [mag]', os.path.join(self.plotsDirectory,'EBLSST_maghist'+suffix), xlim=[12,25], ax=ax[4], showLegend=False)
		#ax[0].set_ylabel(r'$\frac{N_i}{\sum_i N_i}$', fontsize=16)
		#ax[0].set_ylabel('PDF', fontsize=16)
		ax[0][0].set_ylabel(r'$N_\mathrm{Rec.}$', fontsize=18)
		ax[1][0].set_ylabel(r'$N_\mathrm{Rec.}/N_\mathrm{Obs.}$', fontsize=18)

		if (includeASASSN):
			lpbins = d1['lpb']
			df = pd.read_csv('/Users/ageller/WORK/EBdata/asassn-catalog.csv')
			EB = df.loc[(df['Type'] == 'EA')]# | (df['Type'] == 'EB') | (df['Type'] == 'EW') ]
			ax[0][0].hist(np.log10(EB['period']), bins=lpbins, color='lightgray', histtype='step', linewidth=2)

		for i in range(4):
			ax[0][i].xaxis.set_ticklabels([])
			if (i > 0):
				ax[0][i].yaxis.set_ticklabels([])
				ax[1][i].yaxis.set_ticklabels([])

		f.subplots_adjust(wspace=0.1, hspace=0)
		f.savefig(os.path.join(self.plotsDirectory,'EBLSST_ObsRecOtherPDFCombined'+suffix+'.pdf'),format='pdf', bbox_inches = 'tight')
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



