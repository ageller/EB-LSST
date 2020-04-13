import pandas as pd
import numpy as np

#for Quest
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.lines as mlines

def plotHist(histAll, histObs, allhistRec, bin_edges, histAllOD, histObsOD, allhistRecOD, bin_edgesOD, xlim, xtitle, fname, legendLoc='lower right'):
	c1 = '#0294A5'  #turqoise
	c2 = '#d95f02' #orange from color brewer
	c3 = '#00353E' #slate
	c4 = '#508201' #olive

	f,(ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5, 12), sharex=True)

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
	ax1.set_xlim(xlim[0],xlim[1])

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
	ax2.set_xlim(xlim[0],xlim[1])


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
	ax3.set_xlim(xlim[0],xlim[1])

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
	f.savefig(fname+'_ratio.pdf',format='pdf', bbox_inches = 'tight')



if __name__ == "__main__":

	print("period")
	data = pd.read_csv('EBLSST_lphist.csv')
	dataOD = pd.read_csv('colossus/EBLSST_lphist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, \
			dataOD['histAll'].values, dataOD['histObs'].values, dataOD['allhistRec'].values, dataOD['binEdges'].values, \
			[-2, 6], 'log(Period [days])', 'EBLSST_lphist_final')

	print("eccentricity")
	data = pd.read_csv('EBLSST_ehist.csv')
	dataOD = pd.read_csv('colossus/EBLSST_ehist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, \
			dataOD['histAll'].values, dataOD['histObs'].values, dataOD['allhistRec'].values, dataOD['binEdges'].values, \
			[0, 1], 'Eccentricity', 'EBLSST_ehist_final')

	print("distance")
	data = pd.read_csv('EBLSST_dhist.csv')
	dataOD = pd.read_csv('colossus/EBLSST_dhist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, \
			dataOD['histAll'].values, dataOD['histObs'].values, dataOD['allhistRec'].values, dataOD['binEdges'].values, \
			[0, 25], 'Distance (kpc)', 'EBLSST_dhist_final')

	print("mass1")
	data = pd.read_csv('EBLSST_m1hist.csv')
	dataOD = pd.read_csv('colossus/EBLSST_m1hist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, \
			dataOD['histAll'].values, dataOD['histObs'].values, dataOD['allhistRec'].values, dataOD['binEdges'].values, \
			[0, 3], r'm$_1$ (M$_\odot$)', 'EBLSST_m1hist_final')

	print("mass ratio")
	data = pd.read_csv('EBLSST_qhist.csv')
	dataOD = pd.read_csv('colossus/EBLSST_qhist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, \
			dataOD['histAll'].values, dataOD['histObs'].values, dataOD['allhistRec'].values, dataOD['binEdges'].values, \
			[0, 1], r'q (m$_2$/m$_1$)', 'EBLSST_qhist_final')

	print("magnitude")
	data = pd.read_csv('EBLSST_maghist.csv')
	dataOD = pd.read_csv('colossus/EBLSST_maghist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, \
			dataOD['histAll'].values, dataOD['histObs'].values, dataOD['allhistRec'].values, dataOD['binEdges'].values, \
			[11, 24], 'mag', 'EBLSST_maghist_final', legendLoc='upper left')

	print("radius ratio")
	data = pd.read_csv('EBLSST_rhist.csv')
	dataOD = pd.read_csv('colossus/EBLSST_rhist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, \
			dataOD['histAll'].values, dataOD['histObs'].values, dataOD['allhistRec'].values, dataOD['binEdges'].values, \
			[0, 3], r'r$_2$/r$_1$', 'EBLSST_rhist_final')