import pandas as pd
import numpy as np

#for Quest
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.lines as mlines

def plotHist(histAll, histObs, histRec, bin_edges, xtitle, fname, xlim = None):
	c1 = '#0294A5'  #turqoise
	c2 = '#d95f02' #orange from color brewer
	c3 = '#00353E' #slate
	c4 = '#508201' #olive
	f,(ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5, 12), sharex=True)

	binHalf = (bin_edges[1] - bin_edges[0])/2.



	#CDF
	cdfAll = []
	cdfObs = []
	cdfRec = []
	for i in range(len(histAll)):
		cdfAll.append(np.sum(histAll[:i])/np.sum(histAll))
	for i in range(len(histObs)):
		cdfObs.append(np.sum(histObs[:i])/np.sum(histObs))
	for i in range(len(histRec)):
		cdfRec.append(np.sum(histRec[:i])/np.sum(histRec))
	ax1.step(bin_edges, cdfAll, color=c1, label='All')
	ax1.step(bin_edges, cdfObs, color=c2, label='Observable')
	ax1.step(bin_edges, cdfRec, color=c3, label='Recoverable')
	ax1.set_ylabel('CDF')


	#PDF
	ax2.step(bin_edges, histAll/np.sum(histAll), color=c1, label='All')
	ax2.step(bin_edges, histObs/np.sum(histObs), color=c2, label='Observable')
	ax2.step(bin_edges, histRec/np.sum(histRec), color=c3, label='Recoverable')
	ax2.set_ylabel('PDF')
	ax2.set_yscale('log')

	#ratio
	#prepend some values at y of zero so that the step plots look correct
	use = np.where(histAll > 0)[0]
	b = bin_edges[use]
	r = histObs[use]/histAll[use]
	ax3.step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c2, label='Observable/All')
	ax3.plot(b - binHalf, r, 'o',color=c1, markersize=5, markeredgecolor=c2)

	r = histRec[use]/histAll[use]
	ax3.step(np.append(b[0] - 2*binHalf, b), np.append(0,r), color=c3, label='Recoverable/All')
	ax3.plot(b - binHalf, r, 'o',color=c1, markersize=5, markeredgecolor=c3)

	use = np.where(histObs > 0)[0]
	b = bin_edges[use]
	r =  histRec[use]/histObs[use]
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
	f.savefig(fname+'_ratio.pdf',format='pdf', bbox_inches = 'tight')



if __name__ == "__main__":

	data = pd.read_csv('EBLSST_lphist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, 'log(Period [days])', 'EBLSST_lphist_final', xlim=[-2,5])

	data = pd.read_csv('EBLSST_ehist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, 'Eccentricity', 'EBLSST_ehist_final', xlim=[0,1])

	data = pd.read_csv('EBLSST_dhist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, 'Distance (kpc)', 'EBLSST_dhist_final', xlim=[0,25])

	data = pd.read_csv('EBLSST_m1hist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, r'm$_1$ (M$_\odot$)', 'EBLSST_m1hist_final', xlim=[0,3])

	data = pd.read_csv('EBLSST_qhist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, r'q (m$_2$/m$_1$)', 'EBLSST_qhist_final', xlim=[0,1])

	data = pd.read_csv('EBLSST_rhist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, r'r$_2$/r$_1$', 'EBLSST_rhist_final', xlim=[0,3])

	data = pd.read_csv('EBLSST_maghist.csv')
	plotHist(data['histAll'].values, data['histObs'].values, data['allhistRec'].values, data['binEdges'].values, r'$r$_ [mag]', 'EBLSST_maghist_final', xlim=[12, 24])