import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.modeling import models, fitting

#for Quest
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
def file_len(fname):
	i = 0
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

def fitRagfb():
	x = [0.05, 0.1, 1, 8, 15]  #estimates of midpoints in bins, and using this: https://sites.uni.edu/morgans/astro/course/Notes/section2/spectralmasses.html
	y = [0.20, 0.35, 0.50, 0.70, 0.75]
	init = models.PowerLaw1D(amplitude=0.5, x_0=1, alpha=-1.)
	fitter = fitting.LevMarLSQFitter()
	fit = fitter(init, x, y)

	return fit

def saveHist(histAll, histObs, histRec, bin_edges, xtitle, fname, filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_','all']):
	c1 = '#0294A5'  #turqoise
	c2 = '#d95f02' #orange from color brewer
	c3 = '#00353E' #slate
	f,(ax1, ax2) = plt.subplots(2,1,figsize=(5, 8), sharex=True)

	histAll = np.insert(histAll,0,0)
	histObs = np.insert(histObs,0,0)
	for f in filters:
		histRec[f] = np.insert(histRec[f],0,0)

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
	f.subplots_adjust(hspace=0)
	f.savefig(fname+'.pdf',format='pdf', bbox_inches = 'tight')

	#write to a text file
	with open(fname+'.csv','w') as f:
		outline = 'binEdges,histAll,histObs'
		for f in filters:
			outline += ','+f+'histRec'
		outline += '\n'
		f.write(outline)
		for i in range(len(bin_edges)):
			b = bin_edges
			hi
		for (b,a,o,r) in zip(bin_edges, histAll, histObs, histRec):
			outline = str(bin_edges[i])+','+str(histAll[i])+','+str(histObs[i])
			for f in filters:
				outline += ','+str(histRec[h][i])
			outline += '\n'
			f.write(outline

if __name__ == "__main__":

	filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_', 'all']

	#get the Raghavan binary fraction fit
	fbFit= fitRagfb()
	print(fbFit)
		
	#cutoff in percent error for "recovered"
	Pcut = 0.1

	#minimum number of lines to consider in file
	Nlim = 3

	#bins for all the histograms
	Nbins = 25
	mbins = np.arange(0,10, 0.1, dtype='float')
	qbins = np.arange(0,1, 0.1, dtype='float')
	ebins = np.arange(0, 1.05, 0.05, dtype='float')
	lpbins = np.arange(-2, 10, 0.5, dtype='float')
	dbins = np.arange(0, 40, 1, dtype='float')
	magbins = np.arange(11, 25, 1, dtype='float')
	rbins = np.arange(0, 100, 0.2, dtype='float')

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
	for f in filters:
		m1hRec[f] = np.zeros_like(mbins)[1:]
		qhRec[f] = np.zeros_like(qbins)[1:]
		ehRec[f] = np.zeros_like(ebins)[1:]
		lphRec[f] = np.zeros_like(lpbins)[1:]
		dhRec[f] = np.zeros_like(dbins)[1:]
		maghRec[f] = np.zeros_like(magbins)[1:]
		rhRec[f] = np.zeros_like(rbins)[1:]

	RA = []
	Dec = []
	recFrac = []
	recN = []
	rawN = []
	obsN = []
	fileN = []
	fileObsN = []
	fileRecN = []

	allNPrsa = []
	obsNPrsa = []
	recNPrsa = []

	#Read in all the data and make the histograms
	d = "../output_files/"
	files = os.listdir(d)
	IDs = []
	for i, f in enumerate(files):
		print(round(i/len(files),4), f)
		fl = file_len(d+f)
		if (fl >= 4):
			#read in the header
			header = pd.read_csv(d+f, nrows=1)
	######################
	#NEED TO ACCOUNT FOR THE BINARY FRACTION when combining histograms
	#####################
	# Also, this weights those near the galactic plane sooo highly (and these are usually poorly recovered), that the resulting histograms are VERY noisy (since we're basically just looking at a few fields new to galactic plane)
			Nmult = header['NstarsTRILEGAL'][0]
			#Nmult = 1.

			RA.append(header['OpSimRA'])
			Dec.append(header['OpSimDec'])

			#read in rest of the file
			data = pd.read_csv(d+f, header = 2).dropna()
			rF = 0.
			rN = 0.
			Nrec = 0.
			Nobs = 0.
			raN = 0.
			obN = 0.
			fiN = 0.
			fioN = 0.
			firN = 0.
			NallPrsa = 0.
			NobsPrsa = 0.
			NrecPrsa = 0.
			Nall = len(data.index)
			prsa = data.loc[(data['appMagMean_r'] <= 19.5) & (data['p'] < 1000) & (data['p'] > 0.5)]
			NallPrsa = len(prsa.index)
			if (Nall >= Nlim):
				#create histograms
				#All
				m1hAll0, m1b = np.histogram(data["m1"], bins=mbins)
				qhAll0, qb = np.histogram(data["m2"]/data["m1"], bins=qbins)
				ehAll0, eb = np.histogram(data["e"], bins=ebins)
				lphAll0, lpb = np.histogram(np.ma.log10(data["p"].values).filled(-999), bins=lpbins)
				dhAll0, db = np.histogram(data["d"], bins=dbins)
				maghAll0, magb = np.histogram(data["appMagMean_r"], bins=magbins)
				rhAll0, rb = np.histogram(data["r2"]/data["r1"], bins=rbins)

				#account for the binary fraction, as a function of mass
				dm1 = np.diff(m1b)
				m1val = m1b[:-1] + dm1/2.
				fb = np.sum(m1hAll0/Nall*fbFit(m1val))
				#print("fb = ", fb)
				Nmult *= fb

							
				m1hAll += m1hAll0/Nall*Nmult
				qhAll += qhAll0/Nall*Nmult
				ehAll += ehAll0/Nall*Nmult
				lphAll += lphAll0/Nall*Nmult
				dhAll += dhAll0/Nall*Nmult
				maghAll += maghAll0/Nall*Nmult
				rhAll += rhAll0/Nall*Nmult

				#Obs
				obs = data.loc[data['LSM_PERIOD'] != -999]
				Nobs = len(obs.index)
				prsaObs = data.loc[(data['appMagMean_r'] <= 19.5) & (data['p'] < 1000) & (data['p'] >0.5) & (data['LSM_PERIOD'] != -999)]
				NobsPrsa = len(prsaObs.index)
				if (Nobs >= Nlim):
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
					for f in filters:
						key = f+'LSS_PERIOD'
						if (f == 'all'):
							key = 'LSM_PERIOD'
						fullP = abs(data[key] - data['p'])/data['p']
						halfP = abs(data[key] - 0.5*data['p'])/(0.5*data['p'])
						twiceP = abs(data[key] - 2.*data['p'])/(2.*data['p'])
						rec = data.loc[(data[key] != -999) & ( (fullP < Pcut) | (halfP < Pcut) | (twiceP < Pcut))]
						Nrec = len(rec.index)

						if (Nrec >= Nlim):
							m1hRec0, m1b = np.histogram(rec["m1"], bins=mbins)
							qhRec0, qb = np.histogram(rec["m2"]/rec["m1"], bins=qbins)
							ehRec0, eb = np.histogram(rec["e"], bins=ebins)
							lphRec0, lpb = np.histogram(np.ma.log10(rec["p"].values).filled(-999), bins=lpbins)
							dhRec0, db = np.histogram(rec["d"], bins=dbins)
							maghRec0, magb = np.histogram(rec["appMagMean_r"], bins=magbins)
							rhRec0, rb = np.histogram(rec["r2"]/rec["r1"], bins=rbins)
							m1hRec[f] += m1hRec0/Nall*Nmult
							qhRec[f] += qhRec0/Nall*Nmult
							ehRec[f] += ehRec0/Nall*Nmult
							lphRec[f] += lphRec0/Nall*Nmult
							dhRec[f] += dhRec0/Nall*Nmult
							maghRec[f] += maghRec0/Nall*Nmult
							rhRec[f] += rhRec0/Nall*Nmult

							#for the mollweide
							if (f == 'all'):
								rF = Nrec/Nall
								rN = Nrec/Nall*Nmult
								raN = Nmult
								obN = Nobs/Nall*Nmult
								fiN = Nall
								fioN = Nobs
								firN = Nrec

								prsaRec = data.loc[(data['appMagMean_r'] <= 19.5) & (data['p'] < 1000) & (data['p'] >0.5) & (data['LSM_PERIOD'] != -999) & ( (fullP < Pcut) | (halfP < Pcut) | (twiceP < Pcut))]
								NrecPrsa = len(prsaRec.index)
								NrecPrsa = NrecPrsa/Nall*Nmult
								NobsPrsa = NobsPrsa/Nall*Nmult
								NallPrsa = NallPrsa/Nall*Nmult		



			recFrac.append(rF)
			recN.append(rN)
			rawN.append(raN)
			obsN.append(obN)
			fileN.append(fiN)
			fileObsN.append(fioN)
			fileRecN.append(firN)
			allNPrsa.append(NallPrsa)
			obsNPrsa.append(NobsPrsa)
			recNPrsa.append(NrecPrsa)
			#print(np.sum(lphRec), np.sum(recN), np.sum(lphRec)/np.sum(recN), np.sum(lphRec0), Nrec, np.sum(lphRec0)/Nrec, np.sum(lphObs), np.sum(obsN), np.sum(lphObs)/np.sum(obsN))


	#plot and save the histograms
	saveHist(m1hAll, m1hObs, m1hRec, m1b, 'm1 (Msolar)', 'EBLSST_m1hist')
	saveHist(qhAll, qhObs, qhRec, qb, 'q (m2/m1)', 'EBLSST_qhist')
	saveHist(ehAll, ehObs, ehRec, eb, 'e', 'EBLSST_ehist')
	saveHist(lphAll, lphObs, lphRec, lpb, 'log(P [days])', 'EBLSST_lphist')
	saveHist(dhAll, dhObs, dhRec, db, 'd (kpc)', 'EBLSST_dhist')
	saveHist(maghAll, maghObs, maghRec, magb, 'mag', 'EBLSST_maghist')
	saveHist(rhAll, rhObs, rhRec, rb, 'r2/r1', 'EBLSST_rhist')


	#make the mollweide
	coords = SkyCoord(RA, Dec, unit=(units.degree, units.degree),frame='icrs')	
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
	mlw = ax.scatter(np.array(RAwrap).ravel()*np.pi/180., np.array(Decwrap).ravel()*np.pi/180., c=np.array(recFrac)*100., cmap='viridis_r', s = 4)
	cbar = f.colorbar(mlw, shrink=0.7)
	cbar.set_label(r'% recovered')
	f.savefig('mollweide_pct.pdf',format='pdf', bbox_inches = 'tight')

	f, ax = plt.subplots(subplot_kw={'projection': "mollweide"}, figsize=(8,5))
	ax.grid(True)
	#ax.set_xlabel(r"$l$",fontsize=16)
	#ax.set_ylabel(r"$b$",fontsize=16)
	#mlw = ax.scatter(lGal.ravel()*np.pi/180., bGal.ravel()*np.pi/180., c=np.log10(np.array(recN)), cmap='viridis_r', s = 4)
	ax.set_xlabel("RA",fontsize=16)
	ax.set_ylabel("Dec",fontsize=16)
	mlw = ax.scatter(np.array(RAwrap).ravel()*np.pi/180., np.array(Decwrap).ravel()*np.pi/180., c=np.log10(np.array(recN)), cmap='viridis_r', s = 4)
	cbar = f.colorbar(mlw, shrink=0.7)
	cbar.set_label(r'log10(N) recovered')
	f.savefig('mollweide_N.pdf',format='pdf', bbox_inches = 'tight')

	print("###################")
	print("number of binaries in input files (raw, log):",np.sum(fileN), np.log10(np.sum(fileN)))
	print("number of binaries in tested with gatspy (raw, log):",np.sum(fileObsN), np.log10(np.sum(fileObsN)))
	print("number of binaries in recovered with gatspy (raw, log):",np.sum(fileRecN), np.log10(np.sum(fileRecN)))
	print("###################")
	print("total in sample (raw, log):",np.sum(rawN), np.log10(np.sum(rawN)))
	print("total observable (raw, log):",np.sum(obsN), np.log10(np.sum(obsN)))
	print("total recovered (raw, log):",np.sum(recN), np.log10(np.sum(recN)))
	print("###################")
	print("total in Prsa r<19.5 P<1000d sample (raw, log):",np.sum(allNPrsa), np.log10(np.sum(allNPrsa)))
	print("total observable in Prsa r<19.5 P<1000d sample (raw, log):",np.sum(obsNPrsa), np.log10(np.sum(obsNPrsa)))
	print("total recovered in Prsa r<19.5 P<1000d sample (raw, log):",np.sum(recNPrsa), np.log10(np.sum(recNPrsa)))
