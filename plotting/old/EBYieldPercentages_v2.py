import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from matplotlib import cm, colors
from astropy.modeling import models, fitting

# Reading in all data files at once
import glob
path_normal ='/projects/p30137/ageller/testing/EBLSST/add_m5/output_files' 
allFiles_normal = glob.glob(path_normal + "/*.csv")
path_fast = '/projects/p30137/ageller/testing/EBLSST/add_m5/fast/old/output_files'
allFiles_fast = glob.glob(path_fast + "/*.csv")
path_obsDist = '/projects/p30137/ageller/testing/EBLSST/add_m5/fast/old/obsDist/output_files'
allFiles_obsDist = glob.glob(path_obsDist + "/*.csv")

N_totalnormal_array = []
N_totalobservablenormal_array = []
N_totalrecoverablenormal_array = []

N_totalnormal_array_03 = []
N_totalobservablenormal_array_03 = []
N_totalrecoverablenormal_array_03 = []

N_totalnormal_array_1 = []
N_totalobservablenormal_array_1 = []
N_totalrecoverablenormal_array_1 = []

N_totalnormal_array_10 = []
N_totalobservablenormal_array_10 = []
N_totalrecoverablenormal_array_10 = []

N_totalnormal_array_30 = []
N_totalobservablenormal_array_30 = []
N_totalrecoverablenormal_array_30 = []

N_totalnormal_array_100 = []
N_totalobservablenormal_array_100 = []
N_totalrecoverablenormal_array_100 = []

N_totalnormal_array_1000 = []
N_totalobservablenormal_array_1000 = []
N_totalrecoverablenormal_array_1000 = []

N_totalnormal22_array = []
N_totalobservablenormal22_array = []
N_totalrecoverablenormal22_array = []

N_totalnormal22_array_03 = []
N_totalobservablenormal22_array_03 = []
N_totalrecoverablenormal22_array_03 = []

N_totalnormal22_array_1 = []
N_totalobservablenormal22_array_1 = []
N_totalrecoverablenormal22_array_1 = []

N_totalnormal22_array_10 = []
N_totalobservablenormal22_array_10 = []
N_totalrecoverablenormal22_array_10 = []

N_totalnormal22_array_30 = []
N_totalobservablenormal22_array_30 = []
N_totalrecoverablenormal22_array_30 = []

N_totalnormal22_array_100 = []
N_totalobservablenormal22_array_100 = []
N_totalrecoverablenormal22_array_100 = []

N_totalnormal22_array_1000 = []
N_totalobservablenormal22_array_1000 = []
N_totalrecoverablenormal22_array_1000 = []

N_totalnormal195_array = []
N_totalobservablenormal195_array = []
N_totalrecoverablenormal195_array = []

N_totalnormal195_array_03 = []
N_totalobservablenormal195_array_03 = []
N_totalrecoverablenormal195_array_03 = []

N_totalnormal195_array_1 = []
N_totalobservablenormal195_array_1 = []
N_totalrecoverablenormal195_array_1 = []

N_totalnormal195_array_10 = []
N_totalobservablenormal195_array_10 = []
N_totalrecoverablenormal195_array_10 = []

N_totalnormal195_array_30 = []
N_totalobservablenormal195_array_30 = []
N_totalrecoverablenormal195_array_30 = []

N_totalnormal195_array_100 = []
N_totalobservablenormal195_array_100 = []
N_totalrecoverablenormal195_array_100 = []

N_totalnormal195_array_1000 = []
N_totalobservablenormal195_array_1000 = []
N_totalrecoverablenormal195_array_1000 = []


N_totalfast_array = []
N_totalobservablefast_array = []
N_totalrecoverablefast_array = []

N_totalfast_array_03 = []
N_totalobservablefast_array_03 = []
N_totalrecoverablefast_array_03 = []

N_totalfast_array_1 = []
N_totalobservablefast_array_1 = []
N_totalrecoverablefast_array_1 = []

N_totalfast_array_10 = []
N_totalobservablefast_array_10 = []
N_totalrecoverablefast_array_10 = []

N_totalfast_array_30 = []
N_totalobservablefast_array_30 = []
N_totalrecoverablefast_array_30 = []

N_totalfast_array_100 = []
N_totalobservablefast_array_100 = []
N_totalrecoverablefast_array_100 = []

N_totalfast_array_1000 = []
N_totalobservablefast_array_1000 = []
N_totalrecoverablefast_array_1000 = []

N_totalfast22_array = []
N_totalobservablefast22_array = []
N_totalrecoverablefast22_array = []

N_totalfast22_array_03 = []
N_totalobservablefast22_array_03 = []
N_totalrecoverablefast22_array_03 = []

N_totalfast22_array_1 = []
N_totalobservablefast22_array_1 = []
N_totalrecoverablefast22_array_1 = []

N_totalfast22_array_10 = []
N_totalobservablefast22_array_10 = []
N_totalrecoverablefast22_array_10 = []

N_totalfast22_array_30 = []
N_totalobservablefast22_array_30 = []
N_totalrecoverablefast22_array_30 = []

N_totalfast22_array_100 = []
N_totalobservablefast22_array_100 = []
N_totalrecoverablefast22_array_100 = []

N_totalfast22_array_1000 = []
N_totalobservablefast22_array_1000 = []
N_totalrecoverablefast22_array_1000 = []

N_totalfast195_array = []
N_totalobservablefast195_array = []
N_totalrecoverablefast195_array = []

N_totalfast195_array_03 = []
N_totalobservablefast195_array_03 = []
N_totalrecoverablefast195_array_03 = []

N_totalfast195_array_1 = []
N_totalobservablefast195_array_1 = []
N_totalrecoverablefast195_array_1 = []

N_totalfast195_array_10 = []
N_totalobservablefast195_array_10 = []
N_totalrecoverablefast195_array_10 = []

N_totalfast195_array_30 = []
N_totalobservablefast195_array_30 = []
N_totalrecoverablefast195_array_30 = []

N_totalfast195_array_100 = []
N_totalobservablefast195_array_100 = []
N_totalrecoverablefast195_array_100 = []

N_totalfast195_array_1000 = []
N_totalobservablefast195_array_1000 = []
N_totalrecoverablefast195_array_1000 = []

N_totalobsDist_array = []
N_totalobservableobsDist_array = []
N_totalrecoverableobsDist_array = []

N_totalobsDist_array_03 = []
N_totalobservableobsDist_array_03 = []
N_totalrecoverableobsDist_array_03 = []

N_totalobsDist_array_1 = []
N_totalobservableobsDist_array_1 = []
N_totalrecoverableobsDist_array_1 = []

N_totalobsDist_array_10 = []
N_totalobservableobsDist_array_10 = []
N_totalrecoverableobsDist_array_10 = []

N_totalobsDist_array_30 = []
N_totalobservableobsDist_array_30 = []
N_totalrecoverableobsDist_array_30 = []

N_totalobsDist_array_100 = []
N_totalobservableobsDist_array_100 = []
N_totalrecoverableobsDist_array_100 = []

N_totalobsDist_array_1000 = []
N_totalobservableobsDist_array_1000 = []
N_totalrecoverableobsDist_array_1000 = []

N_totalobsDist22_array = []
N_totalobservableobsDist22_array = []
N_totalrecoverableobsDist22_array = []

N_totalobsDist22_array_03 = []
N_totalobservableobsDist22_array_03 = []
N_totalrecoverableobsDist22_array_03 = []

N_totalobsDist22_array_1 = []
N_totalobservableobsDist22_array_1 = []
N_totalrecoverableobsDist22_array_1 = []

N_totalobsDist22_array_10 = []
N_totalobservableobsDist22_array_10 = []
N_totalrecoverableobsDist22_array_10 = []

N_totalobsDist22_array_30 = []
N_totalobservableobsDist22_array_30 = []
N_totalrecoverableobsDist22_array_30 = []

N_totalobsDist22_array_100 = []
N_totalobservableobsDist22_array_100 = []
N_totalrecoverableobsDist22_array_100 = []

N_totalobsDist22_array_1000 = []
N_totalobservableobsDist22_array_1000 = []
N_totalrecoverableobsDist22_array_1000 = []

N_totalobsDist195_array = []
N_totalobservableobsDist195_array = []
N_totalrecoverableobsDist195_array = []

N_totalobsDist195_array_03 = []
N_totalobservableobsDist195_array_03 = []
N_totalrecoverableobsDist195_array_03 = []

N_totalobsDist195_array_1 = []
N_totalobservableobsDist195_array_1 = []
N_totalrecoverableobsDist195_array_1 = []

N_totalobsDist195_array_10 = []
N_totalobservableobsDist195_array_10 = []
N_totalrecoverableobsDist195_array_10 = []

N_totalobsDist195_array_30 = []
N_totalobservableobsDist195_array_30 = []
N_totalrecoverableobsDist195_array_30 = []

N_totalobsDist195_array_100 = []
N_totalobservableobsDist195_array_100 = []
N_totalrecoverableobsDist195_array_100 = []

N_totalobsDist195_array_1000 = []
N_totalobservableobsDist195_array_1000 = []
N_totalrecoverableobsDist195_array_1000 = []


def fitRagfb():
	x = [0.05, 0.1, 1, 8, 15]  #estimates of midpoints in bins, and using this: https://sites.uni.edu/morgans/astro/course/Notes/section2/spectralmasses.html
	y = [0.20, 0.35, 0.50, 0.70, 0.75]
	init = models.PowerLaw1D(amplitude=0.5, x_0=1, alpha=-1.)
	fitter = fitting.LevMarLSQFitter()
	fit = fitter(init, x, y)

	return fit

fbFit= fitRagfb()
mbins = np.arange(0,10, 0.1, dtype='float')


cutP = 0.10	#condition on recoverability/tolerance

for filenormal_ in sorted(allFiles_normal):

	filename = filenormal_[60:]
	fileid = filename.strip('output_file.csv')
	print ("I'm starting " + fileid)


	datnormal = pd.read_csv(filenormal_, sep = ',', header=2)
	PeriodIn = datnormal['p']		 # input period -- 'p' in data file

	##########################################################

	datnormal1 = pd.read_csv(filenormal_, sep = ',', header=0, nrows=1)
	N_tri = datnormal1["NstarsTRILEGAL"][0]
	#print("N_tri = ", N_tri)
	Nall = len(PeriodIn)


	m1hAll0, m1b = np.histogram(datnormal["m1"], bins=mbins)
	dm1 = np.diff(m1b)
	m1val = m1b[:-1] + dm1/2.
	fb = np.sum(m1hAll0/Nall*fbFit(m1val))

	N_mult = N_tri*fb

	##########################################################

	if len(PeriodIn) == 0.:
		continue
	if N_tri == 0:
		continue
	else:
		PeriodOut = datnormal['LSM_PERIOD'] #LSM_PERIOD in data file
		appMagMean = datnormal['appMagMean'] #apparent magnitude, will use to make cuts for 24 (default), 22, and then Kepler's range (?? -- brighter than LSST can manage-- to 19) OR 19.5 (SNR = 10)
		
		observable = datnormal.loc[PeriodOut != -999].index
		observable_03 = datnormal.loc[(PeriodIn <= 0.3) & (PeriodOut != -999)].index
		observable_1 = datnormal.loc[(PeriodIn <= 1) & (PeriodOut != -999)].index
		observable_10 = datnormal.loc[(PeriodIn <= 10) & (PeriodOut != -999)].index
		observable_30 = datnormal.loc[(PeriodIn <= 30) & (PeriodOut != -999)].index
		observable_100 = datnormal.loc[(PeriodIn <= 100) & (PeriodOut != -999)].index
		observable_1000 = datnormal.loc[(PeriodIn <= 1000) & (PeriodOut != -999)].index
		
		observable_22 = datnormal.loc[(PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_03_22 = datnormal.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_1_22 = datnormal.loc[(PeriodIn <= 1) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_10_22 = datnormal.loc[(PeriodIn <= 10) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_30_22 = datnormal.loc[(PeriodIn <= 30) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_100_22 = datnormal.loc[(PeriodIn <= 100) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_1000_22 = datnormal.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & (appMagMean <= 22.)].index

		observable_195 = datnormal.loc[(PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_03_195 = datnormal.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_1_195 = datnormal.loc[(PeriodIn <= 1) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_10_195 = datnormal.loc[(PeriodIn <= 10) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_30_195 = datnormal.loc[(PeriodIn <= 30) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_100_195 = datnormal.loc[(PeriodIn <= 100) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_1000_195 = datnormal.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & (appMagMean <= 19.5)].index

		fullP = abs(PeriodOut - PeriodIn)/PeriodIn
		halfP = abs(PeriodOut - 0.5*PeriodIn)/(0.5*PeriodIn)
		twiceP = abs(PeriodOut - 2*PeriodIn)/(2*PeriodIn)

		recoverable = datnormal.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_03 = datnormal.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_1 = datnormal.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_10 = datnormal.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_30 = datnormal.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_100 = datnormal.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_1000 = datnormal.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index

		recoverable_22 = datnormal.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_03_22 = datnormal.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_1_22 = datnormal.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_10_22 = datnormal.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_30_22 = datnormal.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_100_22 = datnormal.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_1000_22 = datnormal.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index

		recoverable_195 = datnormal.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_03_195 = datnormal.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_1_195 = datnormal.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_10_195 = datnormal.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_30_195 = datnormal.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_100_195 = datnormal.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_1000_195 = datnormal.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index

		P03 = datnormal.loc[PeriodIn <= 0.3].index
		P1 = datnormal.loc[PeriodIn <= 1].index
		P10 = datnormal.loc[PeriodIn <= 10].index
		P30 = datnormal.loc[PeriodIn <= 30].index
		P100 = datnormal.loc[PeriodIn <= 100].index
		P1000 = datnormal.loc[PeriodIn <= 1000].index

		P_22 = datnormal.loc[appMagMean <= 22.].index
		P03_22 = datnormal.loc[(PeriodIn <= 0.3) & (appMagMean <= 22.)].index
		P1_22 = datnormal.loc[(PeriodIn <= 1) & (appMagMean <= 22.)].index
		P10_22 = datnormal.loc[(PeriodIn <= 10) & (appMagMean <= 22.)].index
		P30_22 = datnormal.loc[(PeriodIn <= 30) & (appMagMean <= 22.)].index
		P100_22 = datnormal.loc[(PeriodIn <= 100) & (appMagMean <= 22.)].index
		P1000_22 = datnormal.loc[(PeriodIn <= 1000) & (appMagMean <= 22.)].index

		P_195 = datnormal.loc[appMagMean <= 19.5].index
		P03_195 = datnormal.loc[(PeriodIn <= 0.3) & (appMagMean <= 19.5)].index
		P1_195 = datnormal.loc[(PeriodIn <= 1) & (appMagMean <= 19.5)].index
		P10_195 = datnormal.loc[(PeriodIn <= 10) & (appMagMean <= 19.5)].index
		P30_195 = datnormal.loc[(PeriodIn <= 30) & (appMagMean <= 19.5)].index
		P100_195 = datnormal.loc[(PeriodIn <= 100) & (appMagMean <= 19.5)].index
		P1000_195 = datnormal.loc[(PeriodIn <= 1000) & (appMagMean <= 19.5)].index

		N_all = (len(PeriodIn)/len(PeriodIn))*N_mult
		N_all03 = (len(P03)/len(PeriodIn))*N_mult
		N_all1 = (len(P1)/len(PeriodIn))*N_mult
		N_all10 = (len(P10)/len(PeriodIn))*N_mult
		N_all30 = (len(P30)/len(PeriodIn))*N_mult
		N_all100 = (len(P100)/len(PeriodIn))*N_mult
		N_all1000 = (len(P1000)/len(PeriodIn))*N_mult

		N_all_22 = (len(P_22)/len(PeriodIn))*N_mult
		N_all03_22 = (len(P03_22)/len(PeriodIn))*N_mult
		N_all1_22 = (len(P1_22)/len(PeriodIn))*N_mult
		N_all10_22 = (len(P10_22)/len(PeriodIn))*N_mult
		N_all30_22 = (len(P30_22)/len(PeriodIn))*N_mult
		N_all100_22 = (len(P100_22)/len(PeriodIn))*N_mult
		N_all1000_22 = (len(P1000_22)/len(PeriodIn))*N_mult

		N_all_195 = (len(P_195)/len(PeriodIn))*N_mult
		N_all03_195 = (len(P03_195)/len(PeriodIn))*N_mult
		N_all1_195 = (len(P1_195)/len(PeriodIn))*N_mult
		N_all10_195 = (len(P10_195)/len(PeriodIn))*N_mult
		N_all30_195 = (len(P30_195)/len(PeriodIn))*N_mult
		N_all100_195 = (len(P100_195)/len(PeriodIn))*N_mult
		N_all1000_195 = (len(P1000_195)/len(PeriodIn))*N_mult

		N_obs = (len(observable)/len(PeriodIn))*N_mult
		N_obs03 = (len(observable_03)/len(PeriodIn))*N_mult
		N_obs1 = (len(observable_1)/len(PeriodIn))*N_mult
		N_obs10 = (len(observable_10)/len(PeriodIn))*N_mult
		N_obs30 = (len(observable_30)/len(PeriodIn))*N_mult
		N_obs100 = (len(observable_100)/len(PeriodIn))*N_mult
		N_obs1000 = (len(observable_1000)/len(PeriodIn))*N_mult

		N_obs_22 = (len(observable_22)/len(PeriodIn))*N_mult
		N_obs03_22 = (len(observable_03_22)/len(PeriodIn))*N_mult
		N_obs1_22 = (len(observable_1_22)/len(PeriodIn))*N_mult
		N_obs10_22 = (len(observable_10_22)/len(PeriodIn))*N_mult
		N_obs30_22 = (len(observable_30_22)/len(PeriodIn))*N_mult
		N_obs100_22 = (len(observable_100_22)/len(PeriodIn))*N_mult
		N_obs1000_22 = (len(observable_1000_22)/len(PeriodIn))*N_mult

		N_obs_195 = (len(observable_195)/len(PeriodIn))*N_mult
		N_obs03_195 = (len(observable_03_195)/len(PeriodIn))*N_mult
		N_obs1_195 = (len(observable_1_195)/len(PeriodIn))*N_mult
		N_obs10_195 = (len(observable_10_195)/len(PeriodIn))*N_mult
		N_obs30_195 = (len(observable_30_195)/len(PeriodIn))*N_mult
		N_obs100_195 = (len(observable_100_195)/len(PeriodIn))*N_mult
		N_obs1000_195 = (len(observable_1000_195)/len(PeriodIn))*N_mult

		N_rec = (len(recoverable)/len(PeriodIn))*N_mult
		N_rec03 = (len(recoverable_03)/len(PeriodIn))*N_mult
		N_rec1 = (len(recoverable_1)/len(PeriodIn))*N_mult
		N_rec10 = (len(recoverable_10)/len(PeriodIn))*N_mult
		N_rec30 = (len(recoverable_30)/len(PeriodIn))*N_mult
		N_rec100 = (len(recoverable_100)/len(PeriodIn))*N_mult
		N_rec1000 = (len(recoverable_1000)/len(PeriodIn))*N_mult

		N_rec_22 = (len(recoverable_22)/len(PeriodIn))*N_mult
		N_rec03_22 = (len(recoverable_03_22)/len(PeriodIn))*N_mult
		N_rec1_22 = (len(recoverable_1_22)/len(PeriodIn))*N_mult
		N_rec10_22 = (len(recoverable_10_22)/len(PeriodIn))*N_mult
		N_rec30_22 = (len(recoverable_30_22)/len(PeriodIn))*N_mult
		N_rec100_22 = (len(recoverable_100_22)/len(PeriodIn))*N_mult
		N_rec1000_22 = (len(recoverable_1000_22)/len(PeriodIn))*N_mult

		N_rec_195 = (len(recoverable_195)/len(PeriodIn))*N_mult
		N_rec03_195 = (len(recoverable_03_195)/len(PeriodIn))*N_mult
		N_rec1_195 = (len(recoverable_1_195)/len(PeriodIn))*N_mult
		N_rec10_195 = (len(recoverable_10_195)/len(PeriodIn))*N_mult
		N_rec30_195 = (len(recoverable_30_195)/len(PeriodIn))*N_mult
		N_rec100_195 = (len(recoverable_100_195)/len(PeriodIn))*N_mult
		N_rec1000_195 = (len(recoverable_1000_195)/len(PeriodIn))*N_mult

		N_totalnormal_array.append(float(N_all))
		N_totalobservablenormal_array.append(float(N_obs))
		N_totalrecoverablenormal_array.append(float(N_rec))

		N_totalnormal_array_03.append(float(N_all03))
		N_totalobservablenormal_array_03.append(float(N_obs03))
		N_totalrecoverablenormal_array_03.append(float(N_rec03))

		N_totalnormal_array_1.append(float(N_all1))
		N_totalobservablenormal_array_1.append(float(N_obs1))
		N_totalrecoverablenormal_array_1.append(float(N_rec1))

		N_totalnormal_array_10.append(float(N_all10))
		N_totalobservablenormal_array_10.append(float(N_obs10))
		N_totalrecoverablenormal_array_10.append(float(N_rec10))

		N_totalnormal_array_30.append(float(N_all30))
		N_totalobservablenormal_array_30.append(float(N_obs30))
		N_totalrecoverablenormal_array_30.append(float(N_rec30))

		N_totalnormal_array_100.append(float(N_all100))
		N_totalobservablenormal_array_100.append(float(N_obs100))
		N_totalrecoverablenormal_array_100.append(float(N_rec100))

		N_totalnormal_array_1000.append(float(N_all1000))
		N_totalobservablenormal_array_1000.append(float(N_obs1000))
		N_totalrecoverablenormal_array_1000.append(float(N_rec1000))

		N_totalnormal22_array.append(float(N_all_22))
		N_totalobservablenormal22_array.append(float(N_obs_22))
		N_totalrecoverablenormal22_array.append(float(N_rec_22))

		N_totalnormal22_array_03.append(float(N_all03_22))
		N_totalobservablenormal22_array_03.append(float(N_obs03_22))
		N_totalrecoverablenormal22_array_03.append(float(N_rec03_22))

		N_totalnormal22_array_1.append(float(N_all1_22))
		N_totalobservablenormal22_array_1.append(float(N_obs1_22))
		N_totalrecoverablenormal22_array_1.append(float(N_rec1_22))

		N_totalnormal22_array_10.append(float(N_all10_22))
		N_totalobservablenormal22_array_10.append(float(N_obs10_22))
		N_totalrecoverablenormal22_array_10.append(float(N_rec10_22))

		N_totalnormal22_array_30.append(float(N_all30_22))
		N_totalobservablenormal22_array_30.append(float(N_obs30_22))
		N_totalrecoverablenormal22_array_30.append(float(N_rec30_22))

		N_totalnormal22_array_100.append(float(N_all100_22))
		N_totalobservablenormal22_array_100.append(float(N_obs100_22))
		N_totalrecoverablenormal22_array_100.append(float(N_rec100_22))

		N_totalnormal22_array_1000.append(float(N_all1000_22))
		N_totalobservablenormal22_array_1000.append(float(N_obs1000_22))
		N_totalrecoverablenormal22_array_1000.append(float(N_rec1000_22))

		N_totalnormal195_array.append(float(N_all_195))
		N_totalobservablenormal195_array.append(float(N_obs_195))
		N_totalrecoverablenormal195_array.append(float(N_rec_195))

		N_totalnormal195_array_03.append(float(N_all03_195))
		N_totalobservablenormal195_array_03.append(float(N_obs03_195))
		N_totalrecoverablenormal195_array_03.append(float(N_rec03_195))

		N_totalnormal195_array_1.append(float(N_all1_195))
		N_totalobservablenormal195_array_1.append(float(N_obs1_195))
		N_totalrecoverablenormal195_array_1.append(float(N_rec1_195))

		N_totalnormal195_array_10.append(float(N_all10_195))
		N_totalobservablenormal195_array_10.append(float(N_obs10_195))
		N_totalrecoverablenormal195_array_10.append(float(N_rec10_195))

		N_totalnormal195_array_30.append(float(N_all30_195))
		N_totalobservablenormal195_array_30.append(float(N_obs30_195))
		N_totalrecoverablenormal195_array_30.append(float(N_rec30_195))

		N_totalnormal195_array_100.append(float(N_all100_195))
		N_totalobservablenormal195_array_100.append(float(N_obs100_195))
		N_totalrecoverablenormal195_array_100.append(float(N_rec100_195))

		N_totalnormal195_array_1000.append(float(N_all1000_195))
		N_totalobservablenormal195_array_1000.append(float(N_obs1000_195))
		N_totalrecoverablenormal195_array_1000.append(float(N_rec1000_195))


N_totalnormal = np.sum(N_totalnormal_array)
N_totalnormal_03 = np.sum(N_totalnormal_array_03)
N_totalnormal_1 = np.sum(N_totalnormal_array_1)
N_totalnormal_10 = np.sum(N_totalnormal_array_10)
N_totalnormal_30 = np.sum(N_totalnormal_array_30)
N_totalnormal_100 = np.sum(N_totalnormal_array_100)
N_totalnormal_1000 = np.sum(N_totalnormal_array_1000)


N_totalobservablenormal = np.sum(N_totalobservablenormal_array)
N_totalobservablenormal_03 = np.sum(N_totalobservablenormal_array_03)
N_totalobservablenormal_1 = np.sum(N_totalobservablenormal_array_1)
N_totalobservablenormal_10 = np.sum(N_totalobservablenormal_array_10)
N_totalobservablenormal_30 = np.sum(N_totalobservablenormal_array_30)
N_totalobservablenormal_100 = np.sum(N_totalobservablenormal_array_100)
N_totalobservablenormal_1000 = np.sum(N_totalobservablenormal_array_1000)


N_totalrecoverablenormal = np.sum(N_totalrecoverablenormal_array)
N_totalrecoverablenormal_03 = np.sum(N_totalrecoverablenormal_array_03)
N_totalrecoverablenormal_1 = np.sum(N_totalrecoverablenormal_array_1)
N_totalrecoverablenormal_10 = np.sum(N_totalrecoverablenormal_array_10)
N_totalrecoverablenormal_30 = np.sum(N_totalrecoverablenormal_array_30)
N_totalrecoverablenormal_100 = np.sum(N_totalrecoverablenormal_array_100)
N_totalrecoverablenormal_1000 = np.sum(N_totalrecoverablenormal_array_1000)


N_totalnormal22 = np.sum(N_totalnormal22_array)
N_totalnormal22_03 = np.sum(N_totalnormal22_array_03)
N_totalnormal22_1 = np.sum(N_totalnormal22_array_1)
N_totalnormal22_10 = np.sum(N_totalnormal22_array_10)
N_totalnormal22_30 = np.sum(N_totalnormal22_array_30)
N_totalnormal22_100 = np.sum(N_totalnormal22_array_100)
N_totalnormal22_1000 = np.sum(N_totalnormal22_array_1000)


N_totalobservablenormal22 = np.sum(N_totalobservablenormal22_array)
N_totalobservablenormal22_03 = np.sum(N_totalobservablenormal22_array_03)
N_totalobservablenormal22_1 = np.sum(N_totalobservablenormal22_array_1)
N_totalobservablenormal22_10 = np.sum(N_totalobservablenormal22_array_10)
N_totalobservablenormal22_30 = np.sum(N_totalobservablenormal22_array_30)
N_totalobservablenormal22_100 = np.sum(N_totalobservablenormal22_array_100)
N_totalobservablenormal22_1000 = np.sum(N_totalobservablenormal22_array_1000)


N_totalrecoverablenormal22 = np.sum(N_totalrecoverablenormal22_array)
N_totalrecoverablenormal22_03 = np.sum(N_totalrecoverablenormal22_array_03)
N_totalrecoverablenormal22_1 = np.sum(N_totalrecoverablenormal22_array_1)
N_totalrecoverablenormal22_10 = np.sum(N_totalrecoverablenormal22_array_10)
N_totalrecoverablenormal22_30 = np.sum(N_totalrecoverablenormal22_array_30)
N_totalrecoverablenormal22_100 = np.sum(N_totalrecoverablenormal22_array_100)
N_totalrecoverablenormal22_1000 = np.sum(N_totalrecoverablenormal22_array_1000)

N_totalnormal195 = np.sum(N_totalnormal195_array)
N_totalnormal195_03 = np.sum(N_totalnormal195_array_03)
N_totalnormal195_1 = np.sum(N_totalnormal195_array_1)
N_totalnormal195_10 = np.sum(N_totalnormal195_array_10)
N_totalnormal195_30 = np.sum(N_totalnormal195_array_30)
N_totalnormal195_100 = np.sum(N_totalnormal195_array_100)
N_totalnormal195_1000 = np.sum(N_totalnormal195_array_1000)


N_totalobservablenormal195 = np.sum(N_totalobservablenormal195_array)
N_totalobservablenormal195_03 = np.sum(N_totalobservablenormal195_array_03)
N_totalobservablenormal195_1 = np.sum(N_totalobservablenormal195_array_1)
N_totalobservablenormal195_10 = np.sum(N_totalobservablenormal195_array_10)
N_totalobservablenormal195_30 = np.sum(N_totalobservablenormal195_array_30)
N_totalobservablenormal195_100 = np.sum(N_totalobservablenormal195_array_100)
N_totalobservablenormal195_1000 = np.sum(N_totalobservablenormal195_array_1000)


N_totalrecoverablenormal195 = np.sum(N_totalrecoverablenormal195_array)
N_totalrecoverablenormal195_03 = np.sum(N_totalrecoverablenormal195_array_03)
N_totalrecoverablenormal195_1 = np.sum(N_totalrecoverablenormal195_array_1)
N_totalrecoverablenormal195_10 = np.sum(N_totalrecoverablenormal195_array_10)
N_totalrecoverablenormal195_30 = np.sum(N_totalrecoverablenormal195_array_30)
N_totalrecoverablenormal195_100 = np.sum(N_totalrecoverablenormal195_array_100)
N_totalrecoverablenormal195_1000 = np.sum(N_totalrecoverablenormal195_array_1000)


wholerecoverypercent_normal = (N_totalrecoverablenormal/N_totalobservablenormal)*100
wholerecoverypercent_normal_03 = (N_totalrecoverablenormal_03/N_totalobservablenormal_03)*100
wholerecoverypercent_normal_1 = (N_totalrecoverablenormal_1/N_totalobservablenormal_1)*100
wholerecoverypercent_normal_10 = (N_totalrecoverablenormal_10/N_totalobservablenormal_10)*100
wholerecoverypercent_normal_30 = (N_totalrecoverablenormal_30/N_totalobservablenormal_30)*100
wholerecoverypercent_normal_100 = (N_totalrecoverablenormal_100/N_totalobservablenormal_100)*100
wholerecoverypercent_normal_1000 = (N_totalrecoverablenormal_1000/N_totalobservablenormal_1000)*100
sigmanormal = ((N_totalrecoverablenormal**(1/2))/N_totalobservablenormal)*100
sigmanormal_03 = ((N_totalrecoverablenormal_03**(1/2))/N_totalobservablenormal_03)*100
sigmanormal_1 = ((N_totalrecoverablenormal_1**(1/2))/N_totalobservablenormal_1)*100
sigmanormal_10 = ((N_totalrecoverablenormal_10**(1/2))/N_totalobservablenormal_10)*100
sigmanormal_30 = ((N_totalrecoverablenormal_30**(1/2))/N_totalobservablenormal_30)*100
sigmanormal_100 = ((N_totalrecoverablenormal_100**(1/2))/N_totalobservablenormal_100)*100
sigmanormal_1000 = ((N_totalrecoverablenormal_1000**(1/2))/N_totalobservablenormal_1000)*100
overallrecoverypercent_normal = (N_totalrecoverablenormal/N_totalnormal)*100
overallrecoverypercent_normal_03 = (N_totalrecoverablenormal_03/N_totalnormal_03)*100
overallrecoverypercent_normal_1 = (N_totalrecoverablenormal_1/N_totalnormal_1)*100
overallrecoverypercent_normal_10 = (N_totalrecoverablenormal_10/N_totalnormal_10)*100
overallrecoverypercent_normal_30 = (N_totalrecoverablenormal_30/N_totalnormal_30)*100
overallrecoverypercent_normal_100 = (N_totalrecoverablenormal_100/N_totalnormal_100)*100
overallrecoverypercent_normal_1000 = (N_totalrecoverablenormal_1000/N_totalnormal_1000)*100
overallsigmanormal = ((N_totalrecoverablenormal**(1/2))/N_totalnormal)*100
overallsigmanormal_03 = ((N_totalrecoverablenormal_03**(1/2))/N_totalnormal_03)*100
overallsigmanormal_1 = ((N_totalrecoverablenormal_1**(1/2))/N_totalnormal_1)*100
overallsigmanormal_10 = ((N_totalrecoverablenormal_10**(1/2))/N_totalnormal_10)*100
overallsigmanormal_30 = ((N_totalrecoverablenormal_30**(1/2))/N_totalnormal_30)*100
overallsigmanormal_100 = ((N_totalrecoverablenormal_100**(1/2))/N_totalnormal_100)*100
overallsigmanormal_1000 = ((N_totalrecoverablenormal_1000**(1/2))/N_totalnormal_1000)*100


wholerecoverypercent_normal22 = (N_totalrecoverablenormal22/N_totalobservablenormal22)*100
wholerecoverypercent_normal22_03 = (N_totalrecoverablenormal22_03/N_totalobservablenormal22_03)*100
wholerecoverypercent_normal22_1 = (N_totalrecoverablenormal22_1/N_totalobservablenormal22_1)*100
wholerecoverypercent_normal22_10 = (N_totalrecoverablenormal22_10/N_totalobservablenormal22_10)*100
wholerecoverypercent_normal22_30 = (N_totalrecoverablenormal22_30/N_totalobservablenormal22_30)*100
wholerecoverypercent_normal22_100 = (N_totalrecoverablenormal22_100/N_totalobservablenormal22_100)*100
wholerecoverypercent_normal22_1000 = (N_totalrecoverablenormal22_1000/N_totalobservablenormal22_1000)*100
sigmanormal22 = ((N_totalrecoverablenormal22**(1/2))/N_totalobservablenormal22)*100
sigmanormal22_03 = ((N_totalrecoverablenormal22_03**(1/2))/N_totalobservablenormal22_03)*100
sigmanormal22_1 = ((N_totalrecoverablenormal22_1**(1/2))/N_totalobservablenormal22_1)*100
sigmanormal22_10 = ((N_totalrecoverablenormal22_10**(1/2))/N_totalobservablenormal22_10)*100
sigmanormal22_30 = ((N_totalrecoverablenormal22_30**(1/2))/N_totalobservablenormal22_30)*100
sigmanormal22_100 = ((N_totalrecoverablenormal22_100**(1/2))/N_totalobservablenormal22_100)*100
sigmanormal22_1000 = ((N_totalrecoverablenormal22_1000**(1/2))/N_totalobservablenormal22_1000)*100
overallrecoverypercent_normal22 = (N_totalrecoverablenormal22/N_totalnormal22)*100
overallrecoverypercent_normal22_03 = (N_totalrecoverablenormal22_03/N_totalnormal22_03)*100
overallrecoverypercent_normal22_1 = (N_totalrecoverablenormal22_1/N_totalnormal22_1)*100
overallrecoverypercent_normal22_10 = (N_totalrecoverablenormal22_10/N_totalnormal22_10)*100
overallrecoverypercent_normal22_30 = (N_totalrecoverablenormal22_30/N_totalnormal22_30)*100
overallrecoverypercent_normal22_100 = (N_totalrecoverablenormal22_100/N_totalnormal22_100)*100
overallrecoverypercent_normal22_1000 = (N_totalrecoverablenormal22_1000/N_totalnormal22_1000)*100
overallsigmanormal22 = ((N_totalrecoverablenormal22**(1/2))/N_totalnormal22)*100
overallsigmanormal22_03 = ((N_totalrecoverablenormal22_03**(1/2))/N_totalnormal22_03)*100
overallsigmanormal22_1 = ((N_totalrecoverablenormal22_1**(1/2))/N_totalnormal22_1)*100
overallsigmanormal22_10 = ((N_totalrecoverablenormal22_10**(1/2))/N_totalnormal22_10)*100
overallsigmanormal22_30 = ((N_totalrecoverablenormal22_30**(1/2))/N_totalnormal22_30)*100
overallsigmanormal22_100 = ((N_totalrecoverablenormal22_100**(1/2))/N_totalnormal22_100)*100
overallsigmanormal22_1000 = ((N_totalrecoverablenormal22_1000**(1/2))/N_totalnormal22_1000)*100


wholerecoverypercent_normal195 = (N_totalrecoverablenormal195/N_totalobservablenormal195)*100
wholerecoverypercent_normal195_03 = (N_totalrecoverablenormal195_03/N_totalobservablenormal195_03)*100
wholerecoverypercent_normal195_1 = (N_totalrecoverablenormal195_1/N_totalobservablenormal195_1)*100
wholerecoverypercent_normal195_10 = (N_totalrecoverablenormal195_10/N_totalobservablenormal195_10)*100
wholerecoverypercent_normal195_30 = (N_totalrecoverablenormal195_30/N_totalobservablenormal195_30)*100
wholerecoverypercent_normal195_100 = (N_totalrecoverablenormal195_100/N_totalobservablenormal195_100)*100
wholerecoverypercent_normal195_1000 = (N_totalrecoverablenormal195_1000/N_totalobservablenormal195_1000)*100
sigmanormal195 = ((N_totalrecoverablenormal195**(1/2))/N_totalobservablenormal195)*100
sigmanormal195_03 = ((N_totalrecoverablenormal195_03**(1/2))/N_totalobservablenormal195_03)*100
sigmanormal195_1 = ((N_totalrecoverablenormal195_1**(1/2))/N_totalobservablenormal195_1)*100
sigmanormal195_10 = ((N_totalrecoverablenormal195_10**(1/2))/N_totalobservablenormal195_10)*100
sigmanormal195_30 = ((N_totalrecoverablenormal195_30**(1/2))/N_totalobservablenormal195_30)*100
sigmanormal195_100 = ((N_totalrecoverablenormal195_100**(1/2))/N_totalobservablenormal195_100)*100
sigmanormal195_1000 = ((N_totalrecoverablenormal195_1000**(1/2))/N_totalobservablenormal195_1000)*100
overallrecoverypercent_normal195 = (N_totalrecoverablenormal195/N_totalnormal195)*100
overallrecoverypercent_normal195_03 = (N_totalrecoverablenormal195_03/N_totalnormal195_03)*100
overallrecoverypercent_normal195_1 = (N_totalrecoverablenormal195_1/N_totalnormal195_1)*100
overallrecoverypercent_normal195_10 = (N_totalrecoverablenormal195_10/N_totalnormal195_10)*100
overallrecoverypercent_normal195_30 = (N_totalrecoverablenormal195_30/N_totalnormal195_30)*100
overallrecoverypercent_normal195_100 = (N_totalrecoverablenormal195_100/N_totalnormal195_100)*100
overallrecoverypercent_normal195_1000 = (N_totalrecoverablenormal195_1000/N_totalnormal195_1000)*100
overallsigmanormal195 = ((N_totalrecoverablenormal195**(1/2))/N_totalnormal195)*100
overallsigmanormal195_03 = ((N_totalrecoverablenormal195_03**(1/2))/N_totalnormal195_03)*100
overallsigmanormal195_1 = ((N_totalrecoverablenormal195_1**(1/2))/N_totalnormal195_1)*100
overallsigmanormal195_10 = ((N_totalrecoverablenormal195_10**(1/2))/N_totalnormal195_10)*100
overallsigmanormal195_30 = ((N_totalrecoverablenormal195_30**(1/2))/N_totalnormal195_30)*100
overallsigmanormal195_100 = ((N_totalrecoverablenormal195_100**(1/2))/N_totalnormal195_100)*100
overallsigmanormal195_1000 = ((N_totalrecoverablenormal195_1000**(1/2))/N_totalnormal195_1000)*100\





print("N_totalnormal = ", N_totalnormal, "and in log = ", np.log10(N_totalnormal), "**** N_totalobservablenormal = ", N_totalobservablenormal, "and in log = ", np.log10(N_totalobservablenormal), "**** N_totalrecoverablenormal = ", N_totalrecoverablenormal, "and in log = ", np.log10(N_totalrecoverablenormal))
print("N_totalnormal_03 = ", N_totalnormal_03, "and in log = ", np.log10(N_totalnormal_03), "**** N_totalobservablenormal_03 = ", N_totalobservablenormal_03, "and in log = ", np.log10(N_totalobservablenormal_03), "**** N_totalrecoverablenormal_03 = ", N_totalrecoverablenormal_03, "and in log = ", np.log10(N_totalrecoverablenormal_03))
print("N_totalnormal_1 = ", N_totalnormal_1, "and in log = ", np.log10(N_totalnormal_1), "**** N_totalobservablenormal_1 = ", N_totalobservablenormal_1, "and in log = ", np.log10(N_totalobservablenormal_1), "**** N_totalrecoverablenormal_1 = ", N_totalrecoverablenormal_1, "and in log = ", np.log10(N_totalrecoverablenormal_1))
print("N_totalnormal_10 = ", N_totalnormal_10, "and in log = ", np.log10(N_totalnormal_10), "**** N_totalobservablenormal_10 = ", N_totalobservablenormal_10, "and in log = ", np.log10(N_totalobservablenormal_10), "**** N_totalrecoverablenormal_10 = ", N_totalrecoverablenormal_10, "and in log = ", np.log10(N_totalrecoverablenormal_10))
print("N_totalnormal_30 = ", N_totalnormal_30, "and in log = ", np.log10(N_totalnormal_30), "**** N_totalobservablenormal_30 = ", N_totalobservablenormal_30, "and in log = ", np.log10(N_totalobservablenormal_30), "**** N_totalrecoverablenormal_30 = ", N_totalrecoverablenormal_30, "and in log = ", np.log10(N_totalrecoverablenormal_30))
print("N_totalnormal_100 = ", N_totalnormal_100, "and in log = ", np.log10(N_totalnormal_100), "**** N_totalobservablenormal_100 = ", N_totalobservablenormal_100, "and in log = ", np.log10(N_totalobservablenormal_100), "**** N_totalrecoverablenormal_100 = ", N_totalrecoverablenormal_100, "and in log = ", np.log10(N_totalrecoverablenormal_100))
print("N_totalnormal_1000 = ", N_totalnormal_1000, "and in log = ", np.log10(N_totalnormal_1000), "**** N_totalobservablenormal_1000 = ", N_totalobservablenormal_1000, "and in log = ", np.log10(N_totalobservablenormal_1000), "**** N_totalrecoverablenormal_1000 = ", N_totalrecoverablenormal_1000, "and in log = ", np.log10(N_totalrecoverablenormal_1000))

print("********************************")

print("wholerecoverypercent_normal = $", wholerecoverypercent_normal, "/pm", sigmanormal, "$")
print("wholerecoverypercent_normal_03 = $", wholerecoverypercent_normal_03, "/pm", sigmanormal_03, "$")
print("wholerecoverypercent_normal_1 = $", wholerecoverypercent_normal_1, "/pm", sigmanormal_1, "$")
print("wholerecoverypercent_normal_10 = $", wholerecoverypercent_normal_10, "/pm", sigmanormal_10, "$")
print("wholerecoverypercent_normal_30 = $", wholerecoverypercent_normal_03, "/pm", sigmanormal_30, "$")
print("wholerecoverypercent_normal_100 = $", wholerecoverypercent_normal_100, "/pm", sigmanormal_100, "$")
print("wholerecoverypercent_normal_1000 = $", wholerecoverypercent_normal_1000, "/pm", sigmanormal_1000, "$")

print("********************************")

print("overallrecoverypercent_normal = $", overallrecoverypercent_normal, "/pm", sigmanormal)
print("overallrecoverypercent_normal_03 = $", overallrecoverypercent_normal_03, "/pm", sigmanormal_03)
print("overallrecoverypercent_normal_1 = $", overallrecoverypercent_normal_1, "/pm", sigmanormal_1)
print("overallrecoverypercent_normal_10 = $", overallrecoverypercent_normal_10, "/pm", sigmanormal_10)
print("overallrecoverypercent_normal_30 = $", overallrecoverypercent_normal_03, "/pm", sigmanormal_30)
print("overallrecoverypercent_normal_100 = $", overallrecoverypercent_normal_100, "/pm", sigmanormal_100)
print("overallrecoverypercent_normal_1000 = $", overallrecoverypercent_normal_1000, "/pm", sigmanormal_1000)



print("################################")

print("N_totalnormal22 = ", N_totalnormal22, "and in log = ", np.log10(N_totalnormal22), "**** N_totalobservablenormal22 = ", N_totalobservablenormal22, "and in log = ", np.log10(N_totalobservablenormal22), "**** N_totalrecoverablenormal22 = ", N_totalrecoverablenormal22, "and in log = ", np.log10(N_totalrecoverablenormal22))
print("N_totalnormal22_03 = ", N_totalnormal22_03, "and in log = ", np.log10(N_totalnormal22_03), "**** N_totalobservablenormal22_03 = ", N_totalobservablenormal22_03, "and in log = ", np.log10(N_totalobservablenormal22_03), "**** N_totalrecoverablenormal22_03 = ", N_totalrecoverablenormal22_03, "and in log = ", np.log10(N_totalrecoverablenormal22_03))
print("N_totalnormal22_1 = ", N_totalnormal22_1, "and in log = ", np.log10(N_totalnormal22_1), "**** N_totalobservablenormal22_1 = ", N_totalobservablenormal22_1, "and in log = ", np.log10(N_totalobservablenormal22_1), "**** N_totalrecoverablenormal22_1 = ", N_totalrecoverablenormal22_1, "and in log = ", np.log10(N_totalrecoverablenormal22_1))
print("N_totalnormal22_10 = ", N_totalnormal22_10, "and in log = ", np.log10(N_totalnormal22_10), "**** N_totalobservablenormal22_10 = ", N_totalobservablenormal22_10, "and in log = ", np.log10(N_totalobservablenormal22_10), "**** N_totalrecoverablenormal22_10 = ", N_totalrecoverablenormal22_10, "and in log = ", np.log10(N_totalrecoverablenormal22_10))
print("N_totalnormal22_30 = ", N_totalnormal22_30, "and in log = ", np.log10(N_totalnormal22_30), "**** N_totalobservablenormal22_30 = ", N_totalobservablenormal22_30, "and in log = ", np.log10(N_totalobservablenormal22_30), "**** N_totalrecoverablenormal22_30 = ", N_totalrecoverablenormal22_30, "and in log = ", np.log10(N_totalrecoverablenormal22_30))
print("N_totalnormal22_100 = ", N_totalnormal22_100, "and in log = ", np.log10(N_totalnormal22_100), "**** N_totalobservablenormal22_100 = ", N_totalobservablenormal22_100, "and in log = ", np.log10(N_totalobservablenormal22_100), "**** N_totalrecoverablenormal22_100 = ", N_totalrecoverablenormal22_100, "and in log = ", np.log10(N_totalrecoverablenormal22_100))
print("N_totalnormal22_1000 = ", N_totalnormal22_1000, "and in log = ", np.log10(N_totalnormal22_1000), "**** N_totalobservablenormal22_1000 = ", N_totalobservablenormal22_1000, "and in log = ", np.log10(N_totalobservablenormal22_1000), "**** N_totalrecoverablenormal22_1000 = ", N_totalrecoverablenormal22_1000, "and in log = ", np.log10(N_totalrecoverablenormal22_1000))

print("********************************")

print("wholerecoverypercent_normal22 = $", wholerecoverypercent_normal22, "/pm", sigmanormal22, "$")
print("wholerecoverypercent_normal22_03 = $", wholerecoverypercent_normal22_03, "/pm", sigmanormal22_03, "$")
print("wholerecoverypercent_normal22_1 = $", wholerecoverypercent_normal22_1, "/pm", sigmanormal22_1, "$")
print("wholerecoverypercent_normal22_10 = $", wholerecoverypercent_normal22_10, "/pm", sigmanormal22_10, "$")
print("wholerecoverypercent_normal22_30 = $", wholerecoverypercent_normal22_03, "/pm", sigmanormal22_30, "$")
print("wholerecoverypercent_normal22_100 = $", wholerecoverypercent_normal22_100, "/pm", sigmanormal22_100, "$")
print("wholerecoverypercent_normal22_1000 = $", wholerecoverypercent_normal22_1000, "/pm", sigmanormal22_1000, "$")

print("********************************")

print("overallrecoverypercent_normal22 = $", overallrecoverypercent_normal22, "/pm", sigmanormal22, "$")
print("overallrecoverypercent_normal22_03 = $", overallrecoverypercent_normal22_03, "/pm", sigmanormal22_03, "$")
print("overallrecoverypercent_normal22_1 = $", overallrecoverypercent_normal22_1, "/pm", sigmanormal22_1, "$")
print("overallrecoverypercent_normal22_10 = $", overallrecoverypercent_normal22_10, "/pm", sigmanormal22_10, "$")
print("overallrecoverypercent_normal22_30 = $", overallrecoverypercent_normal22_03, "/pm", sigmanormal22_30, "$")
print("overallrecoverypercent_normal22_100 = $", overallrecoverypercent_normal22_100, "/pm", sigmanormal22_100, "$")
print("overallrecoverypercent_normal22_1000 = $", overallrecoverypercent_normal22_1000, "/pm", sigmanormal22_1000, "$")


print("###############################")

print("N_totalnormal195 = ", N_totalnormal195, "and in log = ", np.log10(N_totalnormal195), "**** N_totalobservablenormal195 = ", N_totalobservablenormal195, "and in log = ", np.log10(N_totalobservablenormal195), "**** N_totalrecoverablenormal195 = ", N_totalrecoverablenormal195, "and in log = ", np.log10(N_totalrecoverablenormal195))
print("N_totalnormal195_03 = ", N_totalnormal195_03, "and in log = ", np.log10(N_totalnormal195_03), "**** N_totalobservablenormal195_03 = ", N_totalobservablenormal195_03, "and in log = ", np.log10(N_totalobservablenormal195_03), "**** N_totalrecoverablenormal195_03 = ", N_totalrecoverablenormal195_03, "and in log = ", np.log10(N_totalrecoverablenormal195_03))
print("N_totalnormal195_1 = ", N_totalnormal195_1, "and in log = ", np.log10(N_totalnormal195_1), "**** N_totalobservablenormal195_1 = ", N_totalobservablenormal195_1, "and in log = ", np.log10(N_totalobservablenormal195_1), "**** N_totalrecoverablenormal195_1 = ", N_totalrecoverablenormal195_1, "and in log = ", np.log10(N_totalrecoverablenormal195_1))
print("N_totalnormal195_10 = ", N_totalnormal195_10, "and in log = ", np.log10(N_totalnormal195_10), "**** N_totalobservablenormal195_10 = ", N_totalobservablenormal195_10, "and in log = ", np.log10(N_totalobservablenormal195_10), "**** N_totalrecoverablenormal195_10 = ", N_totalrecoverablenormal195_10, "and in log = ", np.log10(N_totalrecoverablenormal195_10))
print("N_totalnormal195_30 = ", N_totalnormal195_30, "and in log = ", np.log10(N_totalnormal195_30), "**** N_totalobservablenormal195_30 = ", N_totalobservablenormal195_30, "and in log = ", np.log10(N_totalobservablenormal195_30), "**** N_totalrecoverablenormal195_30 = ", N_totalrecoverablenormal195_30, "and in log = ", np.log10(N_totalrecoverablenormal195_30))
print("N_totalnormal195_100 = ", N_totalnormal195_100, "and in log = ", np.log10(N_totalnormal195_100), "**** N_totalobservablenormal195_100 = ", N_totalobservablenormal195_100, "and in log = ", np.log10(N_totalobservablenormal195_100), "**** N_totalrecoverablenormal195_100 = ", N_totalrecoverablenormal195_100, "and in log = ", np.log10(N_totalrecoverablenormal195_100))
print("N_totalnormal195_1000 = ", N_totalnormal195_1000, "and in log = ", np.log10(N_totalnormal195_1000), "**** N_totalobservablenormal195_1000 = ", N_totalobservablenormal195_1000, "and in log = ", np.log10(N_totalobservablenormal195_1000), "**** N_totalrecoverablenormal195_1000 = ", N_totalrecoverablenormal195_1000, "and in log = ", np.log10(N_totalrecoverablenormal195_1000))

print("********************************")

print("wholerecoverypercent_normal195 = $", wholerecoverypercent_normal195, "/pm", sigmanormal195, "$")
print("wholerecoverypercent_normal195_03 = $", wholerecoverypercent_normal195_03, "/pm", sigmanormal195_03, "$")
print("wholerecoverypercent_normal195_1 = $", wholerecoverypercent_normal195_1, "/pm", sigmanormal195_1, "$")
print("wholerecoverypercent_normal195_10 = $", wholerecoverypercent_normal195_10, "/pm", sigmanormal195_10, "$")
print("wholerecoverypercent_normal195_30 = $", wholerecoverypercent_normal195_03, "/pm", sigmanormal195_30, "$")
print("wholerecoverypercent_normal195_100 = $", wholerecoverypercent_normal195_100, "/pm", sigmanormal195_100, "$")
print("wholerecoverypercent_normal195_1000 = $", wholerecoverypercent_normal195_1000, "/pm", sigmanormal195_1000, "$")

print("********************************")

print("overallrecoverypercent_normal195 = $", overallrecoverypercent_normal195, "/pm", sigmanormal195, "$")
print("overallrecoverypercent_normal195_03 = $", overallrecoverypercent_normal195_03, "/pm", sigmanormal195_03, "$")
print("overallrecoverypercent_normal195_1 = $", overallrecoverypercent_normal195_1, "/pm", sigmanormal195_1, "$")
print("overallrecoverypercent_normal195_10 = $", overallrecoverypercent_normal195_10, "/pm", sigmanormal195_10, "$")
print("overallrecoverypercent_normal195_30 = $", overallrecoverypercent_normal195_03, "/pm", sigmanormal195_30, "$")
print("overallrecoverypercent_normal195_100 = $", overallrecoverypercent_normal195_100, "/pm", sigmanormal195_100, "$")
print("overallrecoverypercent_normal195_1000 = $", overallrecoverypercent_normal195_1000, "/pm", sigmanormal195_1000, "$")

print("#############################")





print("binarypercent_22 = $", (N_totalnormal22/N_totalnormal)*100, "/pm", ((N_totalnormal22**(1/2))/N_totalnormal)*100, "$")
print("binarypercent_195 = $", (N_totalnormal195/N_totalnormal)*100, "/pm", ((N_totalnormal195**(1/2))/N_totalnormal)*100, "$")

print("binarypercent_03 = $", (N_totalnormal_03/N_totalnormal)*100, "/pm", ((N_totalnormal_03**(1/2))/N_totalnormal)*100, "$")
print("binarypercent_1 = $", (N_totalnormal_1/N_totalnormal)*100, "/pm", ((N_totalnormal_1**(1/2))/N_totalnormal)*100, "$")
print("binarypercent_10 = $", (N_totalnormal_10/N_totalnormal)*100, "/pm", ((N_totalnormal_10**(1/2))/N_totalnormal)*100, "$")
print("binarypercent_30 = $", (N_totalnormal_30/N_totalnormal)*100, "/pm", ((N_totalnormal_30**(1/2))/N_totalnormal)*100, "$")
print("binarypercent_100 = $", (N_totalnormal_100/N_totalnormal)*100, "/pm", ((N_totalnormal_100**(1/2))/N_totalnormal)*100, "$")
print("binarypercent_1000 = $", (N_totalnormal_1000/N_totalnormal)*100, "/pm", ((N_totalnormal_1000**(1/2))/N_totalnormal)*100, "$")

print("observablepercent_03 = $", (N_totalobservablenormal_03/N_totalnormal_03)*100, "/pm", ((N_totalobservablenormal_03**(1/2))/N_totalnormal_03)*100, "$")
print("observablepercent_1 = $", (N_totalobservablenormal_1/N_totalnormal_1)*100, "/pm", ((N_totalobservablenormal_1**(1/2))/N_totalnormal_1)*100, "$")
print("observablepercent_10 = $", (N_totalobservablenormal_10/N_totalnormal_10)*100, "/pm", ((N_totalobservablenormal_10**(1/2))/N_totalnormal_10)*100, "$")
print("observablepercent_30 = $", (N_totalobservablenormal_30/N_totalnormal_30)*100, "/pm", ((N_totalobservablenormal_30**(1/2))/N_totalnormal_30)*100, "$")
print("observablepercent_100 = $", (N_totalobservablenormal_100/N_totalnormal_100)*100, "/pm", ((N_totalobservablenormal_100**(1/2))/N_totalnormal_100)*100, "$")
print("observablepercent_1000 = $", (N_totalobservablenormal_1000/N_totalnormal_1000)*100, "/pm", ((N_totalobservablenormal_1000**(1/2))/N_totalnormal_1000)*100, "$")

print("observablepercent = $", (N_totalobservablenormal/N_totalnormal)*100, "/pm", ((N_totalobservablenormal**(1/2))/N_totalnormal)*100, "$")
print("observablepercent22 = $", (N_totalobservablenormal22/N_totalnormal22)*100, "/pm", ((N_totalobservablenormal22**(1/2))/N_totalnormal22)*100, "$")
print("observablepercent195 = $", (N_totalobservablenormal195/N_totalnormal195)*100, "/pm", ((N_totalobservablenormal195**(1/2))/N_totalnormal195)*100, "$")




for filefast_ in sorted(allFiles_fast):

	filename = filefast_[69:] #when file path no longer has /old in it, will be filefast_[65:]
	fileid = filename.strip('output_file.csv')
	print ("I'm starting " + fileid)


	datfast = pd.read_csv(filefast_, sep = ',', header=2)
	PeriodIn = datfast['p']		 # input period -- 'p' in data file

	##########################################################

	datfast1 = pd.read_csv(filefast_, sep = ',', header=0, nrows=1)
	N_tri = datfast1["NstarsTRILEGAL"][0]
	Nall = len(PeriodIn)


	m1hAll0, m1b = np.histogram(datfast["m1"], bins=mbins)
	dm1 = np.diff(m1b)
	m1val = m1b[:-1] + dm1/2.
	fb = np.sum(m1hAll0/Nall*fbFit(m1val))

	N_mult = N_tri*fb

	##########################################################

	if len(PeriodIn) == 0.:
		continue
	if N_tri == 0:
		continue
	else:
		PeriodOut = datfast['LSM_PERIOD'] #LSM_PERIOD in data file
		appMagMean = datfast['appMagMean'] #apparent magnitude, will use to make cuts for 24 (default), 22, and then Kepler's range (?? -- brighter than LSST can manage-- to 19) OR 19.5 (SNR = 10)
		
		observable = datfast.loc[PeriodOut != -999].index
		observable_03 = datfast.loc[(PeriodIn <= 0.3) & (PeriodOut != -999)].index
		observable_1 = datfast.loc[(PeriodIn <= 1) & (PeriodOut != -999)].index
		observable_10 = datfast.loc[(PeriodIn <= 10) & (PeriodOut != -999)].index
		observable_30 = datfast.loc[(PeriodIn <= 30) & (PeriodOut != -999)].index
		observable_100 = datfast.loc[(PeriodIn <= 100) & (PeriodOut != -999)].index
		observable_1000 = datfast.loc[(PeriodIn <= 1000) & (PeriodOut != -999)].index
		
		observable_22 = datfast.loc[(PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_03_22 = datfast.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_1_22 = datfast.loc[(PeriodIn <= 1) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_10_22 = datfast.loc[(PeriodIn <= 10) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_30_22 = datfast.loc[(PeriodIn <= 30) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_100_22 = datfast.loc[(PeriodIn <= 100) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_1000_22 = datfast.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & (appMagMean <= 22.)].index

		observable_195 = datfast.loc[(PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_03_195 = datfast.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_1_195 = datfast.loc[(PeriodIn <= 1) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_10_195 = datfast.loc[(PeriodIn <= 10) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_30_195 = datfast.loc[(PeriodIn <= 30) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_100_195 = datfast.loc[(PeriodIn <= 100) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_1000_195 = datfast.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & (appMagMean <= 19.5)].index

		fullP = abs(PeriodOut - PeriodIn)/PeriodIn
		halfP = abs(PeriodOut - 0.5*PeriodIn)/(0.5*PeriodIn)
		twiceP = abs(PeriodOut - 2*PeriodIn)/(2*PeriodIn)

		recoverable = datfast.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_03 = datfast.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_1 = datfast.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_10 = datfast.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_30 = datfast.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_100 = datfast.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_1000 = datfast.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index

		recoverable_22 = datfast.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_03_22 = datfast.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_1_22 = datfast.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_10_22 = datfast.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_30_22 = datfast.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_100_22 = datfast.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_1000_22 = datfast.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index

		recoverable_195 = datfast.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_03_195 = datfast.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_1_195 = datfast.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_10_195 = datfast.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_30_195 = datfast.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_100_195 = datfast.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_1000_195 = datfast.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index

		P03 = datfast.loc[PeriodIn <= 0.3].index
		P1 = datfast.loc[PeriodIn <= 1].index
		P10 = datfast.loc[PeriodIn <= 10].index
		P30 = datfast.loc[PeriodIn <= 30].index
		P100 = datfast.loc[PeriodIn <= 100].index
		P1000 = datfast.loc[PeriodIn <= 1000].index

		P_22 = datfast.loc[appMagMean <= 22.].index
		P03_22 = datfast.loc[(PeriodIn <= 0.3) & (appMagMean <= 22.)].index
		P1_22 = datfast.loc[(PeriodIn <= 1) & (appMagMean <= 22.)].index
		P10_22 = datfast.loc[(PeriodIn <= 10) & (appMagMean <= 22.)].index
		P30_22 = datfast.loc[(PeriodIn <= 30) & (appMagMean <= 22.)].index
		P100_22 = datfast.loc[(PeriodIn <= 100) & (appMagMean <= 22.)].index
		P1000_22 = datfast.loc[(PeriodIn <= 1000) & (appMagMean <= 22.)].index

		P_195 = datfast.loc[appMagMean <= 19.5].index
		P03_195 = datfast.loc[(PeriodIn <= 0.3) & (appMagMean <= 19.5)].index
		P1_195 = datfast.loc[(PeriodIn <= 1) & (appMagMean <= 19.5)].index
		P10_195 = datfast.loc[(PeriodIn <= 10) & (appMagMean <= 19.5)].index
		P30_195 = datfast.loc[(PeriodIn <= 30) & (appMagMean <= 19.5)].index
		P100_195 = datfast.loc[(PeriodIn <= 100) & (appMagMean <= 19.5)].index
		P1000_195 = datfast.loc[(PeriodIn <= 1000) & (appMagMean <= 19.5)].index

		N_all = (len(PeriodIn)/len(PeriodIn))*N_mult
		N_all03 = (len(P03)/len(PeriodIn))*N_mult
		N_all1 = (len(P1)/len(PeriodIn))*N_mult
		N_all10 = (len(P10)/len(PeriodIn))*N_mult
		N_all30 = (len(P30)/len(PeriodIn))*N_mult
		N_all100 = (len(P100)/len(PeriodIn))*N_mult
		N_all1000 = (len(P1000)/len(PeriodIn))*N_mult

		N_all_22 = (len(P_22)/len(PeriodIn))*N_mult
		N_all03_22 = (len(P03_22)/len(PeriodIn))*N_mult
		N_all1_22 = (len(P1_22)/len(PeriodIn))*N_mult
		N_all10_22 = (len(P10_22)/len(PeriodIn))*N_mult
		N_all30_22 = (len(P30_22)/len(PeriodIn))*N_mult
		N_all100_22 = (len(P100_22)/len(PeriodIn))*N_mult
		N_all1000_22 = (len(P1000_22)/len(PeriodIn))*N_mult

		N_all_195 = (len(P_195)/len(PeriodIn))*N_mult
		N_all03_195 = (len(P03_195)/len(PeriodIn))*N_mult
		N_all1_195 = (len(P1_195)/len(PeriodIn))*N_mult
		N_all10_195 = (len(P10_195)/len(PeriodIn))*N_mult
		N_all30_195 = (len(P30_195)/len(PeriodIn))*N_mult
		N_all100_195 = (len(P100_195)/len(PeriodIn))*N_mult
		N_all1000_195 = (len(P1000_195)/len(PeriodIn))*N_mult

		N_obs = (len(observable)/len(PeriodIn))*N_mult
		N_obs03 = (len(observable_03)/len(PeriodIn))*N_mult
		N_obs1 = (len(observable_1)/len(PeriodIn))*N_mult
		N_obs10 = (len(observable_10)/len(PeriodIn))*N_mult
		N_obs30 = (len(observable_30)/len(PeriodIn))*N_mult
		N_obs100 = (len(observable_100)/len(PeriodIn))*N_mult
		N_obs1000 = (len(observable_1000)/len(PeriodIn))*N_mult

		N_obs_22 = (len(observable_22)/len(PeriodIn))*N_mult
		N_obs03_22 = (len(observable_03_22)/len(PeriodIn))*N_mult
		N_obs1_22 = (len(observable_1_22)/len(PeriodIn))*N_mult
		N_obs10_22 = (len(observable_10_22)/len(PeriodIn))*N_mult
		N_obs30_22 = (len(observable_30_22)/len(PeriodIn))*N_mult
		N_obs100_22 = (len(observable_100_22)/len(PeriodIn))*N_mult
		N_obs1000_22 = (len(observable_1000_22)/len(PeriodIn))*N_mult

		N_obs_195 = (len(observable_195)/len(PeriodIn))*N_mult
		N_obs03_195 = (len(observable_03_195)/len(PeriodIn))*N_mult
		N_obs1_195 = (len(observable_1_195)/len(PeriodIn))*N_mult
		N_obs10_195 = (len(observable_10_195)/len(PeriodIn))*N_mult
		N_obs30_195 = (len(observable_30_195)/len(PeriodIn))*N_mult
		N_obs100_195 = (len(observable_100_195)/len(PeriodIn))*N_mult
		N_obs1000_195 = (len(observable_1000_195)/len(PeriodIn))*N_mult

		N_rec = (len(recoverable)/len(PeriodIn))*N_mult
		N_rec03 = (len(recoverable_03)/len(PeriodIn))*N_mult
		N_rec1 = (len(recoverable_1)/len(PeriodIn))*N_mult
		N_rec10 = (len(recoverable_10)/len(PeriodIn))*N_mult
		N_rec30 = (len(recoverable_30)/len(PeriodIn))*N_mult
		N_rec100 = (len(recoverable_100)/len(PeriodIn))*N_mult
		N_rec1000 = (len(recoverable_1000)/len(PeriodIn))*N_mult

		N_rec_22 = (len(recoverable_22)/len(PeriodIn))*N_mult
		N_rec03_22 = (len(recoverable_03_22)/len(PeriodIn))*N_mult
		N_rec1_22 = (len(recoverable_1_22)/len(PeriodIn))*N_mult
		N_rec10_22 = (len(recoverable_10_22)/len(PeriodIn))*N_mult
		N_rec30_22 = (len(recoverable_30_22)/len(PeriodIn))*N_mult
		N_rec100_22 = (len(recoverable_100_22)/len(PeriodIn))*N_mult
		N_rec1000_22 = (len(recoverable_1000_22)/len(PeriodIn))*N_mult

		N_rec_195 = (len(recoverable_195)/len(PeriodIn))*N_mult
		N_rec03_195 = (len(recoverable_03_195)/len(PeriodIn))*N_mult
		N_rec1_195 = (len(recoverable_1_195)/len(PeriodIn))*N_mult
		N_rec10_195 = (len(recoverable_10_195)/len(PeriodIn))*N_mult
		N_rec30_195 = (len(recoverable_30_195)/len(PeriodIn))*N_mult
		N_rec100_195 = (len(recoverable_100_195)/len(PeriodIn))*N_mult
		N_rec1000_195 = (len(recoverable_1000_195)/len(PeriodIn))*N_mult

		N_totalfast_array.append(float(N_all))
		N_totalobservablefast_array.append(float(N_obs))
		N_totalrecoverablefast_array.append(float(N_rec))

		N_totalfast_array_03.append(float(N_all03))
		N_totalobservablefast_array_03.append(float(N_obs03))
		N_totalrecoverablefast_array_03.append(float(N_rec03))

		N_totalfast_array_1.append(float(N_all1))
		N_totalobservablefast_array_1.append(float(N_obs1))
		N_totalrecoverablefast_array_1.append(float(N_rec1))

		N_totalfast_array_10.append(float(N_all10))
		N_totalobservablefast_array_10.append(float(N_obs10))
		N_totalrecoverablefast_array_10.append(float(N_rec10))

		N_totalfast_array_30.append(float(N_all30))
		N_totalobservablefast_array_30.append(float(N_obs30))
		N_totalrecoverablefast_array_30.append(float(N_rec30))

		N_totalfast_array_100.append(float(N_all100))
		N_totalobservablefast_array_100.append(float(N_obs100))
		N_totalrecoverablefast_array_100.append(float(N_rec100))

		N_totalfast_array_1000.append(float(N_all1000))
		N_totalobservablefast_array_1000.append(float(N_obs1000))
		N_totalrecoverablefast_array_1000.append(float(N_rec1000))

		N_totalfast22_array.append(float(N_all_22))
		N_totalobservablefast22_array.append(float(N_obs_22))
		N_totalrecoverablefast22_array.append(float(N_rec_22))

		N_totalfast22_array_03.append(float(N_all03_22))
		N_totalobservablefast22_array_03.append(float(N_obs03_22))
		N_totalrecoverablefast22_array_03.append(float(N_rec03_22))

		N_totalfast22_array_1.append(float(N_all1_22))
		N_totalobservablefast22_array_1.append(float(N_obs1_22))
		N_totalrecoverablefast22_array_1.append(float(N_rec1_22))

		N_totalfast22_array_10.append(float(N_all10_22))
		N_totalobservablefast22_array_10.append(float(N_obs10_22))
		N_totalrecoverablefast22_array_10.append(float(N_rec10_22))

		N_totalfast22_array_30.append(float(N_all30_22))
		N_totalobservablefast22_array_30.append(float(N_obs30_22))
		N_totalrecoverablefast22_array_30.append(float(N_rec30_22))

		N_totalfast22_array_100.append(float(N_all100_22))
		N_totalobservablefast22_array_100.append(float(N_obs100_22))
		N_totalrecoverablefast22_array_100.append(float(N_rec100_22))

		N_totalfast22_array_1000.append(float(N_all1000_22))
		N_totalobservablefast22_array_1000.append(float(N_obs1000_22))
		N_totalrecoverablefast22_array_1000.append(float(N_rec1000_22))

		N_totalfast195_array.append(float(N_all_195))
		N_totalobservablefast195_array.append(float(N_obs_195))
		N_totalrecoverablefast195_array.append(float(N_rec_195))

		N_totalfast195_array_03.append(float(N_all03_195))
		N_totalobservablefast195_array_03.append(float(N_obs03_195))
		N_totalrecoverablefast195_array_03.append(float(N_rec03_195))

		N_totalfast195_array_1.append(float(N_all1_195))
		N_totalobservablefast195_array_1.append(float(N_obs1_195))
		N_totalrecoverablefast195_array_1.append(float(N_rec1_195))

		N_totalfast195_array_10.append(float(N_all10_195))
		N_totalobservablefast195_array_10.append(float(N_obs10_195))
		N_totalrecoverablefast195_array_10.append(float(N_rec10_195))

		N_totalfast195_array_30.append(float(N_all30_195))
		N_totalobservablefast195_array_30.append(float(N_obs30_195))
		N_totalrecoverablefast195_array_30.append(float(N_rec30_195))

		N_totalfast195_array_100.append(float(N_all100_195))
		N_totalobservablefast195_array_100.append(float(N_obs100_195))
		N_totalrecoverablefast195_array_100.append(float(N_rec100_195))

		N_totalfast195_array_1000.append(float(N_all1000_195))
		N_totalobservablefast195_array_1000.append(float(N_obs1000_195))
		N_totalrecoverablefast195_array_1000.append(float(N_rec1000_195))


N_totalfast = np.sum(N_totalfast_array)
N_totalfast_03 = np.sum(N_totalfast_array_03)
N_totalfast_1 = np.sum(N_totalfast_array_1)
N_totalfast_10 = np.sum(N_totalfast_array_10)
N_totalfast_30 = np.sum(N_totalfast_array_30)
N_totalfast_100 = np.sum(N_totalfast_array_100)
N_totalfast_1000 = np.sum(N_totalfast_array_1000)


N_totalobservablefast = np.sum(N_totalobservablefast_array)
N_totalobservablefast_03 = np.sum(N_totalobservablefast_array_03)
N_totalobservablefast_1 = np.sum(N_totalobservablefast_array_1)
N_totalobservablefast_10 = np.sum(N_totalobservablefast_array_10)
N_totalobservablefast_30 = np.sum(N_totalobservablefast_array_30)
N_totalobservablefast_100 = np.sum(N_totalobservablefast_array_100)
N_totalobservablefast_1000 = np.sum(N_totalobservablefast_array_1000)


N_totalrecoverablefast = np.sum(N_totalrecoverablefast_array)
N_totalrecoverablefast_03 = np.sum(N_totalrecoverablefast_array_03)
N_totalrecoverablefast_1 = np.sum(N_totalrecoverablefast_array_1)
N_totalrecoverablefast_10 = np.sum(N_totalrecoverablefast_array_10)
N_totalrecoverablefast_30 = np.sum(N_totalrecoverablefast_array_30)
N_totalrecoverablefast_100 = np.sum(N_totalrecoverablefast_array_100)
N_totalrecoverablefast_1000 = np.sum(N_totalrecoverablefast_array_1000)


N_totalfast22 = np.sum(N_totalfast22_array)
N_totalfast22_03 = np.sum(N_totalfast22_array_03)
N_totalfast22_1 = np.sum(N_totalfast22_array_1)
N_totalfast22_10 = np.sum(N_totalfast22_array_10)
N_totalfast22_30 = np.sum(N_totalfast22_array_30)
N_totalfast22_100 = np.sum(N_totalfast22_array_100)
N_totalfast22_1000 = np.sum(N_totalfast22_array_1000)


N_totalobservablefast22 = np.sum(N_totalobservablefast22_array)
N_totalobservablefast22_03 = np.sum(N_totalobservablefast22_array_03)
N_totalobservablefast22_1 = np.sum(N_totalobservablefast22_array_1)
N_totalobservablefast22_10 = np.sum(N_totalobservablefast22_array_10)
N_totalobservablefast22_30 = np.sum(N_totalobservablefast22_array_30)
N_totalobservablefast22_100 = np.sum(N_totalobservablefast22_array_100)
N_totalobservablefast22_1000 = np.sum(N_totalobservablefast22_array_1000)


N_totalrecoverablefast22 = np.sum(N_totalrecoverablefast22_array)
N_totalrecoverablefast22_03 = np.sum(N_totalrecoverablefast22_array_03)
N_totalrecoverablefast22_1 = np.sum(N_totalrecoverablefast22_array_1)
N_totalrecoverablefast22_10 = np.sum(N_totalrecoverablefast22_array_10)
N_totalrecoverablefast22_30 = np.sum(N_totalrecoverablefast22_array_30)
N_totalrecoverablefast22_100 = np.sum(N_totalrecoverablefast22_array_100)
N_totalrecoverablefast22_1000 = np.sum(N_totalrecoverablefast22_array_1000)

N_totalfast195 = np.sum(N_totalfast195_array)
N_totalfast195_03 = np.sum(N_totalfast195_array_03)
N_totalfast195_1 = np.sum(N_totalfast195_array_1)
N_totalfast195_10 = np.sum(N_totalfast195_array_10)
N_totalfast195_30 = np.sum(N_totalfast195_array_30)
N_totalfast195_100 = np.sum(N_totalfast195_array_100)
N_totalfast195_1000 = np.sum(N_totalfast195_array_1000)


N_totalobservablefast195 = np.sum(N_totalobservablefast195_array)
N_totalobservablefast195_03 = np.sum(N_totalobservablefast195_array_03)
N_totalobservablefast195_1 = np.sum(N_totalobservablefast195_array_1)
N_totalobservablefast195_10 = np.sum(N_totalobservablefast195_array_10)
N_totalobservablefast195_30 = np.sum(N_totalobservablefast195_array_30)
N_totalobservablefast195_100 = np.sum(N_totalobservablefast195_array_100)
N_totalobservablefast195_1000 = np.sum(N_totalobservablefast195_array_1000)


N_totalrecoverablefast195 = np.sum(N_totalrecoverablefast195_array)
N_totalrecoverablefast195_03 = np.sum(N_totalrecoverablefast195_array_03)
N_totalrecoverablefast195_1 = np.sum(N_totalrecoverablefast195_array_1)
N_totalrecoverablefast195_10 = np.sum(N_totalrecoverablefast195_array_10)
N_totalrecoverablefast195_30 = np.sum(N_totalrecoverablefast195_array_30)
N_totalrecoverablefast195_100 = np.sum(N_totalrecoverablefast195_array_100)
N_totalrecoverablefast195_1000 = np.sum(N_totalrecoverablefast195_array_1000)


wholerecoverypercent_fast = (N_totalrecoverablefast/N_totalobservablefast)*100
wholerecoverypercent_fast_03 = (N_totalrecoverablefast_03/N_totalobservablefast_03)*100
wholerecoverypercent_fast_1 = (N_totalrecoverablefast_1/N_totalobservablefast_1)*100
wholerecoverypercent_fast_10 = (N_totalrecoverablefast_10/N_totalobservablefast_10)*100
wholerecoverypercent_fast_30 = (N_totalrecoverablefast_30/N_totalobservablefast_30)*100
wholerecoverypercent_fast_100 = (N_totalrecoverablefast_100/N_totalobservablefast_100)*100
wholerecoverypercent_fast_1000 = (N_totalrecoverablefast_1000/N_totalobservablefast_1000)*100
sigmafast = ((N_totalrecoverablefast**(1/2))/N_totalobservablefast)*100
sigmafast_03 = ((N_totalrecoverablefast_03**(1/2))/N_totalobservablefast_03)*100
sigmafast_1 = ((N_totalrecoverablefast_1**(1/2))/N_totalobservablefast_1)*100
sigmafast_10 = ((N_totalrecoverablefast_10**(1/2))/N_totalobservablefast_10)*100
sigmafast_30 = ((N_totalrecoverablefast_30**(1/2))/N_totalobservablefast_30)*100
sigmafast_100 = ((N_totalrecoverablefast_100**(1/2))/N_totalobservablefast_100)*100
sigmafast_1000 = ((N_totalrecoverablefast_1000**(1/2))/N_totalobservablefast_1000)*100
overallrecoverypercent_fast = (N_totalrecoverablefast/N_totalfast)*100
overallrecoverypercent_fast_03 = (N_totalrecoverablefast_03/N_totalfast_03)*100
overallrecoverypercent_fast_1 = (N_totalrecoverablefast_1/N_totalfast_1)*100
overallrecoverypercent_fast_10 = (N_totalrecoverablefast_10/N_totalfast_10)*100
overallrecoverypercent_fast_30 = (N_totalrecoverablefast_30/N_totalfast_30)*100
overallrecoverypercent_fast_100 = (N_totalrecoverablefast_100/N_totalfast_100)*100
overallrecoverypercent_fast_1000 = (N_totalrecoverablefast_1000/N_totalfast_1000)*100
overallsigmafast = ((N_totalrecoverablefast**(1/2))/N_totalfast)*100
overallsigmafast_03 = ((N_totalrecoverablefast_03**(1/2))/N_totalfast_03)*100
overallsigmafast_1 = ((N_totalrecoverablefast_1**(1/2))/N_totalfast_1)*100
overallsigmafast_10 = ((N_totalrecoverablefast_10**(1/2))/N_totalfast_10)*100
overallsigmafast_30 = ((N_totalrecoverablefast_30**(1/2))/N_totalfast_30)*100
overallsigmafast_100 = ((N_totalrecoverablefast_100**(1/2))/N_totalfast_100)*100
overallsigmafast_1000 = ((N_totalrecoverablefast_1000**(1/2))/N_totalfast_1000)*100


wholerecoverypercent_fast22 = (N_totalrecoverablefast22/N_totalobservablefast22)*100
wholerecoverypercent_fast22_03 = (N_totalrecoverablefast22_03/N_totalobservablefast22_03)*100
wholerecoverypercent_fast22_1 = (N_totalrecoverablefast22_1/N_totalobservablefast22_1)*100
wholerecoverypercent_fast22_10 = (N_totalrecoverablefast22_10/N_totalobservablefast22_10)*100
wholerecoverypercent_fast22_30 = (N_totalrecoverablefast22_30/N_totalobservablefast22_30)*100
wholerecoverypercent_fast22_100 = (N_totalrecoverablefast22_100/N_totalobservablefast22_100)*100
wholerecoverypercent_fast22_1000 = (N_totalrecoverablefast22_1000/N_totalobservablefast22_1000)*100
sigmafast22 = ((N_totalrecoverablefast22**(1/2))/N_totalobservablefast22)*100
sigmafast22_03 = ((N_totalrecoverablefast22_03**(1/2))/N_totalobservablefast22_03)*100
sigmafast22_1 = ((N_totalrecoverablefast22_1**(1/2))/N_totalobservablefast22_1)*100
sigmafast22_10 = ((N_totalrecoverablefast22_10**(1/2))/N_totalobservablefast22_10)*100
sigmafast22_30 = ((N_totalrecoverablefast22_30**(1/2))/N_totalobservablefast22_30)*100
sigmafast22_100 = ((N_totalrecoverablefast22_100**(1/2))/N_totalobservablefast22_100)*100
sigmafast22_1000 = ((N_totalrecoverablefast22_1000**(1/2))/N_totalobservablefast22_1000)*100
overallrecoverypercent_fast22 = (N_totalrecoverablefast22/N_totalfast22)*100
overallrecoverypercent_fast22_03 = (N_totalrecoverablefast22_03/N_totalfast22_03)*100
overallrecoverypercent_fast22_1 = (N_totalrecoverablefast22_1/N_totalfast22_1)*100
overallrecoverypercent_fast22_10 = (N_totalrecoverablefast22_10/N_totalfast22_10)*100
overallrecoverypercent_fast22_30 = (N_totalrecoverablefast22_30/N_totalfast22_30)*100
overallrecoverypercent_fast22_100 = (N_totalrecoverablefast22_100/N_totalfast22_100)*100
overallrecoverypercent_fast22_1000 = (N_totalrecoverablefast22_1000/N_totalfast22_1000)*100
overallsigmafast22 = ((N_totalrecoverablefast22**(1/2))/N_totalfast22)*100
overallsigmafast22_03 = ((N_totalrecoverablefast22_03**(1/2))/N_totalfast22_03)*100
overallsigmafast22_1 = ((N_totalrecoverablefast22_1**(1/2))/N_totalfast22_1)*100
overallsigmafast22_10 = ((N_totalrecoverablefast22_10**(1/2))/N_totalfast22_10)*100
overallsigmafast22_30 = ((N_totalrecoverablefast22_30**(1/2))/N_totalfast22_30)*100
overallsigmafast22_100 = ((N_totalrecoverablefast22_100**(1/2))/N_totalfast22_100)*100
overallsigmafast22_1000 = ((N_totalrecoverablefast22_1000**(1/2))/N_totalfast22_1000)*100


wholerecoverypercent_fast195 = (N_totalrecoverablefast195/N_totalobservablefast195)*100
wholerecoverypercent_fast195_03 = (N_totalrecoverablefast195_03/N_totalobservablefast195_03)*100
wholerecoverypercent_fast195_1 = (N_totalrecoverablefast195_1/N_totalobservablefast195_1)*100
wholerecoverypercent_fast195_10 = (N_totalrecoverablefast195_10/N_totalobservablefast195_10)*100
wholerecoverypercent_fast195_30 = (N_totalrecoverablefast195_30/N_totalobservablefast195_30)*100
wholerecoverypercent_fast195_100 = (N_totalrecoverablefast195_100/N_totalobservablefast195_100)*100
wholerecoverypercent_fast195_1000 = (N_totalrecoverablefast195_1000/N_totalobservablefast195_1000)*100
sigmafast195 = ((N_totalrecoverablefast195**(1/2))/N_totalobservablefast195)*100
sigmafast195_03 = ((N_totalrecoverablefast195_03**(1/2))/N_totalobservablefast195_03)*100
sigmafast195_1 = ((N_totalrecoverablefast195_1**(1/2))/N_totalobservablefast195_1)*100
sigmafast195_10 = ((N_totalrecoverablefast195_10**(1/2))/N_totalobservablefast195_10)*100
sigmafast195_30 = ((N_totalrecoverablefast195_30**(1/2))/N_totalobservablefast195_30)*100
sigmafast195_100 = ((N_totalrecoverablefast195_100**(1/2))/N_totalobservablefast195_100)*100
sigmafast195_1000 = ((N_totalrecoverablefast195_1000**(1/2))/N_totalobservablefast195_1000)*100
overallrecoverypercent_fast195 = (N_totalrecoverablefast195/N_totalfast195)*100
overallrecoverypercent_fast195_03 = (N_totalrecoverablefast195_03/N_totalfast195_03)*100
overallrecoverypercent_fast195_1 = (N_totalrecoverablefast195_1/N_totalfast195_1)*100
overallrecoverypercent_fast195_10 = (N_totalrecoverablefast195_10/N_totalfast195_10)*100
overallrecoverypercent_fast195_30 = (N_totalrecoverablefast195_30/N_totalfast195_30)*100
overallrecoverypercent_fast195_100 = (N_totalrecoverablefast195_100/N_totalfast195_100)*100
overallrecoverypercent_fast195_1000 = (N_totalrecoverablefast195_1000/N_totalfast195_1000)*100
overallsigmafast195 = ((N_totalrecoverablefast195**(1/2))/N_totalfast195)*100
overallsigmafast195_03 = ((N_totalrecoverablefast195_03**(1/2))/N_totalfast195_03)*100
overallsigmafast195_1 = ((N_totalrecoverablefast195_1**(1/2))/N_totalfast195_1)*100
overallsigmafast195_10 = ((N_totalrecoverablefast195_10**(1/2))/N_totalfast195_10)*100
overallsigmafast195_30 = ((N_totalrecoverablefast195_30**(1/2))/N_totalfast195_30)*100
overallsigmafast195_100 = ((N_totalrecoverablefast195_100**(1/2))/N_totalfast195_100)*100
overallsigmafast195_1000 = ((N_totalrecoverablefast195_1000**(1/2))/N_totalfast195_1000)*100\





print("N_totalfast = ", N_totalfast, "and in log = ", np.log10(N_totalfast), "**** N_totalobservablefast = ", N_totalobservablefast, "and in log = ", np.log10(N_totalobservablefast), "**** N_totalrecoverablefast = ", N_totalrecoverablefast, "and in log = ", np.log10(N_totalrecoverablefast))
print("N_totalfast_03 = ", N_totalfast_03, "and in log = ", np.log10(N_totalfast_03), "**** N_totalobservablefast_03 = ", N_totalobservablefast_03, "and in log = ", np.log10(N_totalobservablefast_03), "**** N_totalrecoverablefast_03 = ", N_totalrecoverablefast_03, "and in log = ", np.log10(N_totalrecoverablefast_03))
print("N_totalfast_1 = ", N_totalfast_1, "and in log = ", np.log10(N_totalfast_1), "**** N_totalobservablefast_1 = ", N_totalobservablefast_1, "and in log = ", np.log10(N_totalobservablefast_1), "**** N_totalrecoverablefast_1 = ", N_totalrecoverablefast_1, "and in log = ", np.log10(N_totalrecoverablefast_1))
print("N_totalfast_10 = ", N_totalfast_10, "and in log = ", np.log10(N_totalfast_10), "**** N_totalobservablefast_10 = ", N_totalobservablefast_10, "and in log = ", np.log10(N_totalobservablefast_10), "**** N_totalrecoverablefast_10 = ", N_totalrecoverablefast_10, "and in log = ", np.log10(N_totalrecoverablefast_10))
print("N_totalfast_30 = ", N_totalfast_30, "and in log = ", np.log10(N_totalfast_30), "**** N_totalobservablefast_30 = ", N_totalobservablefast_30, "and in log = ", np.log10(N_totalobservablefast_30), "**** N_totalrecoverablefast_30 = ", N_totalrecoverablefast_30, "and in log = ", np.log10(N_totalrecoverablefast_30))
print("N_totalfast_100 = ", N_totalfast_100, "and in log = ", np.log10(N_totalfast_100), "**** N_totalobservablefast_100 = ", N_totalobservablefast_100, "and in log = ", np.log10(N_totalobservablefast_100), "**** N_totalrecoverablefast_100 = ", N_totalrecoverablefast_100, "and in log = ", np.log10(N_totalrecoverablefast_100))
print("N_totalfast_1000 = ", N_totalfast_1000, "and in log = ", np.log10(N_totalfast_1000), "**** N_totalobservablefast_1000 = ", N_totalobservablefast_1000, "and in log = ", np.log10(N_totalobservablefast_1000), "**** N_totalrecoverablefast_1000 = ", N_totalrecoverablefast_1000, "and in log = ", np.log10(N_totalrecoverablefast_1000))

print("********************************")

print("wholerecoverypercent_fast = $", wholerecoverypercent_fast, "/pm", sigmafast, "$")
print("wholerecoverypercent_fast_03 = $", wholerecoverypercent_fast_03, "/pm", sigmafast_03, "$")
print("wholerecoverypercent_fast_1 = $", wholerecoverypercent_fast_1, "/pm", sigmafast_1, "$")
print("wholerecoverypercent_fast_10 = $", wholerecoverypercent_fast_10, "/pm", sigmafast_10, "$")
print("wholerecoverypercent_fast_30 = $", wholerecoverypercent_fast_03, "/pm", sigmafast_30, "$")
print("wholerecoverypercent_fast_100 = $", wholerecoverypercent_fast_100, "/pm", sigmafast_100, "$")
print("wholerecoverypercent_fast_1000 = $", wholerecoverypercent_fast_1000, "/pm", sigmafast_1000, "$")

print("********************************")

print("overallrecoverypercent_fast = $", overallrecoverypercent_fast, "/pm", sigmafast, "$")
print("overallrecoverypercent_fast_03 = $", overallrecoverypercent_fast_03, "/pm", sigmafast_03, "$")
print("overallrecoverypercent_fast_1 = $", overallrecoverypercent_fast_1, "/pm", sigmafast_1, "$")
print("overallrecoverypercent_fast_10 = $", overallrecoverypercent_fast_10, "/pm", sigmafast_10, "$")
print("overallrecoverypercent_fast_30 = $", overallrecoverypercent_fast_03, "/pm", sigmafast_30, "$")
print("overallrecoverypercent_fast_100 = $", overallrecoverypercent_fast_100, "/pm", sigmafast_100, "$")
print("overallrecoverypercent_fast_1000 = $", overallrecoverypercent_fast_1000, "/pm", sigmafast_1000, "$")



print("################################")

print("N_totalfast22 = ", N_totalfast22, "and in log = ", np.log10(N_totalfast22), "**** N_totalobservablefast22 = ", N_totalobservablefast22, "and in log = ", np.log10(N_totalobservablefast22), "**** N_totalrecoverablefast22 = ", N_totalrecoverablefast22, "and in log = ", np.log10(N_totalrecoverablefast22))
print("N_totalfast22_03 = ", N_totalfast22_03, "and in log = ", np.log10(N_totalfast22_03), "**** N_totalobservablefast22_03 = ", N_totalobservablefast22_03, "and in log = ", np.log10(N_totalobservablefast22_03), "**** N_totalrecoverablefast22_03 = ", N_totalrecoverablefast22_03, "and in log = ", np.log10(N_totalrecoverablefast22_03))
print("N_totalfast22_1 = ", N_totalfast22_1, "and in log = ", np.log10(N_totalfast22_1), "**** N_totalobservablefast22_1 = ", N_totalobservablefast22_1, "and in log = ", np.log10(N_totalobservablefast22_1), "**** N_totalrecoverablefast22_1 = ", N_totalrecoverablefast22_1, "and in log = ", np.log10(N_totalrecoverablefast22_1))
print("N_totalfast22_10 = ", N_totalfast22_10, "and in log = ", np.log10(N_totalfast22_10), "**** N_totalobservablefast22_10 = ", N_totalobservablefast22_10, "and in log = ", np.log10(N_totalobservablefast22_10), "**** N_totalrecoverablefast22_10 = ", N_totalrecoverablefast22_10, "and in log = ", np.log10(N_totalrecoverablefast22_10))
print("N_totalfast22_30 = ", N_totalfast22_30, "and in log = ", np.log10(N_totalfast22_30), "**** N_totalobservablefast22_30 = ", N_totalobservablefast22_30, "and in log = ", np.log10(N_totalobservablefast22_30), "**** N_totalrecoverablefast22_30 = ", N_totalrecoverablefast22_30, "and in log = ", np.log10(N_totalrecoverablefast22_30))
print("N_totalfast22_100 = ", N_totalfast22_100, "and in log = ", np.log10(N_totalfast22_100), "**** N_totalobservablefast22_100 = ", N_totalobservablefast22_100, "and in log = ", np.log10(N_totalobservablefast22_100), "**** N_totalrecoverablefast22_100 = ", N_totalrecoverablefast22_100, "and in log = ", np.log10(N_totalrecoverablefast22_100))
print("N_totalfast22_1000 = ", N_totalfast22_1000, "and in log = ", np.log10(N_totalfast22_1000), "**** N_totalobservablefast22_1000 = ", N_totalobservablefast22_1000, "and in log = ", np.log10(N_totalobservablefast22_1000), "**** N_totalrecoverablefast22_1000 = ", N_totalrecoverablefast22_1000, "and in log = ", np.log10(N_totalrecoverablefast22_1000))

print("********************************")

print("wholerecoverypercent_fast22 = $", wholerecoverypercent_fast22, "/pm", sigmafast22, "$")
print("wholerecoverypercent_fast22_03 = $", wholerecoverypercent_fast22_03, "/pm", sigmafast22_03, "$")
print("wholerecoverypercent_fast22_1 = $", wholerecoverypercent_fast22_1, "/pm", sigmafast22_1, "$")
print("wholerecoverypercent_fast22_10 = $", wholerecoverypercent_fast22_10, "/pm", sigmafast22_10, "$")
print("wholerecoverypercent_fast22_30 = $", wholerecoverypercent_fast22_03, "/pm", sigmafast22_30, "$")
print("wholerecoverypercent_fast22_100 = $", wholerecoverypercent_fast22_100, "/pm", sigmafast22_100, "$")
print("wholerecoverypercent_fast22_1000 = $", wholerecoverypercent_fast22_1000, "/pm", sigmafast22_1000, "$")

print("********************************")

print("overallrecoverypercent_fast22 = $", overallrecoverypercent_fast22, "/pm", sigmafast22, "$")
print("overallrecoverypercent_fast22_03 = $", overallrecoverypercent_fast22_03, "/pm", sigmafast22_03, "$")
print("overallrecoverypercent_fast22_1 = $", overallrecoverypercent_fast22_1, "/pm", sigmafast22_1, "$")
print("overallrecoverypercent_fast22_10 = $", overallrecoverypercent_fast22_10, "/pm", sigmafast22_10, "$")
print("overallrecoverypercent_fast22_30 = $", overallrecoverypercent_fast22_03, "/pm", sigmafast22_30, "$")
print("overallrecoverypercent_fast22_100 = $", overallrecoverypercent_fast22_100, "/pm", sigmafast22_100, "$")
print("overallrecoverypercent_fast22_1000 = $", overallrecoverypercent_fast22_1000, "/pm", sigmafast22_1000, "$")


print("###############################")

print("N_totalfast195 = ", N_totalfast195, "and in log = ", np.log10(N_totalfast195), "**** N_totalobservablefast195 = ", N_totalobservablefast195, "and in log = ", np.log10(N_totalobservablefast195), "**** N_totalrecoverablefast195 = ", N_totalrecoverablefast195, "and in log = ", np.log10(N_totalrecoverablefast195))
print("N_totalfast195_03 = ", N_totalfast195_03, "and in log = ", np.log10(N_totalfast195_03), "**** N_totalobservablefast195_03 = ", N_totalobservablefast195_03, "and in log = ", np.log10(N_totalobservablefast195_03), "**** N_totalrecoverablefast195_03 = ", N_totalrecoverablefast195_03, "and in log = ", np.log10(N_totalrecoverablefast195_03))
print("N_totalfast195_1 = ", N_totalfast195_1, "and in log = ", np.log10(N_totalfast195_1), "**** N_totalobservablefast195_1 = ", N_totalobservablefast195_1, "and in log = ", np.log10(N_totalobservablefast195_1), "**** N_totalrecoverablefast195_1 = ", N_totalrecoverablefast195_1, "and in log = ", np.log10(N_totalrecoverablefast195_1))
print("N_totalfast195_10 = ", N_totalfast195_10, "and in log = ", np.log10(N_totalfast195_10), "**** N_totalobservablefast195_10 = ", N_totalobservablefast195_10, "and in log = ", np.log10(N_totalobservablefast195_10), "**** N_totalrecoverablefast195_10 = ", N_totalrecoverablefast195_10, "and in log = ", np.log10(N_totalrecoverablefast195_10))
print("N_totalfast195_30 = ", N_totalfast195_30, "and in log = ", np.log10(N_totalfast195_30), "**** N_totalobservablefast195_30 = ", N_totalobservablefast195_30, "and in log = ", np.log10(N_totalobservablefast195_30), "**** N_totalrecoverablefast195_30 = ", N_totalrecoverablefast195_30, "and in log = ", np.log10(N_totalrecoverablefast195_30))
print("N_totalfast195_100 = ", N_totalfast195_100, "and in log = ", np.log10(N_totalfast195_100), "**** N_totalobservablefast195_100 = ", N_totalobservablefast195_100, "and in log = ", np.log10(N_totalobservablefast195_100), "**** N_totalrecoverablefast195_100 = ", N_totalrecoverablefast195_100, "and in log = ", np.log10(N_totalrecoverablefast195_100))
print("N_totalfast195_1000 = ", N_totalfast195_1000, "and in log = ", np.log10(N_totalfast195_1000), "**** N_totalobservablefast195_1000 = ", N_totalobservablefast195_1000, "and in log = ", np.log10(N_totalobservablefast195_1000), "**** N_totalrecoverablefast195_1000 = ", N_totalrecoverablefast195_1000, "and in log = ", np.log10(N_totalrecoverablefast195_1000))

print("********************************")

print("wholerecoverypercent_fast195 = $", wholerecoverypercent_fast195, "/pm", sigmafast195, "$")
print("wholerecoverypercent_fast195_03 = $", wholerecoverypercent_fast195_03, "/pm", sigmafast195_03, "$")
print("wholerecoverypercent_fast195_1 = $", wholerecoverypercent_fast195_1, "/pm", sigmafast195_1, "$")
print("wholerecoverypercent_fast195_10 = $", wholerecoverypercent_fast195_10, "/pm", sigmafast195_10, "$")
print("wholerecoverypercent_fast195_30 = $", wholerecoverypercent_fast195_03, "/pm", sigmafast195_30, "$")
print("wholerecoverypercent_fast195_100 = $", wholerecoverypercent_fast195_100, "/pm", sigmafast195_100, "$")
print("wholerecoverypercent_fast195_1000 = $", wholerecoverypercent_fast195_1000, "/pm", sigmafast195_1000, "$")

print("********************************")

print("overallrecoverypercent_fast195 = $", overallrecoverypercent_fast195, "/pm", sigmafast195, "$")
print("overallrecoverypercent_fast195_03 = $", overallrecoverypercent_fast195_03, "/pm", sigmafast195_03, "$")
print("overallrecoverypercent_fast195_1 = $", overallrecoverypercent_fast195_1, "/pm", sigmafast195_1, "$")
print("overallrecoverypercent_fast195_10 = $", overallrecoverypercent_fast195_10, "/pm", sigmafast195_10, "$")
print("overallrecoverypercent_fast195_30 = $", overallrecoverypercent_fast195_03, "/pm", sigmafast195_30, "$")
print("overallrecoverypercent_fast195_100 = $", overallrecoverypercent_fast195_100, "/pm", sigmafast195_100, "$")
print("overallrecoverypercent_fast195_1000 = $", overallrecoverypercent_fast195_1000, "/pm", sigmafast195_1000, "$")

print("#############################")





print("binarypercent_22 = $", (N_totalfast22/N_totalfast)*100, "/pm", ((N_totalfast22**(1/2))/N_totalfast)*100, "$")
print("binarypercent_195 = $", (N_totalfast195/N_totalfast)*100, "/pm", ((N_totalfast195**(1/2))/N_totalfast)*100, "$")

print("binarypercent_03 = $", (N_totalfast_03/N_totalfast)*100, "/pm", ((N_totalfast_03**(1/2))/N_totalfast)*100, "$")
print("binarypercent_1 = $", (N_totalfast_1/N_totalfast)*100, "/pm", ((N_totalfast_1**(1/2))/N_totalfast)*100, "$")
print("binarypercent_10 = $", (N_totalfast_10/N_totalfast)*100, "/pm", ((N_totalfast_10**(1/2))/N_totalfast)*100, "$")
print("binarypercent_30 = $", (N_totalfast_30/N_totalfast)*100, "/pm", ((N_totalfast_30**(1/2))/N_totalfast)*100, "$")
print("binarypercent_100 = $", (N_totalfast_100/N_totalfast)*100, "/pm", ((N_totalfast_100**(1/2))/N_totalfast)*100, "$")
print("binarypercent_1000 = $", (N_totalfast_1000/N_totalfast)*100, "/pm", ((N_totalfast_1000**(1/2))/N_totalfast)*100, "$")

print("observablepercent_03 = $", (N_totalobservablefast_03/N_totalfast_03)*100, "/pm", ((N_totalobservablefast_03**(1/2))/N_totalfast_03)*100, "$")
print("observablepercent_1 = $", (N_totalobservablefast_1/N_totalfast_1)*100, "/pm", ((N_totalobservablefast_1**(1/2))/N_totalfast_1)*100, "$")
print("observablepercent_10 = $", (N_totalobservablefast_10/N_totalfast_10)*100, "/pm", ((N_totalobservablefast_10**(1/2))/N_totalfast_10)*100, "$")
print("observablepercent_30 = $", (N_totalobservablefast_30/N_totalfast_30)*100, "/pm", ((N_totalobservablefast_30**(1/2))/N_totalfast_30)*100, "$")
print("observablepercent_100 = $", (N_totalobservablefast_100/N_totalfast_100)*100, "/pm", ((N_totalobservablefast_100**(1/2))/N_totalfast_100)*100, "$")
print("observablepercent_1000 = $", (N_totalobservablefast_1000/N_totalfast_1000)*100, "/pm", ((N_totalobservablefast_1000**(1/2))/N_totalfast_1000)*100, "$")

print("observablepercent = $", (N_totalobservablefast/N_totalfast)*100, "/pm", ((N_totalobservablefast**(1/2))/N_totalfast)*100, "$")
print("observablepercent22 = $", (N_totalobservablefast22/N_totalfast22)*100, "/pm", ((N_totalobservablefast22**(1/2))/N_totalfast22)*100, "$")
print("observablepercent195 = $", (N_totalobservablefast195/N_totalfast195)*100, "/pm", ((N_totalobservablefast195**(1/2))/N_totalfast195)*100, "$")



for fileobsDist_ in sorted(allFiles_obsDist):

	filename = fileobsDist_[77:] #when file path no longer has /old in it, will be fileobsDist_[73:]
	fileid = filename.strip('output_file.csv')
	print ("I'm starting " + fileid)


	datobsDist = pd.read_csv(fileobsDist_, sep = ',', header=2)
	PeriodIn = datobsDist['p']		 # input period -- 'p' in data file

	##########################################################

	datobsDist1 = pd.read_csv(fileobsDist_, sep = ',', header=0, nrows=1)
	N_tri = datobsDist1["NstarsTRILEGAL"][0]
	Nall = len(PeriodIn)


	m1hAll0, m1b = np.histogram(datobsDist["m1"], bins=mbins)
	dm1 = np.diff(m1b)
	m1val = m1b[:-1] + dm1/2.
	fb = np.sum(m1hAll0/Nall*fbFit(m1val))

	N_mult = N_tri*fb

	##########################################################

	if len(PeriodIn) == 0.:
		continue
	if N_tri == 0:
		continue
	else:
		PeriodOut = datobsDist['LSM_PERIOD'] #LSM_PERIOD in data file
		appMagMean = datobsDist['appMagMean'] #apparent magnitude, will use to make cuts for 24 (default), 22, and then Kepler's range (?? -- brighter than LSST can manage-- to 19) OR 19.5 (SNR = 10)
		
		observable = datobsDist.loc[PeriodOut != -999].index
		observable_03 = datobsDist.loc[(PeriodIn <= 0.3) & (PeriodOut != -999)].index
		observable_1 = datobsDist.loc[(PeriodIn <= 1) & (PeriodOut != -999)].index
		observable_10 = datobsDist.loc[(PeriodIn <= 10) & (PeriodOut != -999)].index
		observable_30 = datobsDist.loc[(PeriodIn <= 30) & (PeriodOut != -999)].index
		observable_100 = datobsDist.loc[(PeriodIn <= 100) & (PeriodOut != -999)].index
		observable_1000 = datobsDist.loc[(PeriodIn <= 1000) & (PeriodOut != -999)].index
		
		observable_22 = datobsDist.loc[(PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_03_22 = datobsDist.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_1_22 = datobsDist.loc[(PeriodIn <= 1) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_10_22 = datobsDist.loc[(PeriodIn <= 10) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_30_22 = datobsDist.loc[(PeriodIn <= 30) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_100_22 = datobsDist.loc[(PeriodIn <= 100) & (PeriodOut != -999) & (appMagMean <= 22.)].index
		observable_1000_22 = datobsDist.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & (appMagMean <= 22.)].index

		observable_195 = datobsDist.loc[(PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_03_195 = datobsDist.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_1_195 = datobsDist.loc[(PeriodIn <= 1) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_10_195 = datobsDist.loc[(PeriodIn <= 10) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_30_195 = datobsDist.loc[(PeriodIn <= 30) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_100_195 = datobsDist.loc[(PeriodIn <= 100) & (PeriodOut != -999) & (appMagMean <= 19.5)].index
		observable_1000_195 = datobsDist.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & (appMagMean <= 19.5)].index

		fullP = abs(PeriodOut - PeriodIn)/PeriodIn
		halfP = abs(PeriodOut - 0.5*PeriodIn)/(0.5*PeriodIn)
		twiceP = abs(PeriodOut - 2*PeriodIn)/(2*PeriodIn)

		recoverable = datobsDist.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_03 = datobsDist.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_1 = datobsDist.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_10 = datobsDist.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_30 = datobsDist.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_100 = datobsDist.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index
		recoverable_1000 = datobsDist.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP))].index

		recoverable_22 = datobsDist.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_03_22 = datobsDist.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_1_22 = datobsDist.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_10_22 = datobsDist.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_30_22 = datobsDist.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_100_22 = datobsDist.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index
		recoverable_1000_22 = datobsDist.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 22.)].index

		recoverable_195 = datobsDist.loc[(PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_03_195 = datobsDist.loc[(PeriodIn <= 0.3) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_1_195 = datobsDist.loc[(PeriodIn <= 1) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_10_195 = datobsDist.loc[(PeriodIn <= 10) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_30_195 = datobsDist.loc[(PeriodIn <= 30) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_100_195 = datobsDist.loc[(PeriodIn <= 100) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index
		recoverable_1000_195 = datobsDist.loc[(PeriodIn <= 1000) & (PeriodOut != -999) & ((fullP < cutP) | (halfP < cutP) | (twiceP < cutP)) & (appMagMean <= 19.5)].index

		P03 = datobsDist.loc[PeriodIn <= 0.3].index
		P1 = datobsDist.loc[PeriodIn <= 1].index
		P10 = datobsDist.loc[PeriodIn <= 10].index
		P30 = datobsDist.loc[PeriodIn <= 30].index
		P100 = datobsDist.loc[PeriodIn <= 100].index
		P1000 = datobsDist.loc[PeriodIn <= 1000].index

		P_22 = datobsDist.loc[appMagMean <= 22.].index
		P03_22 = datobsDist.loc[(PeriodIn <= 0.3) & (appMagMean <= 22.)].index
		P1_22 = datobsDist.loc[(PeriodIn <= 1) & (appMagMean <= 22.)].index
		P10_22 = datobsDist.loc[(PeriodIn <= 10) & (appMagMean <= 22.)].index
		P30_22 = datobsDist.loc[(PeriodIn <= 30) & (appMagMean <= 22.)].index
		P100_22 = datobsDist.loc[(PeriodIn <= 100) & (appMagMean <= 22.)].index
		P1000_22 = datobsDist.loc[(PeriodIn <= 1000) & (appMagMean <= 22.)].index

		P_195 = datobsDist.loc[appMagMean <= 19.5].index
		P03_195 = datobsDist.loc[(PeriodIn <= 0.3) & (appMagMean <= 19.5)].index
		P1_195 = datobsDist.loc[(PeriodIn <= 1) & (appMagMean <= 19.5)].index
		P10_195 = datobsDist.loc[(PeriodIn <= 10) & (appMagMean <= 19.5)].index
		P30_195 = datobsDist.loc[(PeriodIn <= 30) & (appMagMean <= 19.5)].index
		P100_195 = datobsDist.loc[(PeriodIn <= 100) & (appMagMean <= 19.5)].index
		P1000_195 = datobsDist.loc[(PeriodIn <= 1000) & (appMagMean <= 19.5)].index

		N_all = (len(PeriodIn)/len(PeriodIn))*N_mult
		N_all03 = (len(P03)/len(PeriodIn))*N_mult
		N_all1 = (len(P1)/len(PeriodIn))*N_mult
		N_all10 = (len(P10)/len(PeriodIn))*N_mult
		N_all30 = (len(P30)/len(PeriodIn))*N_mult
		N_all100 = (len(P100)/len(PeriodIn))*N_mult
		N_all1000 = (len(P1000)/len(PeriodIn))*N_mult

		N_all_22 = (len(P_22)/len(PeriodIn))*N_mult
		N_all03_22 = (len(P03_22)/len(PeriodIn))*N_mult
		N_all1_22 = (len(P1_22)/len(PeriodIn))*N_mult
		N_all10_22 = (len(P10_22)/len(PeriodIn))*N_mult
		N_all30_22 = (len(P30_22)/len(PeriodIn))*N_mult
		N_all100_22 = (len(P100_22)/len(PeriodIn))*N_mult
		N_all1000_22 = (len(P1000_22)/len(PeriodIn))*N_mult

		N_all_195 = (len(P_195)/len(PeriodIn))*N_mult
		N_all03_195 = (len(P03_195)/len(PeriodIn))*N_mult
		N_all1_195 = (len(P1_195)/len(PeriodIn))*N_mult
		N_all10_195 = (len(P10_195)/len(PeriodIn))*N_mult
		N_all30_195 = (len(P30_195)/len(PeriodIn))*N_mult
		N_all100_195 = (len(P100_195)/len(PeriodIn))*N_mult
		N_all1000_195 = (len(P1000_195)/len(PeriodIn))*N_mult

		N_obs = (len(observable)/len(PeriodIn))*N_mult
		N_obs03 = (len(observable_03)/len(PeriodIn))*N_mult
		N_obs1 = (len(observable_1)/len(PeriodIn))*N_mult
		N_obs10 = (len(observable_10)/len(PeriodIn))*N_mult
		N_obs30 = (len(observable_30)/len(PeriodIn))*N_mult
		N_obs100 = (len(observable_100)/len(PeriodIn))*N_mult
		N_obs1000 = (len(observable_1000)/len(PeriodIn))*N_mult

		N_obs_22 = (len(observable_22)/len(PeriodIn))*N_mult
		N_obs03_22 = (len(observable_03_22)/len(PeriodIn))*N_mult
		N_obs1_22 = (len(observable_1_22)/len(PeriodIn))*N_mult
		N_obs10_22 = (len(observable_10_22)/len(PeriodIn))*N_mult
		N_obs30_22 = (len(observable_30_22)/len(PeriodIn))*N_mult
		N_obs100_22 = (len(observable_100_22)/len(PeriodIn))*N_mult
		N_obs1000_22 = (len(observable_1000_22)/len(PeriodIn))*N_mult

		N_obs_195 = (len(observable_195)/len(PeriodIn))*N_mult
		N_obs03_195 = (len(observable_03_195)/len(PeriodIn))*N_mult
		N_obs1_195 = (len(observable_1_195)/len(PeriodIn))*N_mult
		N_obs10_195 = (len(observable_10_195)/len(PeriodIn))*N_mult
		N_obs30_195 = (len(observable_30_195)/len(PeriodIn))*N_mult
		N_obs100_195 = (len(observable_100_195)/len(PeriodIn))*N_mult
		N_obs1000_195 = (len(observable_1000_195)/len(PeriodIn))*N_mult

		N_rec = (len(recoverable)/len(PeriodIn))*N_mult
		N_rec03 = (len(recoverable_03)/len(PeriodIn))*N_mult
		N_rec1 = (len(recoverable_1)/len(PeriodIn))*N_mult
		N_rec10 = (len(recoverable_10)/len(PeriodIn))*N_mult
		N_rec30 = (len(recoverable_30)/len(PeriodIn))*N_mult
		N_rec100 = (len(recoverable_100)/len(PeriodIn))*N_mult
		N_rec1000 = (len(recoverable_1000)/len(PeriodIn))*N_mult

		N_rec_22 = (len(recoverable_22)/len(PeriodIn))*N_mult
		N_rec03_22 = (len(recoverable_03_22)/len(PeriodIn))*N_mult
		N_rec1_22 = (len(recoverable_1_22)/len(PeriodIn))*N_mult
		N_rec10_22 = (len(recoverable_10_22)/len(PeriodIn))*N_mult
		N_rec30_22 = (len(recoverable_30_22)/len(PeriodIn))*N_mult
		N_rec100_22 = (len(recoverable_100_22)/len(PeriodIn))*N_mult
		N_rec1000_22 = (len(recoverable_1000_22)/len(PeriodIn))*N_mult

		N_rec_195 = (len(recoverable_195)/len(PeriodIn))*N_mult
		N_rec03_195 = (len(recoverable_03_195)/len(PeriodIn))*N_mult
		N_rec1_195 = (len(recoverable_1_195)/len(PeriodIn))*N_mult
		N_rec10_195 = (len(recoverable_10_195)/len(PeriodIn))*N_mult
		N_rec30_195 = (len(recoverable_30_195)/len(PeriodIn))*N_mult
		N_rec100_195 = (len(recoverable_100_195)/len(PeriodIn))*N_mult
		N_rec1000_195 = (len(recoverable_1000_195)/len(PeriodIn))*N_mult

		N_totalobsDist_array.append(float(N_all))
		N_totalobservableobsDist_array.append(float(N_obs))
		N_totalrecoverableobsDist_array.append(float(N_rec))

		N_totalobsDist_array_03.append(float(N_all03))
		N_totalobservableobsDist_array_03.append(float(N_obs03))
		N_totalrecoverableobsDist_array_03.append(float(N_rec03))

		N_totalobsDist_array_1.append(float(N_all1))
		N_totalobservableobsDist_array_1.append(float(N_obs1))
		N_totalrecoverableobsDist_array_1.append(float(N_rec1))

		N_totalobsDist_array_10.append(float(N_all10))
		N_totalobservableobsDist_array_10.append(float(N_obs10))
		N_totalrecoverableobsDist_array_10.append(float(N_rec10))

		N_totalobsDist_array_30.append(float(N_all30))
		N_totalobservableobsDist_array_30.append(float(N_obs30))
		N_totalrecoverableobsDist_array_30.append(float(N_rec30))

		N_totalobsDist_array_100.append(float(N_all100))
		N_totalobservableobsDist_array_100.append(float(N_obs100))
		N_totalrecoverableobsDist_array_100.append(float(N_rec100))

		N_totalobsDist_array_1000.append(float(N_all1000))
		N_totalobservableobsDist_array_1000.append(float(N_obs1000))
		N_totalrecoverableobsDist_array_1000.append(float(N_rec1000))

		N_totalobsDist22_array.append(float(N_all_22))
		N_totalobservableobsDist22_array.append(float(N_obs_22))
		N_totalrecoverableobsDist22_array.append(float(N_rec_22))

		N_totalobsDist22_array_03.append(float(N_all03_22))
		N_totalobservableobsDist22_array_03.append(float(N_obs03_22))
		N_totalrecoverableobsDist22_array_03.append(float(N_rec03_22))

		N_totalobsDist22_array_1.append(float(N_all1_22))
		N_totalobservableobsDist22_array_1.append(float(N_obs1_22))
		N_totalrecoverableobsDist22_array_1.append(float(N_rec1_22))

		N_totalobsDist22_array_10.append(float(N_all10_22))
		N_totalobservableobsDist22_array_10.append(float(N_obs10_22))
		N_totalrecoverableobsDist22_array_10.append(float(N_rec10_22))

		N_totalobsDist22_array_30.append(float(N_all30_22))
		N_totalobservableobsDist22_array_30.append(float(N_obs30_22))
		N_totalrecoverableobsDist22_array_30.append(float(N_rec30_22))

		N_totalobsDist22_array_100.append(float(N_all100_22))
		N_totalobservableobsDist22_array_100.append(float(N_obs100_22))
		N_totalrecoverableobsDist22_array_100.append(float(N_rec100_22))

		N_totalobsDist22_array_1000.append(float(N_all1000_22))
		N_totalobservableobsDist22_array_1000.append(float(N_obs1000_22))
		N_totalrecoverableobsDist22_array_1000.append(float(N_rec1000_22))

		N_totalobsDist195_array.append(float(N_all_195))
		N_totalobservableobsDist195_array.append(float(N_obs_195))
		N_totalrecoverableobsDist195_array.append(float(N_rec_195))

		N_totalobsDist195_array_03.append(float(N_all03_195))
		N_totalobservableobsDist195_array_03.append(float(N_obs03_195))
		N_totalrecoverableobsDist195_array_03.append(float(N_rec03_195))

		N_totalobsDist195_array_1.append(float(N_all1_195))
		N_totalobservableobsDist195_array_1.append(float(N_obs1_195))
		N_totalrecoverableobsDist195_array_1.append(float(N_rec1_195))

		N_totalobsDist195_array_10.append(float(N_all10_195))
		N_totalobservableobsDist195_array_10.append(float(N_obs10_195))
		N_totalrecoverableobsDist195_array_10.append(float(N_rec10_195))

		N_totalobsDist195_array_30.append(float(N_all30_195))
		N_totalobservableobsDist195_array_30.append(float(N_obs30_195))
		N_totalrecoverableobsDist195_array_30.append(float(N_rec30_195))

		N_totalobsDist195_array_100.append(float(N_all100_195))
		N_totalobservableobsDist195_array_100.append(float(N_obs100_195))
		N_totalrecoverableobsDist195_array_100.append(float(N_rec100_195))

		N_totalobsDist195_array_1000.append(float(N_all1000_195))
		N_totalobservableobsDist195_array_1000.append(float(N_obs1000_195))
		N_totalrecoverableobsDist195_array_1000.append(float(N_rec1000_195))


N_totalobsDist = np.sum(N_totalobsDist_array)
N_totalobsDist_03 = np.sum(N_totalobsDist_array_03)
N_totalobsDist_1 = np.sum(N_totalobsDist_array_1)
N_totalobsDist_10 = np.sum(N_totalobsDist_array_10)
N_totalobsDist_30 = np.sum(N_totalobsDist_array_30)
N_totalobsDist_100 = np.sum(N_totalobsDist_array_100)
N_totalobsDist_1000 = np.sum(N_totalobsDist_array_1000)


N_totalobservableobsDist = np.sum(N_totalobservableobsDist_array)
N_totalobservableobsDist_03 = np.sum(N_totalobservableobsDist_array_03)
N_totalobservableobsDist_1 = np.sum(N_totalobservableobsDist_array_1)
N_totalobservableobsDist_10 = np.sum(N_totalobservableobsDist_array_10)
N_totalobservableobsDist_30 = np.sum(N_totalobservableobsDist_array_30)
N_totalobservableobsDist_100 = np.sum(N_totalobservableobsDist_array_100)
N_totalobservableobsDist_1000 = np.sum(N_totalobservableobsDist_array_1000)


N_totalrecoverableobsDist = np.sum(N_totalrecoverableobsDist_array)
N_totalrecoverableobsDist_03 = np.sum(N_totalrecoverableobsDist_array_03)
N_totalrecoverableobsDist_1 = np.sum(N_totalrecoverableobsDist_array_1)
N_totalrecoverableobsDist_10 = np.sum(N_totalrecoverableobsDist_array_10)
N_totalrecoverableobsDist_30 = np.sum(N_totalrecoverableobsDist_array_30)
N_totalrecoverableobsDist_100 = np.sum(N_totalrecoverableobsDist_array_100)
N_totalrecoverableobsDist_1000 = np.sum(N_totalrecoverableobsDist_array_1000)


N_totalobsDist22 = np.sum(N_totalobsDist22_array)
N_totalobsDist22_03 = np.sum(N_totalobsDist22_array_03)
N_totalobsDist22_1 = np.sum(N_totalobsDist22_array_1)
N_totalobsDist22_10 = np.sum(N_totalobsDist22_array_10)
N_totalobsDist22_30 = np.sum(N_totalobsDist22_array_30)
N_totalobsDist22_100 = np.sum(N_totalobsDist22_array_100)
N_totalobsDist22_1000 = np.sum(N_totalobsDist22_array_1000)


N_totalobservableobsDist22 = np.sum(N_totalobservableobsDist22_array)
N_totalobservableobsDist22_03 = np.sum(N_totalobservableobsDist22_array_03)
N_totalobservableobsDist22_1 = np.sum(N_totalobservableobsDist22_array_1)
N_totalobservableobsDist22_10 = np.sum(N_totalobservableobsDist22_array_10)
N_totalobservableobsDist22_30 = np.sum(N_totalobservableobsDist22_array_30)
N_totalobservableobsDist22_100 = np.sum(N_totalobservableobsDist22_array_100)
N_totalobservableobsDist22_1000 = np.sum(N_totalobservableobsDist22_array_1000)


N_totalrecoverableobsDist22 = np.sum(N_totalrecoverableobsDist22_array)
N_totalrecoverableobsDist22_03 = np.sum(N_totalrecoverableobsDist22_array_03)
N_totalrecoverableobsDist22_1 = np.sum(N_totalrecoverableobsDist22_array_1)
N_totalrecoverableobsDist22_10 = np.sum(N_totalrecoverableobsDist22_array_10)
N_totalrecoverableobsDist22_30 = np.sum(N_totalrecoverableobsDist22_array_30)
N_totalrecoverableobsDist22_100 = np.sum(N_totalrecoverableobsDist22_array_100)
N_totalrecoverableobsDist22_1000 = np.sum(N_totalrecoverableobsDist22_array_1000)

N_totalobsDist195 = np.sum(N_totalobsDist195_array)
N_totalobsDist195_03 = np.sum(N_totalobsDist195_array_03)
N_totalobsDist195_1 = np.sum(N_totalobsDist195_array_1)
N_totalobsDist195_10 = np.sum(N_totalobsDist195_array_10)
N_totalobsDist195_30 = np.sum(N_totalobsDist195_array_30)
N_totalobsDist195_100 = np.sum(N_totalobsDist195_array_100)
N_totalobsDist195_1000 = np.sum(N_totalobsDist195_array_1000)


N_totalobservableobsDist195 = np.sum(N_totalobservableobsDist195_array)
N_totalobservableobsDist195_03 = np.sum(N_totalobservableobsDist195_array_03)
N_totalobservableobsDist195_1 = np.sum(N_totalobservableobsDist195_array_1)
N_totalobservableobsDist195_10 = np.sum(N_totalobservableobsDist195_array_10)
N_totalobservableobsDist195_30 = np.sum(N_totalobservableobsDist195_array_30)
N_totalobservableobsDist195_100 = np.sum(N_totalobservableobsDist195_array_100)
N_totalobservableobsDist195_1000 = np.sum(N_totalobservableobsDist195_array_1000)


N_totalrecoverableobsDist195 = np.sum(N_totalrecoverableobsDist195_array)
N_totalrecoverableobsDist195_03 = np.sum(N_totalrecoverableobsDist195_array_03)
N_totalrecoverableobsDist195_1 = np.sum(N_totalrecoverableobsDist195_array_1)
N_totalrecoverableobsDist195_10 = np.sum(N_totalrecoverableobsDist195_array_10)
N_totalrecoverableobsDist195_30 = np.sum(N_totalrecoverableobsDist195_array_30)
N_totalrecoverableobsDist195_100 = np.sum(N_totalrecoverableobsDist195_array_100)
N_totalrecoverableobsDist195_1000 = np.sum(N_totalrecoverableobsDist195_array_1000)


wholerecoverypercent_obsDist = (N_totalrecoverableobsDist/N_totalobservableobsDist)*100
wholerecoverypercent_obsDist_03 = (N_totalrecoverableobsDist_03/N_totalobservableobsDist_03)*100
wholerecoverypercent_obsDist_1 = (N_totalrecoverableobsDist_1/N_totalobservableobsDist_1)*100
wholerecoverypercent_obsDist_10 = (N_totalrecoverableobsDist_10/N_totalobservableobsDist_10)*100
wholerecoverypercent_obsDist_30 = (N_totalrecoverableobsDist_30/N_totalobservableobsDist_30)*100
wholerecoverypercent_obsDist_100 = (N_totalrecoverableobsDist_100/N_totalobservableobsDist_100)*100
wholerecoverypercent_obsDist_1000 = (N_totalrecoverableobsDist_1000/N_totalobservableobsDist_1000)*100
sigmaobsDist = ((N_totalrecoverableobsDist**(1/2))/N_totalobservableobsDist)*100
sigmaobsDist_03 = ((N_totalrecoverableobsDist_03**(1/2))/N_totalobservableobsDist_03)*100
sigmaobsDist_1 = ((N_totalrecoverableobsDist_1**(1/2))/N_totalobservableobsDist_1)*100
sigmaobsDist_10 = ((N_totalrecoverableobsDist_10**(1/2))/N_totalobservableobsDist_10)*100
sigmaobsDist_30 = ((N_totalrecoverableobsDist_30**(1/2))/N_totalobservableobsDist_30)*100
sigmaobsDist_100 = ((N_totalrecoverableobsDist_100**(1/2))/N_totalobservableobsDist_100)*100
sigmaobsDist_1000 = ((N_totalrecoverableobsDist_1000**(1/2))/N_totalobservableobsDist_1000)*100
overallrecoverypercent_obsDist = (N_totalrecoverableobsDist/N_totalobsDist)*100
overallrecoverypercent_obsDist_03 = (N_totalrecoverableobsDist_03/N_totalobsDist_03)*100
overallrecoverypercent_obsDist_1 = (N_totalrecoverableobsDist_1/N_totalobsDist_1)*100
overallrecoverypercent_obsDist_10 = (N_totalrecoverableobsDist_10/N_totalobsDist_10)*100
overallrecoverypercent_obsDist_30 = (N_totalrecoverableobsDist_30/N_totalobsDist_30)*100
overallrecoverypercent_obsDist_100 = (N_totalrecoverableobsDist_100/N_totalobsDist_100)*100
overallrecoverypercent_obsDist_1000 = (N_totalrecoverableobsDist_1000/N_totalobsDist_1000)*100
overallsigmaobsDist = ((N_totalrecoverableobsDist**(1/2))/N_totalobsDist)*100
overallsigmaobsDist_03 = ((N_totalrecoverableobsDist_03**(1/2))/N_totalobsDist_03)*100
overallsigmaobsDist_1 = ((N_totalrecoverableobsDist_1**(1/2))/N_totalobsDist_1)*100
overallsigmaobsDist_10 = ((N_totalrecoverableobsDist_10**(1/2))/N_totalobsDist_10)*100
overallsigmaobsDist_30 = ((N_totalrecoverableobsDist_30**(1/2))/N_totalobsDist_30)*100
overallsigmaobsDist_100 = ((N_totalrecoverableobsDist_100**(1/2))/N_totalobsDist_100)*100
overallsigmaobsDist_1000 = ((N_totalrecoverableobsDist_1000**(1/2))/N_totalobsDist_1000)*100


wholerecoverypercent_obsDist22 = (N_totalrecoverableobsDist22/N_totalobservableobsDist22)*100
wholerecoverypercent_obsDist22_03 = (N_totalrecoverableobsDist22_03/N_totalobservableobsDist22_03)*100
wholerecoverypercent_obsDist22_1 = (N_totalrecoverableobsDist22_1/N_totalobservableobsDist22_1)*100
wholerecoverypercent_obsDist22_10 = (N_totalrecoverableobsDist22_10/N_totalobservableobsDist22_10)*100
wholerecoverypercent_obsDist22_30 = (N_totalrecoverableobsDist22_30/N_totalobservableobsDist22_30)*100
wholerecoverypercent_obsDist22_100 = (N_totalrecoverableobsDist22_100/N_totalobservableobsDist22_100)*100
wholerecoverypercent_obsDist22_1000 = (N_totalrecoverableobsDist22_1000/N_totalobservableobsDist22_1000)*100
sigmaobsDist22 = ((N_totalrecoverableobsDist22**(1/2))/N_totalobservableobsDist22)*100
sigmaobsDist22_03 = ((N_totalrecoverableobsDist22_03**(1/2))/N_totalobservableobsDist22_03)*100
sigmaobsDist22_1 = ((N_totalrecoverableobsDist22_1**(1/2))/N_totalobservableobsDist22_1)*100
sigmaobsDist22_10 = ((N_totalrecoverableobsDist22_10**(1/2))/N_totalobservableobsDist22_10)*100
sigmaobsDist22_30 = ((N_totalrecoverableobsDist22_30**(1/2))/N_totalobservableobsDist22_30)*100
sigmaobsDist22_100 = ((N_totalrecoverableobsDist22_100**(1/2))/N_totalobservableobsDist22_100)*100
sigmaobsDist22_1000 = ((N_totalrecoverableobsDist22_1000**(1/2))/N_totalobservableobsDist22_1000)*100
overallrecoverypercent_obsDist22 = (N_totalrecoverableobsDist22/N_totalobsDist22)*100
overallrecoverypercent_obsDist22_03 = (N_totalrecoverableobsDist22_03/N_totalobsDist22_03)*100
overallrecoverypercent_obsDist22_1 = (N_totalrecoverableobsDist22_1/N_totalobsDist22_1)*100
overallrecoverypercent_obsDist22_10 = (N_totalrecoverableobsDist22_10/N_totalobsDist22_10)*100
overallrecoverypercent_obsDist22_30 = (N_totalrecoverableobsDist22_30/N_totalobsDist22_30)*100
overallrecoverypercent_obsDist22_100 = (N_totalrecoverableobsDist22_100/N_totalobsDist22_100)*100
overallrecoverypercent_obsDist22_1000 = (N_totalrecoverableobsDist22_1000/N_totalobsDist22_1000)*100
overallsigmaobsDist22 = ((N_totalrecoverableobsDist22**(1/2))/N_totalobsDist22)*100
overallsigmaobsDist22_03 = ((N_totalrecoverableobsDist22_03**(1/2))/N_totalobsDist22_03)*100
overallsigmaobsDist22_1 = ((N_totalrecoverableobsDist22_1**(1/2))/N_totalobsDist22_1)*100
overallsigmaobsDist22_10 = ((N_totalrecoverableobsDist22_10**(1/2))/N_totalobsDist22_10)*100
overallsigmaobsDist22_30 = ((N_totalrecoverableobsDist22_30**(1/2))/N_totalobsDist22_30)*100
overallsigmaobsDist22_100 = ((N_totalrecoverableobsDist22_100**(1/2))/N_totalobsDist22_100)*100
overallsigmaobsDist22_1000 = ((N_totalrecoverableobsDist22_1000**(1/2))/N_totalobsDist22_1000)*100


wholerecoverypercent_obsDist195 = (N_totalrecoverableobsDist195/N_totalobservableobsDist195)*100
wholerecoverypercent_obsDist195_03 = (N_totalrecoverableobsDist195_03/N_totalobservableobsDist195_03)*100
wholerecoverypercent_obsDist195_1 = (N_totalrecoverableobsDist195_1/N_totalobservableobsDist195_1)*100
wholerecoverypercent_obsDist195_10 = (N_totalrecoverableobsDist195_10/N_totalobservableobsDist195_10)*100
wholerecoverypercent_obsDist195_30 = (N_totalrecoverableobsDist195_30/N_totalobservableobsDist195_30)*100
wholerecoverypercent_obsDist195_100 = (N_totalrecoverableobsDist195_100/N_totalobservableobsDist195_100)*100
wholerecoverypercent_obsDist195_1000 = (N_totalrecoverableobsDist195_1000/N_totalobservableobsDist195_1000)*100
sigmaobsDist195 = ((N_totalrecoverableobsDist195**(1/2))/N_totalobservableobsDist195)*100
sigmaobsDist195_03 = ((N_totalrecoverableobsDist195_03**(1/2))/N_totalobservableobsDist195_03)*100
sigmaobsDist195_1 = ((N_totalrecoverableobsDist195_1**(1/2))/N_totalobservableobsDist195_1)*100
sigmaobsDist195_10 = ((N_totalrecoverableobsDist195_10**(1/2))/N_totalobservableobsDist195_10)*100
sigmaobsDist195_30 = ((N_totalrecoverableobsDist195_30**(1/2))/N_totalobservableobsDist195_30)*100
sigmaobsDist195_100 = ((N_totalrecoverableobsDist195_100**(1/2))/N_totalobservableobsDist195_100)*100
sigmaobsDist195_1000 = ((N_totalrecoverableobsDist195_1000**(1/2))/N_totalobservableobsDist195_1000)*100
overallrecoverypercent_obsDist195 = (N_totalrecoverableobsDist195/N_totalobsDist195)*100
overallrecoverypercent_obsDist195_03 = (N_totalrecoverableobsDist195_03/N_totalobsDist195_03)*100
overallrecoverypercent_obsDist195_1 = (N_totalrecoverableobsDist195_1/N_totalobsDist195_1)*100
overallrecoverypercent_obsDist195_10 = (N_totalrecoverableobsDist195_10/N_totalobsDist195_10)*100
overallrecoverypercent_obsDist195_30 = (N_totalrecoverableobsDist195_30/N_totalobsDist195_30)*100
overallrecoverypercent_obsDist195_100 = (N_totalrecoverableobsDist195_100/N_totalobsDist195_100)*100
overallrecoverypercent_obsDist195_1000 = (N_totalrecoverableobsDist195_1000/N_totalobsDist195_1000)*100
overallsigmaobsDist195 = ((N_totalrecoverableobsDist195**(1/2))/N_totalobsDist195)*100
overallsigmaobsDist195_03 = ((N_totalrecoverableobsDist195_03**(1/2))/N_totalobsDist195_03)*100
overallsigmaobsDist195_1 = ((N_totalrecoverableobsDist195_1**(1/2))/N_totalobsDist195_1)*100
overallsigmaobsDist195_10 = ((N_totalrecoverableobsDist195_10**(1/2))/N_totalobsDist195_10)*100
overallsigmaobsDist195_30 = ((N_totalrecoverableobsDist195_30**(1/2))/N_totalobsDist195_30)*100
overallsigmaobsDist195_100 = ((N_totalrecoverableobsDist195_100**(1/2))/N_totalobsDist195_100)*100
overallsigmaobsDist195_1000 = ((N_totalrecoverableobsDist195_1000**(1/2))/N_totalobsDist195_1000)*100\





print("N_totalobsDist = ", N_totalobsDist, "and in log = ", np.log10(N_totalobsDist), "**** N_totalobservableobsDist = ", N_totalobservableobsDist, "and in log = ", np.log10(N_totalobservableobsDist), "**** N_totalrecoverableobsDist = ", N_totalrecoverableobsDist, "and in log = ", np.log10(N_totalrecoverableobsDist))
print("N_totalobsDist_03 = ", N_totalobsDist_03, "and in log = ", np.log10(N_totalobsDist_03), "**** N_totalobservableobsDist_03 = ", N_totalobservableobsDist_03, "and in log = ", np.log10(N_totalobservableobsDist_03), "**** N_totalrecoverableobsDist_03 = ", N_totalrecoverableobsDist_03, "and in log = ", np.log10(N_totalrecoverableobsDist_03))
print("N_totalobsDist_1 = ", N_totalobsDist_1, "and in log = ", np.log10(N_totalobsDist_1), "**** N_totalobservableobsDist_1 = ", N_totalobservableobsDist_1, "and in log = ", np.log10(N_totalobservableobsDist_1), "**** N_totalrecoverableobsDist_1 = ", N_totalrecoverableobsDist_1, "and in log = ", np.log10(N_totalrecoverableobsDist_1))
print("N_totalobsDist_10 = ", N_totalobsDist_10, "and in log = ", np.log10(N_totalobsDist_10), "**** N_totalobservableobsDist_10 = ", N_totalobservableobsDist_10, "and in log = ", np.log10(N_totalobservableobsDist_10), "**** N_totalrecoverableobsDist_10 = ", N_totalrecoverableobsDist_10, "and in log = ", np.log10(N_totalrecoverableobsDist_10))
print("N_totalobsDist_30 = ", N_totalobsDist_30, "and in log = ", np.log10(N_totalobsDist_30), "**** N_totalobservableobsDist_30 = ", N_totalobservableobsDist_30, "and in log = ", np.log10(N_totalobservableobsDist_30), "**** N_totalrecoverableobsDist_30 = ", N_totalrecoverableobsDist_30, "and in log = ", np.log10(N_totalrecoverableobsDist_30))
print("N_totalobsDist_100 = ", N_totalobsDist_100, "and in log = ", np.log10(N_totalobsDist_100), "**** N_totalobservableobsDist_100 = ", N_totalobservableobsDist_100, "and in log = ", np.log10(N_totalobservableobsDist_100), "**** N_totalrecoverableobsDist_100 = ", N_totalrecoverableobsDist_100, "and in log = ", np.log10(N_totalrecoverableobsDist_100))
print("N_totalobsDist_1000 = ", N_totalobsDist_1000, "and in log = ", np.log10(N_totalobsDist_1000), "**** N_totalobservableobsDist_1000 = ", N_totalobservableobsDist_1000, "and in log = ", np.log10(N_totalobservableobsDist_1000), "**** N_totalrecoverableobsDist_1000 = ", N_totalrecoverableobsDist_1000, "and in log = ", np.log10(N_totalrecoverableobsDist_1000))

print("********************************")

print("wholerecoverypercent_obsDist = $", wholerecoverypercent_obsDist, "/pm", sigmaobsDist, "$")
print("wholerecoverypercent_obsDist_03 = $", wholerecoverypercent_obsDist_03, "/pm", sigmaobsDist_03, "$")
print("wholerecoverypercent_obsDist_1 = $", wholerecoverypercent_obsDist_1, "/pm", sigmaobsDist_1, "$")
print("wholerecoverypercent_obsDist_10 = $", wholerecoverypercent_obsDist_10, "/pm", sigmaobsDist_10, "$")
print("wholerecoverypercent_obsDist_30 = $", wholerecoverypercent_obsDist_03, "/pm", sigmaobsDist_30, "$")
print("wholerecoverypercent_obsDist_100 = $", wholerecoverypercent_obsDist_100, "/pm", sigmaobsDist_100, "$")
print("wholerecoverypercent_obsDist_1000 = $", wholerecoverypercent_obsDist_1000, "/pm", sigmaobsDist_1000, "$")

print("********************************")

print("overallrecoverypercent_obsDist = $", overallrecoverypercent_obsDist, "/pm", sigmaobsDist, "$")
print("overallrecoverypercent_obsDist_03 = $", overallrecoverypercent_obsDist_03, "/pm", sigmaobsDist_03, "$")
print("overallrecoverypercent_obsDist_1 = $", overallrecoverypercent_obsDist_1, "/pm", sigmaobsDist_1, "$")
print("overallrecoverypercent_obsDist_10 = $", overallrecoverypercent_obsDist_10, "/pm", sigmaobsDist_10, "$")
print("overallrecoverypercent_obsDist_30 = $", overallrecoverypercent_obsDist_03, "/pm", sigmaobsDist_30, "$")
print("overallrecoverypercent_obsDist_100 = $", overallrecoverypercent_obsDist_100, "/pm", sigmaobsDist_100, "$")
print("overallrecoverypercent_obsDist_1000 = $", overallrecoverypercent_obsDist_1000, "/pm", sigmaobsDist_1000, "$")



print("################################")

print("N_totalobsDist22 = ", N_totalobsDist22, "and in log = ", np.log10(N_totalobsDist22), "**** N_totalobservableobsDist22 = ", N_totalobservableobsDist22, "and in log = ", np.log10(N_totalobservableobsDist22), "**** N_totalrecoverableobsDist22 = ", N_totalrecoverableobsDist22, "and in log = ", np.log10(N_totalrecoverableobsDist22))
print("N_totalobsDist22_03 = ", N_totalobsDist22_03, "and in log = ", np.log10(N_totalobsDist22_03), "**** N_totalobservableobsDist22_03 = ", N_totalobservableobsDist22_03, "and in log = ", np.log10(N_totalobservableobsDist22_03), "**** N_totalrecoverableobsDist22_03 = ", N_totalrecoverableobsDist22_03, "and in log = ", np.log10(N_totalrecoverableobsDist22_03))
print("N_totalobsDist22_1 = ", N_totalobsDist22_1, "and in log = ", np.log10(N_totalobsDist22_1), "**** N_totalobservableobsDist22_1 = ", N_totalobservableobsDist22_1, "and in log = ", np.log10(N_totalobservableobsDist22_1), "**** N_totalrecoverableobsDist22_1 = ", N_totalrecoverableobsDist22_1, "and in log = ", np.log10(N_totalrecoverableobsDist22_1))
print("N_totalobsDist22_10 = ", N_totalobsDist22_10, "and in log = ", np.log10(N_totalobsDist22_10), "**** N_totalobservableobsDist22_10 = ", N_totalobservableobsDist22_10, "and in log = ", np.log10(N_totalobservableobsDist22_10), "**** N_totalrecoverableobsDist22_10 = ", N_totalrecoverableobsDist22_10, "and in log = ", np.log10(N_totalrecoverableobsDist22_10))
print("N_totalobsDist22_30 = ", N_totalobsDist22_30, "and in log = ", np.log10(N_totalobsDist22_30), "**** N_totalobservableobsDist22_30 = ", N_totalobservableobsDist22_30, "and in log = ", np.log10(N_totalobservableobsDist22_30), "**** N_totalrecoverableobsDist22_30 = ", N_totalrecoverableobsDist22_30, "and in log = ", np.log10(N_totalrecoverableobsDist22_30))
print("N_totalobsDist22_100 = ", N_totalobsDist22_100, "and in log = ", np.log10(N_totalobsDist22_100), "**** N_totalobservableobsDist22_100 = ", N_totalobservableobsDist22_100, "and in log = ", np.log10(N_totalobservableobsDist22_100), "**** N_totalrecoverableobsDist22_100 = ", N_totalrecoverableobsDist22_100, "and in log = ", np.log10(N_totalrecoverableobsDist22_100))
print("N_totalobsDist22_1000 = ", N_totalobsDist22_1000, "and in log = ", np.log10(N_totalobsDist22_1000), "**** N_totalobservableobsDist22_1000 = ", N_totalobservableobsDist22_1000, "and in log = ", np.log10(N_totalobservableobsDist22_1000), "**** N_totalrecoverableobsDist22_1000 = ", N_totalrecoverableobsDist22_1000, "and in log = ", np.log10(N_totalrecoverableobsDist22_1000))

print("********************************")

print("wholerecoverypercent_obsDist22 = $", wholerecoverypercent_obsDist22, "/pm", sigmaobsDist22, "$")
print("wholerecoverypercent_obsDist22_03 = $", wholerecoverypercent_obsDist22_03, "/pm", sigmaobsDist22_03, "$")
print("wholerecoverypercent_obsDist22_1 = $", wholerecoverypercent_obsDist22_1, "/pm", sigmaobsDist22_1, "$")
print("wholerecoverypercent_obsDist22_10 = $", wholerecoverypercent_obsDist22_10, "/pm", sigmaobsDist22_10, "$")
print("wholerecoverypercent_obsDist22_30 = $", wholerecoverypercent_obsDist22_03, "/pm", sigmaobsDist22_30, "$")
print("wholerecoverypercent_obsDist22_100 = $", wholerecoverypercent_obsDist22_100, "/pm", sigmaobsDist22_100, "$")
print("wholerecoverypercent_obsDist22_1000 = $", wholerecoverypercent_obsDist22_1000, "/pm", sigmaobsDist22_1000, "$")

print("********************************")

print("overallrecoverypercent_obsDist22 = $", overallrecoverypercent_obsDist22, "/pm", sigmaobsDist22, "$")
print("overallrecoverypercent_obsDist22_03 = $", overallrecoverypercent_obsDist22_03, "/pm", sigmaobsDist22_03, "$")
print("overallrecoverypercent_obsDist22_1 = $", overallrecoverypercent_obsDist22_1, "/pm", sigmaobsDist22_1, "$")
print("overallrecoverypercent_obsDist22_10 = $", overallrecoverypercent_obsDist22_10, "/pm", sigmaobsDist22_10, "$")
print("overallrecoverypercent_obsDist22_30 = $", overallrecoverypercent_obsDist22_03, "/pm", sigmaobsDist22_30, "$")
print("overallrecoverypercent_obsDist22_100 = $", overallrecoverypercent_obsDist22_100, "/pm", sigmaobsDist22_100, "$")
print("overallrecoverypercent_obsDist22_1000 = $", overallrecoverypercent_obsDist22_1000, "/pm", sigmaobsDist22_1000, "$")


print("###############################")

print("N_totalobsDist195 = ", N_totalobsDist195, "and in log = ", np.log10(N_totalobsDist195), "**** N_totalobservableobsDist195 = ", N_totalobservableobsDist195, "and in log = ", np.log10(N_totalobservableobsDist195), "**** N_totalrecoverableobsDist195 = ", N_totalrecoverableobsDist195, "and in log = ", np.log10(N_totalrecoverableobsDist195))
print("N_totalobsDist195_03 = ", N_totalobsDist195_03, "and in log = ", np.log10(N_totalobsDist195_03), "**** N_totalobservableobsDist195_03 = ", N_totalobservableobsDist195_03, "and in log = ", np.log10(N_totalobservableobsDist195_03), "**** N_totalrecoverableobsDist195_03 = ", N_totalrecoverableobsDist195_03, "and in log = ", np.log10(N_totalrecoverableobsDist195_03))
print("N_totalobsDist195_1 = ", N_totalobsDist195_1, "and in log = ", np.log10(N_totalobsDist195_1), "**** N_totalobservableobsDist195_1 = ", N_totalobservableobsDist195_1, "and in log = ", np.log10(N_totalobservableobsDist195_1), "**** N_totalrecoverableobsDist195_1 = ", N_totalrecoverableobsDist195_1, "and in log = ", np.log10(N_totalrecoverableobsDist195_1))
print("N_totalobsDist195_10 = ", N_totalobsDist195_10, "and in log = ", np.log10(N_totalobsDist195_10), "**** N_totalobservableobsDist195_10 = ", N_totalobservableobsDist195_10, "and in log = ", np.log10(N_totalobservableobsDist195_10), "**** N_totalrecoverableobsDist195_10 = ", N_totalrecoverableobsDist195_10, "and in log = ", np.log10(N_totalrecoverableobsDist195_10))
print("N_totalobsDist195_30 = ", N_totalobsDist195_30, "and in log = ", np.log10(N_totalobsDist195_30), "**** N_totalobservableobsDist195_30 = ", N_totalobservableobsDist195_30, "and in log = ", np.log10(N_totalobservableobsDist195_30), "**** N_totalrecoverableobsDist195_30 = ", N_totalrecoverableobsDist195_30, "and in log = ", np.log10(N_totalrecoverableobsDist195_30))
print("N_totalobsDist195_100 = ", N_totalobsDist195_100, "and in log = ", np.log10(N_totalobsDist195_100), "**** N_totalobservableobsDist195_100 = ", N_totalobservableobsDist195_100, "and in log = ", np.log10(N_totalobservableobsDist195_100), "**** N_totalrecoverableobsDist195_100 = ", N_totalrecoverableobsDist195_100, "and in log = ", np.log10(N_totalrecoverableobsDist195_100))
print("N_totalobsDist195_1000 = ", N_totalobsDist195_1000, "and in log = ", np.log10(N_totalobsDist195_1000), "**** N_totalobservableobsDist195_1000 = ", N_totalobservableobsDist195_1000, "and in log = ", np.log10(N_totalobservableobsDist195_1000), "**** N_totalrecoverableobsDist195_1000 = ", N_totalrecoverableobsDist195_1000, "and in log = ", np.log10(N_totalrecoverableobsDist195_1000))

print("********************************")

print("wholerecoverypercent_obsDist195 = $", wholerecoverypercent_obsDist195, "/pm", sigmaobsDist195, "$")
print("wholerecoverypercent_obsDist195_03 = $", wholerecoverypercent_obsDist195_03, "/pm", sigmaobsDist195_03, "$")
print("wholerecoverypercent_obsDist195_1 = $", wholerecoverypercent_obsDist195_1, "/pm", sigmaobsDist195_1, "$")
print("wholerecoverypercent_obsDist195_10 = $", wholerecoverypercent_obsDist195_10, "/pm", sigmaobsDist195_10, "$")
print("wholerecoverypercent_obsDist195_30 = $", wholerecoverypercent_obsDist195_03, "/pm", sigmaobsDist195_30, "$")
print("wholerecoverypercent_obsDist195_100 = $", wholerecoverypercent_obsDist195_100, "/pm", sigmaobsDist195_100, "$")
print("wholerecoverypercent_obsDist195_1000 = $", wholerecoverypercent_obsDist195_1000, "/pm", sigmaobsDist195_1000, "$")

print("********************************")

print("overallrecoverypercent_obsDist195 = $", overallrecoverypercent_obsDist195, "/pm", sigmaobsDist195, "$")
print("overallrecoverypercent_obsDist195_03 = $", overallrecoverypercent_obsDist195_03, "/pm", sigmaobsDist195_03, "$")
print("overallrecoverypercent_obsDist195_1 = $", overallrecoverypercent_obsDist195_1, "/pm", sigmaobsDist195_1, "$")
print("overallrecoverypercent_obsDist195_10 = $", overallrecoverypercent_obsDist195_10, "/pm", sigmaobsDist195_10, "$")
print("overallrecoverypercent_obsDist195_30 = $", overallrecoverypercent_obsDist195_03, "/pm", sigmaobsDist195_30, "$")
print("overallrecoverypercent_obsDist195_100 = $", overallrecoverypercent_obsDist195_100, "/pm", sigmaobsDist195_100, "$")
print("overallrecoverypercent_obsDist195_1000 = $", overallrecoverypercent_obsDist195_1000, "/pm", sigmaobsDist195_1000, "$")

print("#############################")





print("binarypercent_22 = $", (N_totalobsDist22/N_totalobsDist)*100, "/pm", ((N_totalobsDist22**(1/2))/N_totalobsDist)*100, "$")
print("binarypercent_195 = $", (N_totalobsDist195/N_totalobsDist)*100, "/pm", ((N_totalobsDist195**(1/2))/N_totalobsDist)*100, "$")

print("binarypercent_03 = $", (N_totalobsDist_03/N_totalobsDist)*100, "/pm", ((N_totalobsDist_03**(1/2))/N_totalobsDist)*100, "$")
print("binarypercent_1 = $", (N_totalobsDist_1/N_totalobsDist)*100, "/pm", ((N_totalobsDist_1**(1/2))/N_totalobsDist)*100, "$")
print("binarypercent_10 = $", (N_totalobsDist_10/N_totalobsDist)*100, "/pm", ((N_totalobsDist_10**(1/2))/N_totalobsDist)*100, "$")
print("binarypercent_30 = $", (N_totalobsDist_30/N_totalobsDist)*100, "/pm", ((N_totalobsDist_30**(1/2))/N_totalobsDist)*100, "$")
print("binarypercent_100 = $", (N_totalobsDist_100/N_totalobsDist)*100, "/pm", ((N_totalobsDist_100**(1/2))/N_totalobsDist)*100, "$")
print("binarypercent_1000 = $", (N_totalobsDist_1000/N_totalobsDist)*100, "/pm", ((N_totalobsDist_1000**(1/2))/N_totalobsDist)*100, "$")

print("observablepercent_03 = $", (N_totalobservableobsDist_03/N_totalobsDist_03)*100, "/pm", ((N_totalobservableobsDist_03**(1/2))/N_totalobsDist_03)*100, "$")
print("observablepercent_1 = $", (N_totalobservableobsDist_1/N_totalobsDist_1)*100, "/pm", ((N_totalobservableobsDist_1**(1/2))/N_totalobsDist_1)*100, "$")
print("observablepercent_10 = $", (N_totalobservableobsDist_10/N_totalobsDist_10)*100, "/pm", ((N_totalobservableobsDist_10**(1/2))/N_totalobsDist_10)*100, "$")
print("observablepercent_30 = $", (N_totalobservableobsDist_30/N_totalobsDist_30)*100, "/pm", ((N_totalobservableobsDist_30**(1/2))/N_totalobsDist_30)*100, "$")
print("observablepercent_100 = $", (N_totalobservableobsDist_100/N_totalobsDist_100)*100, "/pm", ((N_totalobservableobsDist_100**(1/2))/N_totalobsDist_100)*100, "$")
print("observablepercent_1000 = $", (N_totalobservableobsDist_1000/N_totalobsDist_1000)*100, "/pm", ((N_totalobservableobsDist_1000**(1/2))/N_totalobsDist_1000)*100, "$")

print("observablepercent = $", (N_totalobservableobsDist/N_totalobsDist)*100, "/pm", ((N_totalobservableobsDist**(1/2))/N_totalobsDist)*100, "$")
print("observablepercent22 = $", (N_totalobservableobsDist22/N_totalobsDist22)*100, "/pm", ((N_totalobservableobsDist22**(1/2))/N_totalobsDist22)*100, "$")
print("observablepercent195 = $", (N_totalobservableobsDist195/N_totalobsDist195)*100, "/pm", ((N_totalobservableobsDist195**(1/2))/N_totalobsDist195)*100, "$")



