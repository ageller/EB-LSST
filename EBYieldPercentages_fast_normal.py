import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from matplotlib import cm, colors
from astropy.modeling import models, fitting

cmap = cm.ScalarMappable(colors.Normalize(1, 5200), cm.viridis)

# Reading in all data files at once
import glob
path_normal ='/projects/p30137/ageller/testing/EBLSST/add_m5/output_files' 
allFiles_normal = glob.glob(path_normal + "/*.csv")
path_fast = '/projects/p30137/ageller/testing/EBLSST/add_m5/fast/old/output_files'
allFiles_fast = glob.glob(path_fast + "/*.csv")
path_obsDist = '/projects/p30137/ageller/testing/EBLSST/add_m5/fast/old/obsDist/output_files'
allFiles_obsDist = glob.glob(path_obsDist + "/*.csv")

#will want to remove old when the updates come in from Katie

#normal =[]
#fast=[]
#obsDist = []

#normal_03 =[]
#fast_03=[]
#obsDist_03 = []

#normal_1 =[]
#fast_1 =[]
#obsDist_1 = []

#normal_10 =[]
#fast_10=[]
#obsDist_10 = []

#normal_30 =[]
#fast_30=[]
#obsDist_30 = []

#normal_100 =[]
#fast_100 =[]
#obsDist_100 = []

#normal_1000 =[]
#fast_1000 =[]
#obsDist_1000 = []

#normal_overall =[]
#fast_overall=[]
#obsDist_overall = []

#normal_overall_03 =[]
#fast_overall_03=[]
#obsDist_overall_03 = []

#normal_overall_1 =[]
#fast_overall_1 =[]
#obsDist_overall_1 = []

#normal_overall_10 =[]
#fast_overall_10=[]
#obsDist_overall_10 = []

#normal_overall_30 =[]
#fast_overall_30=[]
#obsDist_overall_30 = []

#normal_overall_100 =[]
#fast_overall_100 =[]
#obsDist_overall_100 = []

#normal_overall_1000=[]
#fast_overall_1000 =[]
#obsDist_overall_1000 = []



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

colorvalue_normal = []
colorvalue_fast = []
colorvalue_obsDist = []


def fitRagfb():
	x = [0.05, 0.1, 1, 8, 15]  #estimates of midpoints in bins, and using this: https://sites.uni.edu/morgans/astro/course/Notes/section2/spectralmasses.html
	y = [0.20, 0.35, 0.50, 0.70, 0.75]
	init = models.PowerLaw1D(amplitude=0.5, x_0=1, alpha=-1.)
	fitter = fitting.LevMarLSQFitter()
	fit = fitter(init, x, y)

	return fit


fbFit= fitRagfb()
mbins = np.arange(0,10, 0.1, dtype='float')

for filenormal_ in sorted(allFiles_normal):

	filename1 = filenormal_[60:]
	fileid1 = filename1.strip('output_file.csv')
	colorvalue1 = int(fileid1)
	colorvalue_normal.append(colorvalue1)
	print ("I'm starting " + fileid1)


	datnormal = pd.read_csv(filenormal_, sep = ',', header=2)

	##########################################################

	datnormal1 = pd.read_csv(filenormal_, sep = ',', header=0, nrows=1)
	N_tri1 = datnormal1["NstarsTRILEGAL"][0]
	print("N_tri1 = ", N_tri1)


	m1hAll01, m1b1 = np.histogram(datnormal["m1"], bins=mbins)
	dm11 = np.diff(m1b1)
	m1val1 = m1b1[:-1] + dm11/2.

	fb1 = np.sum(m1hAll01*dm11*fbFit(m1val1))

	N_mult1 = N_tri1*fb1

	##########################################################
	PeriodIn1 = datnormal['p']
	if len(PeriodIn1) == 0.:
		continue
	if N_tri1 == 0:
		continue
	else:

		 # input period -- 'p' in data file
		print('length period in = ', len(PeriodIn1))

		PeriodOut1 = datnormal['LSM_PERIOD'] #LSM_PERIOD in data file
		appMagMean1 = datnormal['appMagMean'] #apparent magnitude, will use to make cuts for 24 (default), 22, and then Kepler's range (?? -- brighter than LSST can manage-- to 19) OR 19.5 (SNR = 10)
		print('length period out = ', len(PeriodOut1))
		observable1 = np.where(PeriodOut1 != -999)[0]
		observable1_03 = np.where(PeriodIn1[observable1] <= 0.3)[0]
		observable1_1 = np.where(PeriodIn1[observable1] <= 1)[0]
		observable1_10 = np.where(PeriodIn1[observable1] <= 10)[0]
		observable1_30 = np.where(PeriodIn1[observable1] <= 30)[0]
		observable1_100 = np.where(PeriodIn1[observable1] <= 100)[0]
		observable1_1000 = np.where(PeriodIn1[observable1] <= 1000)[0]

		observable1_22 = np.where(appMagMean1[observable1] <= 22.)[0]
		observable1_03_22 = np.where(appMagMean1[observable1_03] <= 22.)[0]
		observable1_1_22 = np.where(appMagMean1[observable1_1] <= 22.)[0]
		observable1_10_22 = np.where(appMagMean1[observable1_10] <= 22.)[0]
		observable1_30_22 = np.where(appMagMean1[observable1_30] <= 22.)[0]
		observable1_100_22 = np.where(appMagMean1[observable1_100] <= 22.)[0]
		observable1_1000_22 = np.where(appMagMean1[observable1_1000] <= 22.)[0]

		observable1_195 = np.where(appMagMean1[observable1] <= 19.5)[0]
		observable1_03_195 = np.where(appMagMean1[observable1_03] <= 19.5)[0]
		observable1_1_195 = np.where(appMagMean1[observable1_1] <= 19.5)[0]
		observable1_10_195 = np.where(appMagMean1[observable1_10] <= 19.5)[0]
		observable1_30_195 = np.where(appMagMean1[observable1_30] <= 19.5)[0]
		observable1_100_195 = np.where(appMagMean1[observable1_100] <= 19.5)[0]
		observable1_1000_195 = np.where(appMagMean1[observable1_1000] <= 19.5)[0]


		Sigma_Period_Whole1 = abs(PeriodOut1 - PeriodIn1)/PeriodIn1
		Sigma_Period_Half1 = abs(PeriodOut1 - 0.5*PeriodIn1)/(0.5*PeriodIn1)
		Sigma_Period_Twice1 = abs(PeriodOut1 - 2*PeriodIn1)/(2*PeriodIn1)

		#print(type(Sigma_Period_Twice1))
		#print("Sigma_Period_Twice1: ", Sigma_Period_Twice1)
		#print(type(Sigma_Period_Half1))
		#print("Sigma_Period_Half1: ", Sigma_Period_Half1)
		#print(type(Sigma_Period_Whole1))
		#print("Sigma_Period_Whole1: ", Sigma_Period_Whole1)

		#recover_twice1 = np.where(np.logical_and(np.isfinite(Sigma_Period_Twice1), Sigma_Period_Twice1 <= 0.1))[0]
		recover_twice1 = np.where(Sigma_Period_Twice1 <= 0.1)[0]
		recover_twice1_03 = np.where(PeriodIn1[recover_twice1] <= 0.3)[0]
		recover_twice1_1 = np.where(PeriodIn1[recover_twice1] <= 1)[0]
		recover_twice1_10 = np.where(PeriodIn1[recover_twice1] <= 10)[0]
		recover_twice1_30 = np.where(PeriodIn1[recover_twice1] <= 30)[0]
		recover_twice1_100 = np.where(PeriodIn1[recover_twice1] <= 100)[0]
		recover_twice1_1000 = np.where(PeriodIn1[recover_twice1] <= 1000)[0]
		#recover_half1 = np.where(np.logical_and(np.isfinite(Sigma_Period_Half1), Sigma_Period_Half1 <= 0.1))[0]
		recover_half1 = np.where(Sigma_Period_Half1 <= 0.1)[0]
		recover_half1_03 = np.where(PeriodIn1[recover_half1] <= 0.3)[0]
		recover_half1_1 = np.where(PeriodIn1[recover_half1] <= 1)[0]
		recover_half1_10 = np.where(PeriodIn1[recover_half1] <= 10)[0]
		recover_half1_30 = np.where(PeriodIn1[recover_half1] <= 30)[0]
		recover_half1_100 = np.where(PeriodIn1[recover_half1] <= 100)[0]
		recover_half1_1000 = np.where(PeriodIn1[recover_half1] <= 1000)[0]
		#recover_whole1 = np.where(np.logical_and(np.isfinite(Sigma_Period_Whole1), Sigma_Period_Whole1 <= 0.1))[0]
		recover_whole1 = np.where(Sigma_Period_Whole1 <= 0.1)[0]
		recover_whole1_03 = np.where(PeriodIn1[recover_whole1] <= 0.3)[0]
		recover_whole1_1 = np.where(PeriodIn1[recover_whole1] <= 1)[0]
		recover_whole1_10 = np.where(PeriodIn1[recover_whole1] <= 10)[0]
		recover_whole1_30 = np.where(PeriodIn1[recover_whole1] <= 30)[0]
		recover_whole1_100 = np.where(PeriodIn1[recover_whole1] <= 100)[0]
		recover_whole1_1000 = np.where(PeriodIn1[recover_whole1] <= 1000)[0]

		recoverable1 = np.concatenate((recover_twice1, recover_whole1, recover_half1), axis=0)
		recoverable1_03 = np.concatenate((recover_twice1_03, recover_whole1_03, recover_half1_03), axis=0)
		recoverable1_1 = np.concatenate((recover_twice1_1, recover_whole1_1, recover_half1_1), axis=0)
		recoverable1_10 = np.concatenate((recover_twice1_10, recover_whole1_10, recover_half1_10), axis=0)
		recoverable1_30 = np.concatenate((recover_twice1_30, recover_whole1_30, recover_half1_30), axis=0)
		recoverable1_100 = np.concatenate((recover_twice1_100, recover_whole1_100, recover_half1_100), axis=0)
		recoverable1_1000 = np.concatenate((recover_twice1_1000, recover_whole1_1000, recover_half1_1000), axis=0)

		recoverable1_22 = np.where(appMagMean1[recoverable1] <= 22.)[0]
		recoverable1_03_22 = np.where(appMagMean1[recoverable1_03] <= 22.)[0]
		recoverable1_1_22 = np.where(appMagMean1[recoverable1_1] <= 22.)[0]
		recoverable1_10_22 = np.where(appMagMean1[recoverable1_10] <= 22.)[0]
		recoverable1_30_22 = np.where(appMagMean1[recoverable1_30] <= 22.)[0]
		recoverable1_100_22 = np.where(appMagMean1[recoverable1_100] <= 22.)[0]
		recoverable1_1000_22 = np.where(appMagMean1[recoverable1_1000] <= 22.)[0]
		
		recoverable1_195 = np.where(appMagMean1[recoverable1] <= 19.5)[0]
		recoverable1_03_195 = np.where(appMagMean1[recoverable1_03] <= 19.5)[0]
		recoverable1_1_195 = np.where(appMagMean1[recoverable1_1] <= 19.5)[0]
		recoverable1_10_195 = np.where(appMagMean1[recoverable1_10] <= 19.5)[0]
		recoverable1_30_195 = np.where(appMagMean1[recoverable1_30] <= 19.5)[0]
		recoverable1_100_195 = np.where(appMagMean1[recoverable1_100] <= 19.5)[0]
		recoverable1_1000_195 = np.where(appMagMean1[recoverable1_1000] <= 19.5)[0]

		P03 = np.where(PeriodIn1 <= 0.3)[0]
		P1 = np.where(PeriodIn1 <= 1)[0]
		P10 = np.where(PeriodIn1 <= 10)[0]
		P30 = np.where(PeriodIn1 <= 30)[0]
		P100 = np.where(PeriodIn1 <= 100)[0]
		P1000 = np.where(PeriodIn1 <= 1000)[0]

		N_all1 = float(len(PeriodIn1)) #unnormalized
		N_all1_03 = float(len(P03))
		N_all1_1 = float(len(P1))
		N_all1_10 = float(len(P10))
		N_all1_30 = float(len(P30))
		N_all1_100 = float(len(P100))
		N_all1_1000 = float(len(P1000))

		N_all1_22 = float(len(np.where(appMagMean1 <= 22))) #unnormalized
		N_all1_03_22 = float(len(np.where(appMagMean1[P03] <= 22)))
		N_all1_1_22 = float(len(np.where(appMagMean1[P1] <= 22)))
		N_all1_10_22 = float(len(np.where(appMagMean1[P10] <= 22)))
		N_all1_30_22 = float(len(np.where(appMagMean1[P30] <= 22)))
		N_all1_100_22 = float(len(np.where(appMagMean1[P100] <= 22)))
		N_all1_1000_22 = float(len(np.where(appMagMean1[P1000] <= 22)))

		N_all1_195 = float(len(np.where(appMagMean1 <= 19.5))) #unnormalized
		N_all1_03_195 = float(len(np.where(appMagMean1[P03] <= 19.5)))
		N_all1_1_195 = float(len(np.where(appMagMean1[P1] <= 19.5)))
		N_all1_10_195 = float(len(np.where(appMagMean1[P10] <= 19.5)))
		N_all1_30_195 = float(len(np.where(appMagMean1[P30] <= 19.5)))
		N_all1_100_195 = float(len(np.where(appMagMean1[P100] <= 19.5)))
		N_all1_1000_195 = float(len(np.where(appMagMean1[P1000] <= 19.5)))

		#NORMALIZED FROM HERE vv


		N_all1_norm = (N_all1/N_all1)*N_mult1 #normalized
		N_all1_03norm = (N_all1_03/N_all1)*N_mult1
		N_all1_1norm = (N_all1_1/N_all1)*N_mult1
		N_all1_10norm = (N_all1_10/N_all1)*N_mult1
		N_all1_30norm = (N_all1_30/N_all1)*N_mult1
		N_all1_100norm = (N_all1_100/N_all1)*N_mult1
		N_all1_1000norm = (N_all1_1000/N_all1)*N_mult1

		N_observable1 = (float(len(observable1))/float(N_all1))*N_mult1
		N_observable1_03 = (float(len(observable1_03))/float(N_all1))*N_mult1
		N_observable1_1 = (float(len(observable1_1))/float(N_all1))*N_mult1
		N_observable1_10 = (float(len(observable1_10))/float(N_all1))*N_mult1
		N_observable1_30 = (float(len(observable1_30))/float(N_all1))*N_mult1
		N_observable1_100 = (float(len(observable1_100))/float(N_all1))*N_mult1
		N_observable1_1000 = (float(len(observable1_1000))/float(N_all1))*N_mult1

		N_recoverable1 = (float(len(recoverable1))/float(N_all1))*N_mult1
		N_recoverable1_03 = (float(len(recoverable1_03))/float(N_all1))*N_mult1
		N_recoverable1_1 = (float(len(recoverable1_1))/float(N_all1))*N_mult1
		N_recoverable1_10 = (float(len(recoverable1_10))/float(N_all1))*N_mult1
		N_recoverable1_30 = (float(len(recoverable1_30))/float(N_all1))*N_mult1
		N_recoverable1_100 = (float(len(recoverable1_100))/float(N_all1))*N_mult1
		N_recoverable1_1000 = (float(len(recoverable1_1000))/float(N_all1))*N_mult1


		N_all1_norm_22 = (N_all1_22/N_all1)*N_mult1 #normalized
		N_all1_03norm_22 = (N_all1_03_22/N_all1)*N_mult1
		N_all1_1norm_22 = (N_all1_1_22/N_all1)*N_mult1
		N_all1_10norm_22 = (N_all1_10_22/N_all1)*N_mult1
		N_all1_30norm_22 = (N_all1_30_22/N_all1)*N_mult1
		N_all1_100norm_22 = (N_all1_100_22/N_all1)*N_mult1
		N_all1_1000norm_22 = (N_all1_1000_22/N_all1)*N_mult1

		N_observable1_22 = (float(len(observable1_22))/float(N_all1))*N_mult1
		N_observable1_03_22 = (float(len(observable1_03_22))/float(N_all1))*N_mult1
		N_observable1_1_22 = (float(len(observable1_1_22))/float(N_all1))*N_mult1
		N_observable1_10_22 = (float(len(observable1_10_22))/float(N_all1))*N_mult1
		N_observable1_30_22 = (float(len(observable1_30_22))/float(N_all1))*N_mult1
		N_observable1_100_22 = (float(len(observable1_100_22))/float(N_all1))*N_mult1
		N_observable1_1000_22 = (float(len(observable1_1000_22))/float(N_all1))*N_mult1

		N_recoverable1_22 = (float(len(recoverable1_22))/float(N_all1))*N_mult1
		N_recoverable1_03_22 = (float(len(recoverable1_03_22))/float(N_all1))*N_mult1
		N_recoverable1_1_22 = (float(len(recoverable1_1_22))/float(N_all1))*N_mult1
		N_recoverable1_10_22 = (float(len(recoverable1_10_22))/float(N_all1))*N_mult1
		N_recoverable1_30_22 = (float(len(recoverable1_30_22))/float(N_all1))*N_mult1
		N_recoverable1_100_22 = (float(len(recoverable1_100_22))/float(N_all1))*N_mult1
		N_recoverable1_1000_22 = (float(len(recoverable1_1000_22))/float(N_all1))*N_mult1


		N_all1_norm_195 = (N_all1_195/N_all1)*N_mult1 #normalized
		N_all1_03norm_195 = (N_all1_03_195/N_all1)*N_mult1
		N_all1_1norm_195 = (N_all1_1_195/N_all1)*N_mult1
		N_all1_10norm_195 = (N_all1_10_195/N_all1)*N_mult1
		N_all1_30norm_195 = (N_all1_30_195/N_all1)*N_mult1
		N_all1_100norm_195 = (N_all1_100_195/N_all1)*N_mult1
		N_all1_1000norm_195 = (N_all1_1000_195/N_all1)*N_mult1

		N_observable1_195 = (float(len(observable1_195))/float(N_all1))*N_mult1
		N_observable1_03_195 = (float(len(observable1_03_195))/float(N_all1))*N_mult1
		N_observable1_1_195 = (float(len(observable1_1_195))/float(N_all1))*N_mult1
		N_observable1_10_195 = (float(len(observable1_10_195))/float(N_all1))*N_mult1
		N_observable1_30_195 = (float(len(observable1_30_195))/float(N_all1))*N_mult1
		N_observable1_100_195 = (float(len(observable1_100_195))/float(N_all1))*N_mult1
		N_observable1_1000_195 = (float(len(observable1_1000_195))/float(N_all1))*N_mult1

		N_recoverable1_195 = (float(len(recoverable1_195))/float(N_all1))*N_mult1
		N_recoverable1_03_195 = (float(len(recoverable1_03_195))/float(N_all1))*N_mult1
		N_recoverable1_1_195 = (float(len(recoverable1_1_195))/float(N_all1))*N_mult1
		N_recoverable1_10_195 = (float(len(recoverable1_10_195))/float(N_all1))*N_mult1
		N_recoverable1_30_195 = (float(len(recoverable1_30_195))/float(N_all1))*N_mult1
		N_recoverable1_100_195 = (float(len(recoverable1_100_195))/float(N_all1))*N_mult1
		N_recoverable1_1000_195 = (float(len(recoverable1_1000_195))/float(N_all1))*N_mult1


		N_totalnormal_array.append(float(N_all1_norm))
		N_totalobservablenormal_array.append(float(N_observable1))
		N_totalrecoverablenormal_array.append(float(N_recoverable1))

		N_totalnormal_array_03.append(float(N_all1_03norm))
		N_totalobservablenormal_array_03.append(float(N_observable1_03))
		N_totalrecoverablenormal_array_03.append(float(N_recoverable1_03))

		N_totalnormal_array_1.append(float(N_all1_1norm))
		N_totalobservablenormal_array_1.append(float(N_observable1_1))
		N_totalrecoverablenormal_array_1.append(float(N_recoverable1_1))

		N_totalnormal_array_10.append(float(N_all1_10norm))
		N_totalobservablenormal_array_10.append(float(N_observable1_10))
		N_totalrecoverablenormal_array_10.append(float(N_recoverable1_10))

		N_totalnormal_array_30.append(float(N_all1_30norm))
		N_totalobservablenormal_array_30.append(float(N_observable1_30))
		N_totalrecoverablenormal_array_30.append(float(N_recoverable1_30))

		N_totalnormal_array_100.append(float(N_all1_100norm))
		N_totalobservablenormal_array_100.append(float(N_observable1_100))
		N_totalrecoverablenormal_array_100.append(float(N_recoverable1_100))

		N_totalnormal_array_1000.append(float(N_all1_1000norm))
		N_totalobservablenormal_array_1000.append(float(N_observable1_1000))
		N_totalrecoverablenormal_array_1000.append(float(N_recoverable1_1000))

		N_totalnormal22_array.append(float(N_all1_norm_22))
		N_totalobservablenormal22_array.append(float(N_observable1_22))
		N_totalrecoverablenormal22_array.append(float(N_recoverable1_22))

		N_totalnormal22_array_03.append(float(N_all1_03norm_22))
		N_totalobservablenormal22_array_03.append(float(N_observable1_03_22))
		N_totalrecoverablenormal22_array_03.append(float(N_recoverable1_03_22))

		N_totalnormal22_array_1.append(float(N_all1_1norm_22))
		N_totalobservablenormal22_array_1.append(float(N_observable1_1_22))
		N_totalrecoverablenormal22_array_1.append(float(N_recoverable1_1_22))

		N_totalnormal22_array_10.append(float(N_all1_10norm_22))
		N_totalobservablenormal22_array_10.append(float(N_observable1_10_22))
		N_totalrecoverablenormal22_array_10.append(float(N_recoverable1_10_22))

		N_totalnormal22_array_30.append(float(N_all1_30norm_22))
		N_totalobservablenormal22_array_30.append(float(N_observable1_30_22))
		N_totalrecoverablenormal22_array_30.append(float(N_recoverable1_30_22))

		N_totalnormal22_array_100.append(float(N_all1_100norm_22))
		N_totalobservablenormal22_array_100.append(float(N_observable1_100_22))
		N_totalrecoverablenormal22_array_100.append(float(N_recoverable1_100_22))

		N_totalnormal22_array_1000.append(float(N_all1_1000norm_22))
		N_totalobservablenormal22_array_1000.append(float(N_observable1_1000_22))
		N_totalrecoverablenormal22_array_1000.append(float(N_recoverable1_1000_22))

		N_totalnormal195_array.append(float(N_all1_norm_195))
		N_totalobservablenormal195_array.append(float(N_observable1_195))
		N_totalrecoverablenormal195_array.append(float(N_recoverable1_195))

		N_totalnormal195_array_03.append(float(N_all1_03norm_195))
		N_totalobservablenormal195_array_03.append(float(N_observable1_03_195))
		N_totalrecoverablenormal195_array_03.append(float(N_recoverable1_03_195))

		N_totalnormal195_array_1.append(float(N_all1_1norm_195))
		N_totalobservablenormal195_array_1.append(float(N_observable1_1_195))
		N_totalrecoverablenormal195_array_1.append(float(N_recoverable1_1_195))

		N_totalnormal195_array_10.append(float(N_all1_10norm_195))
		N_totalobservablenormal195_array_10.append(float(N_observable1_10_195))
		N_totalrecoverablenormal195_array_10.append(float(N_recoverable1_10_195))

		N_totalnormal195_array_30.append(float(N_all1_30norm_195))
		N_totalobservablenormal195_array_30.append(float(N_observable1_30_195))
		N_totalrecoverablenormal195_array_30.append(float(N_recoverable1_30_195))

		N_totalnormal195_array_100.append(float(N_all1_100norm_195))
		N_totalobservablenormal195_array_100.append(float(N_observable1_100_195))
		N_totalrecoverablenormal195_array_100.append(float(N_recoverable1_100_195))

		N_totalnormal195_array_1000.append(float(N_all1_1000norm_195))
		N_totalobservablenormal195_array_1000.append(float(N_observable1_1000_195))
		N_totalrecoverablenormal195_array_1000.append(float(N_recoverable1_1000_195))

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

print("N_totalnormal = ", N_totalnormal, "N_totalobservablenormal = ", N_totalobservablenormal, "N_totalrecoverablenormal = ", N_totalrecoverablenormal)

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
print("wholerecoverypercent_normal = ", wholerecoverypercent_normal, "wholerecoverypercent_normal_03 = ", wholerecoverypercent_normal_03, "wholerecoverypercent_normal_1 = ", wholerecoverypercent_normal_1, "wholerecoverypercent_normal_10 = ", wholerecoverypercent_normal_10, "wholerecoverypercent_normal_30 = ", wholerecoverypercent_normal_30, "wholerecoverypercent_normal_100 = ", wholerecoverypercent_normal_100, "wholerecoverypercent_normal_1000 = ", wholerecoverypercent_normal_1000)
print("sigmanormal = ", sigmanormal, "sigmanormal_03 = ", sigmanormal_03, "sigmanormal_1 = ", sigmanormal_1, "sigmanormal_10 = ", sigmanormal_10, "sigmanormal_30 = ", sigmanormal_30, "sigmanormal_100 = ", sigmanormal_100, "sigmanormal_1000 = ", sigmanormal_1000)
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
print("overallrecoverypercent_normal = ", overallrecoverypercent_normal, "overallrecoverypercent_normal_03 = ", overallrecoverypercent_normal_03, "overallrecoverypercent_normal_1 = ", overallrecoverypercent_normal_1, "overallrecoverypercent_normal_10 = ", overallrecoverypercent_normal_10, "overallrecoverypercent_normal_30 = ", overallrecoverypercent_normal_30, "overallrecoverypercent_normal_100 = ", overallrecoverypercent_normal_100, "overallrecoverypercent_normal_1000 = ", overallrecoverypercent_normal_1000)
print("overallsigmanormal = ", overallsigmanormal, "overallsigmanormal_03 = ", overallsigmanormal_03, "overallsigmanormal_1 = ", overallsigmanormal_1, "overallsigmanormal_10 = ", overallsigmanormal_10, "overallsigmanormal_30 = ", overallsigmanormal_30, "overallsigmanormal_100 = ", overallsigmanormal_100, "overallsigmanormal_1000 = ", overallsigmanormal_1000)



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
print("wholerecoverypercent_normal22 = ", wholerecoverypercent_normal22, "wholerecoverypercent_normal22_03 = ", wholerecoverypercent_normal22_03, "wholerecoverypercent_normal22_1 = ", wholerecoverypercent_normal22_1, "wholerecoverypercent_normal22_10 = ", wholerecoverypercent_normal22_10, "wholerecoverypercent_normal22_30 = ", wholerecoverypercent_normal22_30, "wholerecoverypercent_normal22_100 = ", wholerecoverypercent_normal22_100, "wholerecoverypercent_normal22_1000 = ", wholerecoverypercent_normal22_1000)
print("sigmanormal22 = ", sigmanormal22, "sigmanormal22_03 = ", sigmanormal22_03, "sigmanormal22_1 = ", sigmanormal22_1, "sigmanormal22_10 = ", sigmanormal22_10, "sigmanormal22_30 = ", sigmanormal22_30, "sigmanormal22_100 = ", sigmanormal22_100, "sigmanormal22_1000 = ", sigmanormal22_1000)
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
print("overallrecoverypercent_normal22 = ", overallrecoverypercent_normal22, "overallrecoverypercent_normal22_03 = ", overallrecoverypercent_normal22_03, "overallrecoverypercent_normal22_1 = ", overallrecoverypercent_normal22_1, "overallrecoverypercent_normal22_10 = ", overallrecoverypercent_normal22_10, "overallrecoverypercent_normal22_30 = ", overallrecoverypercent_normal22_30, "overallrecoverypercent_normal22_100 = ", overallrecoverypercent_normal22_100, "overallrecoverypercent_normal22_1000 = ", overallrecoverypercent_normal22_1000)
print("overallsigmanormal22 = ", overallsigmanormal22, "overallsigmanormal22_03 = ", overallsigmanormal22_03, "overallsigmanormal22_1 = ", overallsigmanormal22_1, "overallsigmanormal22_10 = ", overallsigmanormal22_10, "overallsigmanormal22_30 = ", overallsigmanormal22_30, "overallsigmanormal22_100 = ", overallsigmanormal22_100, "overallsigmanormal22_1000 = ", overallsigmanormal22_1000)



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
print("wholerecoverypercent_normal195 = ", wholerecoverypercent_normal195, "wholerecoverypercent_normal195_03 = ", wholerecoverypercent_normal195_03, "wholerecoverypercent_normal195_1 = ", wholerecoverypercent_normal195_1, "wholerecoverypercent_normal195_10 = ", wholerecoverypercent_normal195_10, "wholerecoverypercent_normal195_30 = ", wholerecoverypercent_normal195_30, "wholerecoverypercent_normal195_100 = ", wholerecoverypercent_normal195_100, "wholerecoverypercent_normal195_1000 = ", wholerecoverypercent_normal195_1000)
print("sigmanormal195 = ", sigmanormal195, "sigmanormal195_03 = ", sigmanormal195_03, "sigmanormal195_1 = ", sigmanormal195_1, "sigmanormal195_10 = ", sigmanormal195_10, "sigmanormal195_30 = ", sigmanormal195_30, "sigmanormal195_100 = ", sigmanormal195_100, "sigmanormal195_1000 = ", sigmanormal195_1000)
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
overallsigmanormal195_1000 = ((N_totalrecoverablenormal195_1000**(1/2))/N_totalnormal195_1000)*100
print("overallrecoverypercent_normal195 = ", overallrecoverypercent_normal195, "overallrecoverypercent_normal195_03 = ", overallrecoverypercent_normal195_03, "overallrecoverypercent_normal195_1 = ", overallrecoverypercent_normal195_1, "overallrecoverypercent_normal195_10 = ", overallrecoverypercent_normal195_10, "overallrecoverypercent_normal195_30 = ", overallrecoverypercent_normal195_30, "overallrecoverypercent_normal195_100 = ", overallrecoverypercent_normal195_100, "overallrecoverypercent_normal195_1000 = ", overallrecoverypercent_normal195_1000)
print("overallsigmanormal195 = ", overallsigmanormal195, "overallsigmanormal195_03 = ", overallsigmanormal195_03, "overallsigmanormal195_1 = ", overallsigmanormal195_1, "overallsigmanormal195_10 = ", overallsigmanormal195_10, "overallsigmanormal195_30 = ", overallsigmanormal195_30, "overallsigmanormal195_100 = ", overallsigmanormal195_100, "overallsigmanormal195_1000 = ", overallsigmanormal195_1000)



print("binarypercent_22 = ", (N_totalnormal22/N_totalnormal)*100, "+/-", ((N_totalnormal22**(1/2))/N_totalnormal)*100)
print("binarypercent_195 = ", (N_totalnormal195/N_totalnormal)*100, "+/-", ((N_totalnormal195**(1/2))/N_totalnormal)*100)

print("binarypercent_03 = ", (N_totalnormal_03/N_totalnormal)*100, "+/-", ((N_totalnormal_03**(1/2))/N_totalnormal)*100)
print("binarypercent_1 = ", (N_totalnormal_1/N_totalnormal)*100, "+/-", ((N_totalnormal_1**(1/2))/N_totalnormal)*100)
print("binarypercent_10 = ", (N_totalnormal_10/N_totalnormal)*100, "+/-", ((N_totalnormal_10**(1/2))/N_totalnormal)*100)
print("binarypercent_30 = ", (N_totalnormal_30/N_totalnormal)*100, "+/-", ((N_totalnormal_30**(1/2))/N_totalnormal)*100)
print("binarypercent_100 = ", (N_totalnormal_100/N_totalnormal)*100, "+/-", ((N_totalnormal_100**(1/2))/N_totalnormal)*100)
print("binarypercent_1000 = ", (N_totalnormal_1000/N_totalnormal)*100, "+/-", ((N_totalnormal_1000**(1/2))/N_totalnormal)*100)

print("observablepercent_03 = ", (N_totalobservablenormal_03/N_totalnormal_03)*100, "+/-", ((N_totalobservablenormal_03**(1/2))/N_totalnormal_03)*100)
print("observablepercent_1 = ", (N_totalobservablenormal_1/N_totalnormal_1)*100, "+/-", ((N_totalobservablenormal_1**(1/2))/N_totalnormal_1)*100)
print("observablepercent_10 = ", (N_totalobservablenormal_10/N_totalnormal_10)*100, "+/-", ((N_totalobservablenormal_10**(1/2))/N_totalnormal_10)*100)
print("observablepercent_30 = ", (N_totalobservablenormal_30/N_totalnormal_30)*100, "+/-", ((N_totalobservablenormal_30**(1/2))/N_totalnormal_30)*100)
print("observablepercent_100 = ", (N_totalobservablenormal_100/N_totalnormal_100)*100, "+/-", ((N_totalobservablenormal_100**(1/2))/N_totalnormal_100)*100)
print("observablepercent_1000 = ", (N_totalobservablenormal_1000/N_totalnormal_1000)*100, "+/-", ((N_totalobservablenormal_1000**(1/2))/N_totalnormal_1000)*100)

print("observablepercent = ", (N_totalobservablenormal/N_totalnormal)*100, "+/-", ((N_totalobservablenormal**(1/2))/N_totalnormal)*100)
print("observablepercent22 = ", (N_totalobservablenormal22/N_totalnormal22)*100, "+/-", ((N_totalobservablenormal22**(1/2))/N_totalnormal22)*100)
print("observablepercent195 = ", (N_totalobservablenormal195/N_totalnormal195)*100, "+/-", ((N_totalobservablenormal195**(1/2))/N_totalnormal195)*100)

for filefast_ in sorted(allFiles_fast):
	
	filename2 = filefast_[69:] #when file path no longer has /old in it, will be filefast_[65:]
	fileid2 = filename2.strip('output_file.csv')
	colorvalue2 = int(fileid2)
	colorvalue_fast.append(colorvalue2)
	print ("I'm starting " + fileid2)


	datfast = pd.read_csv(filefast_, sep = ',', header=2)

	##########################################################

	datfast1 = pd.read_csv(filefast_, sep = ',', header=0, nrows=1)
	N_tri2 = datfast1["NstarsTRILEGAL"][0]
	print("N_tri2 = ", N_tri2)

	m1hAll02, m1b2 = np.histogram(datfast["m1"], bins=mbins)
	dm12 = np.diff(m1b2)
	m1val2 = m1b2[:-1] + dm12/2.

	fb2 = np.sum(m1hAll02*dm12*fbFit(m1val2))

	N_mult2 = N_tri2*fb2
	
	##########################################################
	PeriodIn2 = datfast['p'] # input period -- 'p' in data file
	if len(PeriodIn2) == 0.:
		continue
	if N_tri2 == 0:
		continue
	else:


		PeriodOut2 = datfast['LSM_PERIOD'] #LSM_PERIOD in data file
		appMagMean2 = datfast['appMagMean'] #when file path is back to fast/output_files vs. fast/old/output_files, this should be shanged to appMagMean_r
		observable2 = np.where(PeriodOut2 != -999)[0]
		observable2_03 = np.where(PeriodIn2[observable2] <= 0.3)[0]
		observable2_1 = np.where(PeriodIn2[observable2] <= 1)[0]
		observable2_10 = np.where(PeriodIn2[observable2] <= 10)[0]
		observable2_30 = np.where(PeriodIn2[observable2] <= 30)[0]
		observable2_100 = np.where(PeriodIn2[observable2] <= 100)[0]
		observable2_1000 = np.where(PeriodIn2[observable2] <= 1000)[0]

		observable2_22 = np.where(appMagMean2[observable2] <= 22.)[0]
		observable2_03_22 = np.where(appMagMean2[observable2_03] <= 22.)[0]
		observable2_1_22 = np.where(appMagMean2[observable2_1] <= 22.)[0]
		observable2_10_22 = np.where(appMagMean2[observable2_10] <= 22.)[0]
		observable2_30_22 = np.where(appMagMean2[observable2_30] <= 22.)[0]
		observable2_100_22 = np.where(appMagMean2[observable2_100] <= 22.)[0]
		observable2_1000_22 = np.where(appMagMean2[observable2_1000] <= 22.)[0]

		observable2_195 = np.where(appMagMean2[observable2] <= 19.5)[0]
		observable2_03_195 = np.where(appMagMean2[observable2_03] <= 19.5)[0]
		observable2_1_195 = np.where(appMagMean2[observable2_1] <= 19.5)[0]
		observable2_10_195 = np.where(appMagMean2[observable2_10] <= 19.5)[0]
		observable2_30_195 = np.where(appMagMean2[observable2_30] <= 19.5)[0]
		observable2_100_195 = np.where(appMagMean2[observable2_100] <= 19.5)[0]
		observable2_1000_195 = np.where(appMagMean2[observable2_1000] <= 19.5)[0]


		Sigma_Period_Whole2 = abs(PeriodOut2 - PeriodIn2)/PeriodIn2
		Sigma_Period_Half2 = abs(PeriodOut2 - 0.5*PeriodIn2)/(0.5*PeriodIn2)
		Sigma_Period_Twice2 = abs(PeriodOut2 - 2*PeriodIn2)/(2*PeriodIn2)

		recover_twice2 = np.where(Sigma_Period_Twice2 <= 0.1)[0]
		recover_twice2_03 = np.where(PeriodIn2[recover_twice2] <= 0.3)[0]
		recover_twice2_1 = np.where(PeriodIn2[recover_twice2] <= 1)[0]
		recover_twice2_10 = np.where(PeriodIn2[recover_twice2] <= 10)[0]
		recover_twice2_30 = np.where(PeriodIn2[recover_twice2] <= 30)[0]
		recover_twice2_100 = np.where(PeriodIn2[recover_twice2] <= 100)[0]
		recover_twice2_1000 = np.where(PeriodIn2[recover_twice2] <= 1000)[0]
		#recover_half1 = np.where(np.logical_and(np.isfinite(Sigma_Period_Half1), Sigma_Period_Half1 <= 0.1))[0]
		recover_half2 = np.where(Sigma_Period_Half2 <= 0.1)[0]
		recover_half2_03 = np.where(PeriodIn2[recover_half2] <= 0.3)[0]
		recover_half2_1 = np.where(PeriodIn2[recover_half2] <= 1)[0]
		recover_half2_10 = np.where(PeriodIn2[recover_half2] <= 10)[0]
		recover_half2_30 = np.where(PeriodIn2[recover_half2] <= 30)[0]
		recover_half2_100 = np.where(PeriodIn2[recover_half2] <= 100)[0]
		recover_half2_1000 = np.where(PeriodIn2[recover_half2] <= 1000)[0]
		#recover_whole1 = np.where(np.logical_and(np.isfinite(Sigma_Period_Whole1), Sigma_Period_Whole1 <= 0.1))[0]
		recover_whole2 = np.where(Sigma_Period_Whole2 <= 0.1)[0]
		recover_whole2_03 = np.where(PeriodIn2[recover_whole2] <= 0.3)[0]
		recover_whole2_1 = np.where(PeriodIn2[recover_whole2] <= 1)[0]
		recover_whole2_10 = np.where(PeriodIn2[recover_whole2] <= 10)[0]
		recover_whole2_30 = np.where(PeriodIn2[recover_whole2] <= 30)[0]
		recover_whole2_100 = np.where(PeriodIn2[recover_whole2] <= 100)[0]
		recover_whole2_1000 = np.where(PeriodIn2[recover_whole2] <= 1000)[0]

		recoverable2 = np.concatenate((recover_twice2, recover_whole2, recover_half2), axis=0)
		recoverable2_03 = np.concatenate((recover_twice2_03, recover_whole2_03, recover_half2_03), axis=0)
		recoverable2_1 = np.concatenate((recover_twice2_1, recover_whole2_1, recover_half2_1), axis=0)
		recoverable2_10 = np.concatenate((recover_twice2_10, recover_whole2_10, recover_half2_10), axis=0)
		recoverable2_30 = np.concatenate((recover_twice2_30, recover_whole2_30, recover_half2_30), axis=0)
		recoverable2_100 = np.concatenate((recover_twice2_100, recover_whole2_100, recover_half2_100), axis=0)
		recoverable2_1000 = np.concatenate((recover_twice2_1000, recover_whole2_1000, recover_half2_1000), axis=0)

		recoverable2_22 = np.where(appMagMean2[recoverable2] <= 22.)[0]
		recoverable2_03_22 = np.where(appMagMean2[recoverable2_03] <= 22.)[0]
		recoverable2_1_22 = np.where(appMagMean2[recoverable2_1] <= 22.)[0]
		recoverable2_10_22 = np.where(appMagMean2[recoverable2_10] <= 22.)[0]
		recoverable2_30_22 = np.where(appMagMean2[recoverable2_30] <= 22.)[0]
		recoverable2_100_22 = np.where(appMagMean2[recoverable2_100] <= 22.)[0]
		recoverable2_1000_22 = np.where(appMagMean2[recoverable2_1000] <= 22.)[0]
		
		recoverable2_195 = np.where(appMagMean2[recoverable2] <= 19.5)[0]
		recoverable2_03_195 = np.where(appMagMean2[recoverable2_03] <= 19.5)[0]
		recoverable2_1_195 = np.where(appMagMean2[recoverable2_1] <= 19.5)[0]
		recoverable2_10_195 = np.where(appMagMean2[recoverable2_10] <= 19.5)[0]
		recoverable2_30_195 = np.where(appMagMean2[recoverable2_30] <= 19.5)[0]
		recoverable2_100_195 = np.where(appMagMean2[recoverable2_100] <= 19.5)[0]
		recoverable2_1000_195 = np.where(appMagMean2[recoverable2_1000] <= 19.5)[0]

		P03 = np.where(PeriodIn2 <= 0.3)[0]
		P1 = np.where(PeriodIn2 <= 1)[0]
		P10 = np.where(PeriodIn2 <= 10)[0]
		P30 = np.where(PeriodIn2 <= 30)[0]
		P100 = np.where(PeriodIn2 <= 100)[0]
		P1000 = np.where(PeriodIn2 <= 1000)[0]

		N_all2 = float(len(PeriodIn2)) #unnormalized
		N_all2_03 = float(len(P03))
		N_all2_1 = float(len(P1))
		N_all2_10 = float(len(P10))
		N_all2_30 = float(len(P30))
		N_all2_100 = float(len(P100))
		N_all2_1000 = float(len(P1000))


		N_all2_22 = float(len(np.where(appMagMean2 <= 22))) #unnormalized
		N_all2_03_22 = float(len(np.where(appMagMean2[P03] <= 22)))
		N_all2_1_22 = float(len(np.where(appMagMean2[P1] <= 22)))
		N_all2_10_22 = float(len(np.where(appMagMean2[P10] <= 22)))
		N_all2_30_22 = float(len(np.where(appMagMean2[P30] <= 22)))
		N_all2_100_22 = float(len(np.where(appMagMean2[P100] <= 22)))
		N_all2_1000_22 = float(len(np.where(appMagMean2[P1000] <= 22)))

		N_all2_195 = float(len(np.where(appMagMean2 <= 19.5))) #unnormalized
		N_all2_03_195 = float(len(np.where(appMagMean2[P03] <= 19.5)))
		N_all2_1_195 = float(len(np.where(appMagMean2[P1] <= 19.5)))
		N_all2_10_195 = float(len(np.where(appMagMean2[P10] <= 19.5)))
		N_all2_30_195 = float(len(np.where(appMagMean2[P30] <= 19.5)))
		N_all2_100_195 = float(len(np.where(appMagMean2[P100] <= 19.5)))
		N_all2_1000_195 = float(len(np.where(appMagMean2[P1000] <= 19.5)))



		#NORMALIZED FROM HERE vv


		N_all2_norm = (N_all2/N_all2)*N_mult2 #normalized
		N_all2_03norm = (N_all2_03/N_all2)*N_mult2
		N_all2_1norm = (N_all2_1/N_all2)*N_mult2
		N_all2_10norm = (N_all2_10/N_all2)*N_mult2
		N_all2_30norm = (N_all2_30/N_all2)*N_mult2
		N_all2_100norm = (N_all2_100/N_all2)*N_mult2
		N_all2_1000norm = (N_all2_1000/N_all2)*N_mult2

		N_observable2 = (float(len(observable2))/float(N_all2))*N_mult2
		N_observable2_03 = (float(len(observable2_03))/float(N_all2))*N_mult2
		N_observable2_1 = (float(len(observable2_1))/float(N_all2))*N_mult2
		N_observable2_10 = (float(len(observable2_10))/float(N_all2))*N_mult2
		N_observable2_30 = (float(len(observable2_30))/float(N_all2))*N_mult2
		N_observable2_100 = (float(len(observable2_100))/float(N_all2))*N_mult2
		N_observable2_1000 = (float(len(observable2_1000))/float(N_all2))*N_mult2

		N_recoverable2 = (float(len(recoverable2))/float(N_all2))*N_mult2
		N_recoverable2_03 = (float(len(recoverable2_03))/float(N_all2))*N_mult2
		N_recoverable2_1 = (float(len(recoverable2_1))/float(N_all2))*N_mult2
		N_recoverable2_10 = (float(len(recoverable2_10))/float(N_all2))*N_mult2
		N_recoverable2_30 = (float(len(recoverable2_30))/float(N_all2))*N_mult2
		N_recoverable2_100 = (float(len(recoverable2_100))/float(N_all2))*N_mult2
		N_recoverable2_1000 = (float(len(recoverable2_1000))/float(N_all2))*N_mult2


		N_all2_norm_22 = (N_all2_22/N_all2)*N_mult2 #normalized
		N_all2_03norm_22 = (N_all2_03_22/N_all2)*N_mult2
		N_all2_1norm_22 = (N_all2_1_22/N_all2)*N_mult2
		N_all2_10norm_22 = (N_all2_10_22/N_all2)*N_mult2
		N_all2_30norm_22 = (N_all2_30_22/N_all2)*N_mult2
		N_all2_100norm_22 = (N_all2_100_22/N_all2)*N_mult2
		N_all2_1000norm_22 = (N_all2_1000_22/N_all2)*N_mult2

		N_observable2_22 = (float(len(observable2_22))/float(N_all2))*N_mult2
		N_observable2_03_22 = (float(len(observable2_03_22))/float(N_all2))*N_mult2
		N_observable2_1_22 = (float(len(observable2_1_22))/float(N_all2))*N_mult2
		N_observable2_10_22 = (float(len(observable2_10_22))/float(N_all2))*N_mult2
		N_observable2_30_22 = (float(len(observable2_30_22))/float(N_all2))*N_mult2
		N_observable2_100_22 = (float(len(observable2_100_22))/float(N_all2))*N_mult2
		N_observable2_1000_22 = (float(len(observable2_1000_22))/float(N_all2))*N_mult2

		N_recoverable2_22 = (float(len(recoverable2_22))/float(N_all2))*N_mult2
		N_recoverable2_03_22 = (float(len(recoverable2_03_22))/float(N_all2))*N_mult2
		N_recoverable2_1_22 = (float(len(recoverable2_1_22))/float(N_all2))*N_mult2
		N_recoverable2_10_22 = (float(len(recoverable2_10_22))/float(N_all2))*N_mult2
		N_recoverable2_30_22 = (float(len(recoverable2_30_22))/float(N_all2))*N_mult2
		N_recoverable2_100_22 = (float(len(recoverable2_100_22))/float(N_all2))*N_mult2
		N_recoverable2_1000_22 = (float(len(recoverable2_1000_22))/float(N_all2))*N_mult2


		N_all2_norm_195 = (N_all2_195/N_all2)*N_mult2 #normalized
		N_all2_03norm_195 = (N_all2_03_195/N_all2)*N_mult2
		N_all2_1norm_195 = (N_all2_1_195/N_all2)*N_mult2
		N_all2_10norm_195 = (N_all2_10_195/N_all2)*N_mult2
		N_all2_30norm_195 = (N_all2_30_195/N_all2)*N_mult2
		N_all2_100norm_195 = (N_all2_100_195/N_all2)*N_mult2
		N_all2_1000norm_195 = (N_all2_1000_195/N_all2)*N_mult2

		N_observable2_195 = (float(len(observable2_195))/float(N_all2))*N_mult2
		N_observable2_03_195 = (float(len(observable2_03_195))/float(N_all2))*N_mult2
		N_observable2_1_195 = (float(len(observable2_1_195))/float(N_all2))*N_mult2
		N_observable2_10_195 = (float(len(observable2_10_195))/float(N_all2))*N_mult2
		N_observable2_30_195 = (float(len(observable2_30_195))/float(N_all2))*N_mult2
		N_observable2_100_195 = (float(len(observable2_100_195))/float(N_all2))*N_mult2
		N_observable2_1000_195 = (float(len(observable2_1000_195))/float(N_all2))*N_mult2

		N_recoverable2_195 = (float(len(recoverable2_195))/float(N_all2))*N_mult2
		N_recoverable2_03_195 = (float(len(recoverable2_03_195))/float(N_all2))*N_mult2
		N_recoverable2_1_195 = (float(len(recoverable2_1_195))/float(N_all2))*N_mult2
		N_recoverable2_10_195 = (float(len(recoverable2_10_195))/float(N_all2))*N_mult2
		N_recoverable2_30_195 = (float(len(recoverable2_30_195))/float(N_all2))*N_mult2
		N_recoverable2_100_195 = (float(len(recoverable2_100_195))/float(N_all2))*N_mult2
		N_recoverable2_1000_195 = (float(len(recoverable2_1000_195))/float(N_all2))*N_mult2

		N_totalfast_array.append(float(N_all2_norm))
		N_totalobservablefast_array.append(float(N_observable2))
		N_totalrecoverablefast_array.append(float(N_recoverable2))

		N_totalfast_array_03.append(float(N_all2_03norm))
		N_totalobservablefast_array_03.append(float(N_observable2_03))
		N_totalrecoverablefast_array_03.append(float(N_recoverable2_03))

		N_totalfast_array_1.append(float(N_all2_1norm))
		N_totalobservablefast_array_1.append(float(N_observable2_1))
		N_totalrecoverablefast_array_1.append(float(N_recoverable2_1))

		N_totalfast_array_10.append(float(N_all2_10norm))
		N_totalobservablefast_array_10.append(float(N_observable2_10))
		N_totalrecoverablefast_array_10.append(float(N_recoverable2_10))

		N_totalfast_array_30.append(float(N_all2_30norm))
		N_totalobservablefast_array_30.append(float(N_observable2_30))
		N_totalrecoverablefast_array_30.append(float(N_recoverable2_30))

		N_totalfast_array_100.append(float(N_all2_100norm))
		N_totalobservablefast_array_100.append(float(N_observable2_100))
		N_totalrecoverablefast_array_100.append(float(N_recoverable2_100))

		N_totalfast_array_1000.append(float(N_all2_1000norm))
		N_totalobservablefast_array_1000.append(float(N_observable2_1000))
		N_totalrecoverablefast_array_1000.append(float(N_recoverable2_1000))

		N_totalfast22_array.append(float(N_all2_norm_22))
		N_totalobservablefast22_array.append(float(N_observable2_22))
		N_totalrecoverablefast22_array.append(float(N_recoverable2_22))

		N_totalfast22_array_03.append(float(N_all2_03norm_22))
		N_totalobservablefast22_array_03.append(float(N_observable2_03_22))
		N_totalrecoverablefast22_array_03.append(float(N_recoverable2_03_22))

		N_totalfast22_array_1.append(float(N_all2_1norm_22))
		N_totalobservablefast22_array_1.append(float(N_observable2_1_22))
		N_totalrecoverablefast22_array_1.append(float(N_recoverable2_1_22))

		N_totalfast22_array_10.append(float(N_all2_10norm_22))
		N_totalobservablefast22_array_10.append(float(N_observable2_10_22))
		N_totalrecoverablefast22_array_10.append(float(N_recoverable2_10_22))

		N_totalfast22_array_30.append(float(N_all2_30norm_22))
		N_totalobservablefast22_array_30.append(float(N_observable2_30_22))
		N_totalrecoverablefast22_array_30.append(float(N_recoverable2_30_22))

		N_totalfast22_array_100.append(float(N_all2_100norm_22))
		N_totalobservablefast22_array_100.append(float(N_observable2_100_22))
		N_totalrecoverablefast22_array_100.append(float(N_recoverable2_100_22))

		N_totalfast22_array_1000.append(float(N_all2_1000norm_22))
		N_totalobservablefast22_array_1000.append(float(N_observable2_1000_22))
		N_totalrecoverablefast22_array_1000.append(float(N_recoverable2_1000_22))

		N_totalfast195_array.append(float(N_all2_norm_195))
		N_totalobservablefast195_array.append(float(N_observable2_195))
		N_totalrecoverablefast195_array.append(float(N_recoverable2_195))

		N_totalfast195_array_03.append(float(N_all2_03norm_195))
		N_totalobservablefast195_array_03.append(float(N_observable2_03_195))
		N_totalrecoverablefast195_array_03.append(float(N_recoverable2_03_195))

		N_totalfast195_array_1.append(float(N_all2_1norm_195))
		N_totalobservablefast195_array_1.append(float(N_observable2_1_195))
		N_totalrecoverablefast195_array_1.append(float(N_recoverable2_1_195))

		N_totalfast195_array_10.append(float(N_all2_10norm_195))
		N_totalobservablefast195_array_10.append(float(N_observable2_10_195))
		N_totalrecoverablefast195_array_10.append(float(N_recoverable2_10_195))

		N_totalfast195_array_30.append(float(N_all2_30norm_195))
		N_totalobservablefast195_array_30.append(float(N_observable2_30_195))
		N_totalrecoverablefast195_array_30.append(float(N_recoverable2_30_195))

		N_totalfast195_array_100.append(float(N_all2_100norm_195))
		N_totalobservablefast195_array_100.append(float(N_observable2_100_195))
		N_totalrecoverablefast195_array_100.append(float(N_recoverable2_100_195))

		N_totalfast195_array_1000.append(float(N_all2_1000norm_195))
		N_totalobservablefast195_array_1000.append(float(N_observable2_1000_195))
		N_totalrecoverablefast195_array_1000.append(float(N_recoverable2_1000_195))

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

print("N_totalfast = ", N_totalfast, "N_totalobservablefast = ", N_totalobservablefast, "N_totalrecoverablefast = ", N_totalrecoverablefast)

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
print("wholerecoverypercent_fast = ", wholerecoverypercent_fast, "wholerecoverypercent_fast_03 = ", wholerecoverypercent_fast_03, "wholerecoverypercent_fast_1 = ", wholerecoverypercent_fast_1, "wholerecoverypercent_fast_10 = ", wholerecoverypercent_fast_10, "wholerecoverypercent_fast_30 = ", wholerecoverypercent_fast_30, "wholerecoverypercent_fast_100 = ", wholerecoverypercent_fast_100, "wholerecoverypercent_fast_1000 = ", wholerecoverypercent_fast_1000)
print("sigmafast = ", sigmafast, "sigmafast_03 = ", sigmafast_03, "sigmafast_1 = ", sigmafast_1, "sigmafast_10 = ", sigmafast_10, "sigmafast_30 = ", sigmafast_30, "sigmafast_100 = ", sigmafast_100, "sigmafast_1000 = ", sigmafast_1000)
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
print("overallrecoverypercent_fast = ", overallrecoverypercent_fast, "overallrecoverypercent_fast_03 = ", overallrecoverypercent_fast_03, "overallrecoverypercent_fast_1 = ", overallrecoverypercent_fast_1, "overallrecoverypercent_fast_10 = ", overallrecoverypercent_fast_10, "overallrecoverypercent_fast_30 = ", overallrecoverypercent_fast_30, "overallrecoverypercent_fast_100 = ", overallrecoverypercent_fast_100, "overallrecoverypercent_fast_1000 = ", overallrecoverypercent_fast_1000)
print("overallsigmafast = ", overallsigmafast, "overallsigmafast_03 = ", overallsigmafast_03, "overallsigmafast_1 = ", overallsigmafast_1, "overallsigmafast_10 = ", overallsigmafast_10, "overallsigmafast_30 = ", overallsigmafast_30, "overallsigmafast_100 = ", overallsigmafast_100, "overallsigmafast_1000 = ", overallsigmafast_1000)



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
print("wholerecoverypercent_fast22 = ", wholerecoverypercent_fast22, "wholerecoverypercent_fast22_03 = ", wholerecoverypercent_fast22_03, "wholerecoverypercent_fast22_1 = ", wholerecoverypercent_fast22_1, "wholerecoverypercent_fast22_10 = ", wholerecoverypercent_fast22_10, "wholerecoverypercent_fast22_30 = ", wholerecoverypercent_fast22_30, "wholerecoverypercent_fast22_100 = ", wholerecoverypercent_fast22_100, "wholerecoverypercent_fast22_1000 = ", wholerecoverypercent_fast22_1000)
print("sigmafast22 = ", sigmafast22, "sigmafast22_03 = ", sigmafast22_03, "sigmafast22_1 = ", sigmafast22_1, "sigmafast22_10 = ", sigmafast22_10, "sigmafast22_30 = ", sigmafast22_30, "sigmafast22_100 = ", sigmafast22_100, "sigmafast22_1000 = ", sigmafast22_1000)
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
print("overallrecoverypercent_fast22 = ", overallrecoverypercent_fast22, "overallrecoverypercent_fast22_03 = ", overallrecoverypercent_fast22_03, "overallrecoverypercent_fast22_1 = ", overallrecoverypercent_fast22_1, "overallrecoverypercent_fast22_10 = ", overallrecoverypercent_fast22_10, "overallrecoverypercent_fast22_30 = ", overallrecoverypercent_fast22_30, "overallrecoverypercent_fast22_100 = ", overallrecoverypercent_fast22_100, "overallrecoverypercent_fast22_1000 = ", overallrecoverypercent_fast22_1000)
print("overallsigmafast22 = ", overallsigmafast22, "overallsigmafast22_03 = ", overallsigmafast22_03, "overallsigmafast22_1 = ", overallsigmafast22_1, "overallsigmafast22_10 = ", overallsigmafast22_10, "overallsigmafast22_30 = ", overallsigmafast22_30, "overallsigmafast22_100 = ", overallsigmafast22_100, "overallsigmafast22_1000 = ", overallsigmafast22_1000)



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
print("wholerecoverypercent_fast195 = ", wholerecoverypercent_fast195, "wholerecoverypercent_fast195_03 = ", wholerecoverypercent_fast195_03, "wholerecoverypercent_fast195_1 = ", wholerecoverypercent_fast195_1, "wholerecoverypercent_fast195_10 = ", wholerecoverypercent_fast195_10, "wholerecoverypercent_fast195_30 = ", wholerecoverypercent_fast195_30, "wholerecoverypercent_fast195_100 = ", wholerecoverypercent_fast195_100, "wholerecoverypercent_fast195_1000 = ", wholerecoverypercent_fast195_1000)
print("sigmafast195 = ", sigmafast195, "sigmafast195_03 = ", sigmafast195_03, "sigmafast195_1 = ", sigmafast195_1, "sigmafast195_10 = ", sigmafast195_10, "sigmafast195_30 = ", sigmafast195_30, "sigmafast195_100 = ", sigmafast195_100, "sigmafast195_1000 = ", sigmafast195_1000)
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
overallsigmafast195_1000 = ((N_totalrecoverablefast195_1000**(1/2))/N_totalfast195_1000)*100
print("overallrecoverypercent_fast195 = ", overallrecoverypercent_fast195, "overallrecoverypercent_fast195_03 = ", overallrecoverypercent_fast195_03, "overallrecoverypercent_fast195_1 = ", overallrecoverypercent_fast195_1, "overallrecoverypercent_fast195_10 = ", overallrecoverypercent_fast195_10, "overallrecoverypercent_fast195_30 = ", overallrecoverypercent_fast195_30, "overallrecoverypercent_fast195_100 = ", overallrecoverypercent_fast195_100, "overallrecoverypercent_fast195_1000 = ", overallrecoverypercent_fast195_1000)
print("overallsigmafast195 = ", overallsigmafast195, "overallsigmafast195_03 = ", overallsigmafast195_03, "overallsigmafast195_1 = ", overallsigmafast195_1, "overallsigmafast195_10 = ", overallsigmafast195_10, "overallsigmafast195_30 = ", overallsigmafast195_30, "overallsigmafast195_100 = ", overallsigmafast195_100, "overallsigmafast195_1000 = ", overallsigmafast195_1000)



for fileobsDist_ in sorted(allFiles_obsDist):
	
	filename3 = fileobsDist_[77:] #when file path no longer has /old in it, will be fileobsDist_[73:]
	fileid3 = filename3.strip('output_file.csv')
	colorvalue3 = int(fileid3)
	colorvalue_obsDist.append(colorvalue3)
	print ("I'm starting " + fileid3)


	datobsDist = pd.read_csv(fileobsDist_, sep = ',', header=2)


	##########################################################

	datobsDist1 = pd.read_csv(fileobsDist_, sep = ',', header=0, nrows=1)
	N_tri3 = datibsDist1["NstarsTRILEGAL"][0]
	print("N_tri3 = ", N_tri3)

	m1hAll03, m1b3 = np.histogram(datobDist["m1"], bins=mbins)
	dm13 = np.diff(m1b3)
	m1val3 = m1b3[:-1] + dm13/2.

	fb3 = np.sum(m1hAll03*dm13*fbFit(m1val3))

	N_mult3 = N_tri3*fb3
	
	##########################################################
	PeriodIn3 = datobsDist['p'] # input period -- 'p' in data file	
	if len(PeriodIn3) ==0.:
		continue
	if N_tri3 == 0:
		continue
	else:

		PeriodOut3 = datobsDist['LSM_PERIOD'] #LSM_PERIOD in data file
		appMagMean3 = datobsDist['appMagMean'] #when file path is back to fast/obsDist/output_files vs. fast/old/obsDist/output_files, this should be shanged to appMagMean_r
		observable3 = np.where(PeriodOut3 != -999)[0]
		observable3_03 = np.where(PeriodIn3[observable3] <= 0.3)[0]
		observable3_1 = np.where(PeriodIn3[observable3] <= 1)[0]
		observable3_10 = np.where(PeriodIn3[observable3] <= 10)[0]
		observable3_30 = np.where(PeriodIn3[observable3] <= 30)[0]
		observable3_100 = np.where(PeriodIn3[observable3] <= 100)[0]
		observable3_1000 = np.where(PeriodIn3[observable3] <= 1000)[0]

		observable3_22 = np.where(appMagMean3[observable3] <= 22.)[0]
		observable3_03_22 = np.where(appMagMean3[observable3_03] <= 22.)[0]
		observable3_1_22 = np.where(appMagMean3[observable3_1] <= 22.)[0]
		observable3_10_22 = np.where(appMagMean3[observable3_10] <= 22.)[0]
		observable3_30_22 = np.where(appMagMean3[observable3_30] <= 22.)[0]
		observable3_100_22 = np.where(appMagMean3[observable3_100] <= 22.)[0]
		observable3_1000_22 = np.where(appMagMean3[observable3_1000] <= 22.)[0]

		observable3_195 = np.where(appMagMean3[observable3] <= 19.5)[0]
		observable3_03_195 = np.where(appMagMean3[observable3_03] <= 19.5)[0]
		observable3_1_195 = np.where(appMagMean3[observable3_1] <= 19.5)[0]
		observable3_10_195 = np.where(appMagMean3[observable3_10] <= 19.5)[0]
		observable3_30_195 = np.where(appMagMean3[observable3_30] <= 19.5)[0]
		observable3_100_195 = np.where(appMagMean3[observable3_100] <= 19.5)[0]
		observable3_1000_195 = np.where(appMagMean3[observable3_1000] <= 19.5)[0]

		Sigma_Period_Whole3 = abs(PeriodOut3 - PeriodIn3)/PeriodIn3
		Sigma_Period_Half3 = abs(PeriodOut3 - 0.5*PeriodIn3)/(0.5*PeriodIn3)
		Sigma_Period_Twice3 = abs(PeriodOut3 - 2*PeriodIn3)/(2*PeriodIn3)

		recover_twice3 = np.where(Sigma_Period_Twice3 <= 0.1)[0]
		recover_twice3_03 = np.where(PeriodIn3[recover_twice3] <= 0.3)[0]
		recover_twice3_1 = np.where(PeriodIn3[recover_twice3] <= 1)[0]
		recover_twice3_10 = np.where(PeriodIn3[recover_twice3] <= 10)[0]
		recover_twice3_30 = np.where(PeriodIn3[recover_twice3] <= 30)[0]
		recover_twice3_100 = np.where(PeriodIn3[recover_twice3] <= 100)[0]
		recover_twice3_1000 = np.where(PeriodIn3[recover_twice3] <= 1000)[0]
		#recover_half1 = np.where(np.logical_and(np.isfinite(Sigma_Period_Half1), Sigma_Period_Half1 <= 0.1))[0]
		recover_half3 = np.where(Sigma_Period_Half3 <= 0.1)[0]
		recover_half3_03 = np.where(PeriodIn3[recover_half3] <= 0.3)[0]
		recover_half3_1 = np.where(PeriodIn3[recover_half3] <= 1)[0]
		recover_half3_10 = np.where(PeriodIn3[recover_half3] <= 10)[0]
		recover_half3_30 = np.where(PeriodIn3[recover_half3] <= 30)[0]
		recover_half3_100 = np.where(PeriodIn3[recover_half3] <= 100)[0]
		recover_half3_1000 = np.where(PeriodIn3[recover_half3] <= 1000)[0]
		#recover_whole1 = np.where(np.logical_and(np.isfinite(Sigma_Period_Whole1), Sigma_Period_Whole1 <= 0.1))[0]
		recover_whole3 = np.where(Sigma_Period_Whole3 <= 0.1)[0]
		recover_whole3_03 = np.where(PeriodIn3[recover_whole3] <= 0.3)[0]
		recover_whole3_1 = np.where(PeriodIn3[recover_whole3] <= 1)[0]
		recover_whole3_10 = np.where(PeriodIn3[recover_whole3] <= 10)[0]
		recover_whole3_30 = np.where(PeriodIn3[recover_whole3] <= 30)[0]
		recover_whole3_100 = np.where(PeriodIn3[recover_whole3] <= 100)[0]
		recover_whole3_1000 = np.where(PeriodIn3[recover_whole3] <= 1000)[0]

		recoverable3 = np.concatenate((recover_twice3, recover_whole3, recover_half3), axis=0)
		recoverable3_03 = np.concatenate((recover_twice3_03, recover_whole3_03, recover_half3_03), axis=0)
		recoverable3_1 = np.concatenate((recover_twice3_1, recover_whole3_1, recover_half3_1), axis=0)
		recoverable3_10 = np.concatenate((recover_twice3_10, recover_whole3_10, recover_half3_10), axis=0)
		recoverable3_30 = np.concatenate((recover_twice3_30, recover_whole3_30, recover_half3_30), axis=0)
		recoverable3_100 = np.concatenate((recover_twice3_100, recover_whole3_100, recover_half3_100), axis=0)
		recoverable3_1000 = np.concatenate((recover_twice3_1000, recover_whole3_1000, recover_half3_1000), axis=0)

		recoverable3_22 = np.where(appMagMean3[recoverable3] <= 22.)[0]
		recoverable3_03_22 = np.where(appMagMean3[recoverable3_03] <= 22.)[0]
		recoverable3_1_22 = np.where(appMagMean3[recoverable3_1] <= 22.)[0]
		recoverable3_10_22 = np.where(appMagMean3[recoverable3_10] <= 22.)[0]
		recoverable3_30_22 = np.where(appMagMean3[recoverable3_30] <= 22.)[0]
		recoverable3_100_22 = np.where(appMagMean3[recoverable3_100] <= 22.)[0]
		recoverable3_1000_22 = np.where(appMagMean3[recoverable3_1000] <= 22.)[0]
		
		recoverable3_195 = np.where(appMagMean3[recoverable3] <= 19.5)[0]
		recoverable3_03_195 = np.where(appMagMean3[recoverable3_03] <= 19.5)[0]
		recoverable3_1_195 = np.where(appMagMean3[recoverable3_1] <= 19.5)[0]
		recoverable3_10_195 = np.where(appMagMean3[recoverable3_10] <= 19.5)[0]
		recoverable3_30_195 = np.where(appMagMean3[recoverable3_30] <= 19.5)[0]
		recoverable3_100_195 = np.where(appMagMean3[recoverable3_100] <= 19.5)[0]
		recoverable3_1000_195 = np.where(appMagMean3[recoverable3_1000] <= 19.5)[0]

		P03 = np.where(PeriodIn3 <= 0.3)[0]
		P1 = np.where(PeriodIn3 <= 1)[0]
		P10 = np.where(PeriodIn3 <= 10)[0]
		P30 = np.where(PeriodIn3 <= 30)[0]
		P100 = np.where(PeriodIn3 <= 100)[0]
		P1000 = np.where(PeriodIn3 <= 1000)[0]

		N_all3 = float(len(PeriodIn3)) #unnormalized
		N_all3_03 = float(len(P03))
		N_all3_1 = float(len(P1))
		N_all3_10 = float(len(P10))
		N_all3_30 = float(len(P30))
		N_all3_100 = float(len(P100))
		N_all3_1000 = float(len(P1000))


		N_all3_22 = float(len(np.where(appMagMean3 <= 22))) #unnormalized
		N_all3_03_22 = float(len(np.where(appMagMean3[P03] <= 22)))
		N_all3_1_22 = float(len(np.where(appMagMean3[P1] <= 22)))
		N_all3_10_22 = float(len(np.where(appMagMean3[P10] <= 22)))
		N_all3_30_22 = float(len(np.where(appMagMean3[P30] <= 22)))
		N_all3_100_22 = float(len(np.where(appMagMean3[P100] <= 22)))
		N_all3_1000_22 = float(len(np.where(appMagMean3[P1000] <= 22)))

		N_all3_195 = float(len(np.where(appMagMean3 <= 19.5))) #unnormalized
		N_all3_03_195 = float(len(np.where(appMagMean3[P03] <= 19.5)))
		N_all3_1_195 = float(len(np.where(appMagMean3[P1] <= 19.5)))
		N_all3_10_195 = float(len(np.where(appMagMean3[P10] <= 19.5)))
		N_all3_30_195 = float(len(np.where(appMagMean3[P30] <= 19.5)))
		N_all3_100_195 = float(len(np.where(appMagMean3[P100] <= 19.5)))
		N_all3_1000_195 = float(len(np.where(appMagMean3[P1000] <= 19.5)))


		#NORMALIZED FROM HERE vv


		N_all3_norm = (N_all3/N_all3)*N_mult3 #normalized
		N_all3_03norm = (N_all3_03/N_all3)*N_mult3
		N_all3_1norm = (N_all3_1/N_all3)*N_mult3
		N_all3_10norm = (N_all3_10/N_all3)*N_mult3
		N_all3_30norm = (N_all3_30/N_all3)*N_mult3
		N_all3_100norm = (N_all3_100/N_all3)*N_mult3
		N_all3_1000norm = (N_all3_1000/N_all3)*N_mult3

		N_observable3 = (float(len(observable3))/float(N_all3))*N_mult3
		N_observable3_03 = (float(len(observable3_03))/float(N_all3))*N_mult3
		N_observable3_1 = (float(len(observable3_1))/float(N_all3))*N_mult3
		N_observable3_10 = (float(len(observable3_10))/float(N_all3))*N_mult3
		N_observable3_30 = (float(len(observable3_30))/float(N_all3))*N_mult3
		N_observable3_100 = (float(len(observable3_100))/float(N_all3))*N_mult3
		N_observable3_1000 = (float(len(observable3_1000))/float(N_all3))*N_mult3

		N_recoverable3 = (float(len(recoverable3))/float(N_all3))*N_mult3
		N_recoverable3_03 = (float(len(recoverable3_03))/float(N_all3))*N_mult3
		N_recoverable3_1 = (float(len(recoverable3_1))/float(N_all3))*N_mult3
		N_recoverable3_10 = (float(len(recoverable3_10))/float(N_all3))*N_mult3
		N_recoverable3_30 = (float(len(recoverable3_30))/float(N_all3))*N_mult3
		N_recoverable3_100 = (float(len(recoverable3_100))/float(N_all3))*N_mult3
		N_recoverable3_1000 = (float(len(recoverable3_1000))/float(N_all3))*N_mult3


		N_all3_norm_22 = (N_all3_22/N_all3)*N_mult3 #normalized
		N_all3_03norm_22 = (N_all3_03_22/N_all3)*N_mult3
		N_all3_1norm_22 = (N_all3_1_22/N_all3)*N_mult3
		N_all3_10norm_22 = (N_all3_10_22/N_all3)*N_mult3
		N_all3_30norm_22 = (N_all3_30_22/N_all3)*N_mult3
		N_all3_100norm_22 = (N_all3_100_22/N_all3)*N_mult3
		N_all3_1000norm_22 = (N_all3_1000_22/N_all3)*N_mult3

		N_observable3_22 = (float(len(observable3_22))/float(N_all3))*N_mult3
		N_observable3_03_22 = (float(len(observable3_03_22))/float(N_all3))*N_mult3
		N_observable3_1_22 = (float(len(observable3_1_22))/float(N_all3))*N_mult3
		N_observable3_10_22 = (float(len(observable3_10_22))/float(N_all3))*N_mult3
		N_observable3_30_22 = (float(len(observable3_30_22))/float(N_all3))*N_mult3
		N_observable3_100_22 = (float(len(observable3_100_22))/float(N_all3))*N_mult3
		N_observable3_1000_22 = (float(len(observable3_1000_22))/float(N_all3))*N_mult3

		N_recoverable3_22 = (float(len(recoverable3_22))/float(N_all3))*N_mult3
		N_recoverable3_03_22 = (float(len(recoverable3_03_22))/float(N_all3))*N_mult3
		N_recoverable3_1_22 = (float(len(recoverable3_1_22))/float(N_all3))*N_mult3
		N_recoverable3_10_22 = (float(len(recoverable3_10_22))/float(N_all3))*N_mult3
		N_recoverable3_30_22 = (float(len(recoverable3_30_22))/float(N_all3))*N_mult3
		N_recoverable3_100_22 = (float(len(recoverable3_100_22))/float(N_all3))*N_mult3
		N_recoverable3_1000_22 = (float(len(recoverable3_1000_22))/float(N_all3))*N_mult3


		N_all3_norm_195 = (N_all3_195/N_all3)*N_mult3 #normalized
		N_all3_03norm_195 = (N_all3_03_195/N_all3)*N_mult3
		N_all3_1norm_195 = (N_all3_1_195/N_all3)*N_mult3
		N_all3_10norm_195 = (N_all3_10_195/N_all3)*N_mult3
		N_all3_30norm_195 = (N_all3_30_195/N_all3)*N_mult3
		N_all3_100norm_195 = (N_all3_100_195/N_all3)*N_mult3
		N_all3_1000norm_195 = (N_all3_1000_195/N_all3)*N_mult3

		N_observable3_195 = (float(len(observable3_195))/float(N_all3))*N_mult3
		N_observable3_03_195 = (float(len(observable3_03_195))/float(N_all3))*N_mult3
		N_observable3_1_195 = (float(len(observable3_1_195))/float(N_all3))*N_mult3
		N_observable3_10_195 = (float(len(observable3_10_195))/float(N_all3))*N_mult3
		N_observable3_30_195 = (float(len(observable3_30_195))/float(N_all3))*N_mult3
		N_observable3_100_195 = (float(len(observable3_100_195))/float(N_all3))*N_mult3
		N_observable3_1000_195 = (float(len(observable3_1000_195))/float(N_all3))*N_mult3

		N_recoverable3_195 = (float(len(recoverable3_195))/float(N_all3))*N_mult3
		N_recoverable3_03_195 = (float(len(recoverable3_03_195))/float(N_all3))*N_mult3
		N_recoverable3_1_195 = (float(len(recoverable3_1_195))/float(N_all3))*N_mult3
		N_recoverable3_10_195 = (float(len(recoverable3_10_195))/float(N_all3))*N_mult3
		N_recoverable3_30_195 = (float(len(recoverable3_30_195))/float(N_all3))*N_mult3
		N_recoverable3_100_195 = (float(len(recoverable3_100_195))/float(N_all3))*N_mult3
		N_recoverable3_1000_195 = (float(len(recoverable3_1000_195))/float(N_all3))*N_mult3

		N_totalobsDist_array.append(float(N_all3_norm))
		N_totalobservableobsDist_array.append(float(N_observable3))
		N_totalrecoverableobsDist_array.append(float(N_recoverable3))

		N_totalobsDist_array_03.append(float(N_all3_03norm))
		N_totalobservableobsDist_array_03.append(float(N_observable3_03))
		N_totalrecoverableobsDist_array_03.append(float(N_recoverable3_03))

		N_totalobsDist_array_1.append(float(N_all3_1norm))
		N_totalobservableobsDist_array_1.append(float(N_observable3_1))
		N_totalrecoverableobsDist_array_1.append(float(N_recoverable3_1))

		N_totalobsDist_array_10.append(float(N_all3_10norm))
		N_totalobservableobsDist_array_10.append(float(N_observable3_10))
		N_totalrecoverableobsDist_array_10.append(float(N_recoverable3_10))

		N_totalobsDist_array_30.append(float(N_all3_30norm))
		N_totalobservableobsDist_array_30.append(float(N_observable3_30))
		N_totalrecoverableobsDist_array_30.append(float(N_recoverable3_30))

		N_totalobsDist_array_100.append(float(N_all3_100norm))
		N_totalobservableobsDist_array_100.append(float(N_observable3_100))
		N_totalrecoverableobsDist_array_100.append(float(N_recoverable3_100))

		N_totalobsDist_array_1000.append(float(N_all3_1000norm))
		N_totalobservableobsDist_array_1000.append(float(N_observable3_1000))
		N_totalrecoverableobsDist_array_1000.append(float(N_recoverable3_1000))

		N_totalobsDist22_array.append(float(N_all3_norm_22))
		N_totalobservableobsDist22_array.append(float(N_observable3_22))
		N_totalrecoverableobsDist22_array.append(float(N_recoverable3_22))

		N_totalobsDist22_array_03.append(float(N_all3_03norm_22))
		N_totalobservableobsDist22_array_03.append(float(N_observable3_03_22))
		N_totalrecoverableobsDist22_array_03.append(float(N_recoverable3_03_22))

		N_totalobsDist22_array_1.append(float(N_all3_1norm_22))
		N_totalobservableobsDist22_array_1.append(float(N_observable3_1_22))
		N_totalrecoverableobsDist22_array_1.append(float(N_recoverable3_1_22))

		N_totalobsDist22_array_10.append(float(N_all3_10norm_22))
		N_totalobservableobsDist22_array_10.append(float(N_observable3_10_22))
		N_totalrecoverableobsDist22_array_10.append(float(N_recoverable3_10_22))

		N_totalobsDist22_array_30.append(float(N_all3_30norm_22))
		N_totalobservableobsDist22_array_30.append(float(N_observable3_30_22))
		N_totalrecoverableobsDist22_array_30.append(float(N_recoverable3_30_22))

		N_totalobsDist22_array_100.append(float(N_all3_100norm_22))
		N_totalobservableobsDist22_array_100.append(float(N_observable3_100_22))
		N_totalrecoverableobsDist22_array_100.append(float(N_recoverable3_100_22))

		N_totalobsDist22_array_1000.append(float(N_all3_1000norm_22))
		N_totalobservableobsDist22_array_1000.append(float(N_observable3_1000_22))
		N_totalrecoverableobsDist22_array_1000.append(float(N_recoverable3_1000_22))

		N_totalobsDist195_array.append(float(N_all3_norm_195))
		N_totalobservableobsDist195_array.append(float(N_observable3_195))
		N_totalrecoverableobsDist195_array.append(float(N_recoverable3_195))

		N_totalobsDist195_array_03.append(float(N_all3_03norm_195))
		N_totalobservableobsDist195_array_03.append(float(N_observable3_03_195))
		N_totalrecoverableobsDist195_array_03.append(float(N_recoverable3_03_195))

		N_totalobsDist195_array_1.append(float(N_all3_1norm_195))
		N_totalobservableobsDist195_array_1.append(float(N_observable3_1_195))
		N_totalrecoverableobsDist195_array_1.append(float(N_recoverable3_1_195))

		N_totalobsDist195_array_10.append(float(N_all3_10norm_195))
		N_totalobservableobsDist195_array_10.append(float(N_observable3_10_195))
		N_totalrecoverableobsDist195_array_10.append(float(N_recoverable3_10_195))

		N_totalobsDist195_array_30.append(float(N_all3_30norm_195))
		N_totalobservableobsDist195_array_30.append(float(N_observable3_30_195))
		N_totalrecoverableobsDist195_array_30.append(float(N_recoverable3_30_195))

		N_totalobsDist195_array_100.append(float(N_all3_100norm_195))
		N_totalobservableobsDist195_array_100.append(float(N_observable3_100_195))
		N_totalrecoverableobsDist195_array_100.append(float(N_recoverable3_100_195))

		N_totalobsDist195_array_1000.append(float(N_all3_1000norm_195))
		N_totalobservableobsDist195_array_1000.append(float(N_observable3_1000_195))
		N_totalrecoverableobsDist195_array_1000.append(float(N_recoverable3_1000_195))

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

print("N_totalobsDist = ", N_totalobsDist, "N_totalobservableobsDist = ", N_totalobservableobsDist, "N_totalrecoverableobsDist = ", N_totalrecoverableobsDist)

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
print("wholerecoverypercent_obsDist = ", wholerecoverypercent_obsDist, "wholerecoverypercent_obsDist_03 = ", wholerecoverypercent_obsDist_03, "wholerecoverypercent_obsDist_1 = ", wholerecoverypercent_obsDist_1, "wholerecoverypercent_obsDist_10 = ", wholerecoverypercent_obsDist_10, "wholerecoverypercent_obsDist_30 = ", wholerecoverypercent_obsDist_30, "wholerecoverypercent_obsDist_100 = ", wholerecoverypercent_obsDist_100, "wholerecoverypercent_obsDist_1000 = ", wholerecoverypercent_obsDist_1000)
print("sigmaobsDist = ", sigmaobsDist, "sigmaobsDist_03 = ", sigmaobsDist_03, "sigmaobsDist_1 = ", sigmaobsDist_1, "sigmaobsDist_10 = ", sigmaobsDist_10, "sigmaobsDist_30 = ", sigmaobsDist_30, "sigmaobsDist_100 = ", sigmaobsDist_100, "sigmaobsDist_1000 = ", sigmaobsDist_1000)
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
print("overallrecoverypercent_obsDist = ", overallrecoverypercent_obsDist, "overallrecoverypercent_obsDist_03 = ", overallrecoverypercent_obsDist_03, "overallrecoverypercent_obsDist_1 = ", overallrecoverypercent_obsDist_1, "overallrecoverypercent_obsDist_10 = ", overallrecoverypercent_obsDist_10, "overallrecoverypercent_obsDist_30 = ", overallrecoverypercent_obsDist_30, "overallrecoverypercent_obsDist_100 = ", overallrecoverypercent_obsDist_100, "overallrecoverypercent_obsDist_1000 = ", overallrecoverypercent_obsDist_1000)
print("overallsigmaobsDist = ", overallsigmaobsDist, "overallsigmaobsDist_03 = ", overallsigmaobsDist_03, "overallsigmaobsDist_1 = ", overallsigmaobsDist_1, "overallsigmaobsDist_10 = ", overallsigmaobsDist_10, "overallsigmaobsDist_30 = ", overallsigmaobsDist_30, "overallsigmaobsDist_100 = ", overallsigmaobsDist_100, "overallsigmaobsDist_1000 = ", overallsigmaobsDist_1000)



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
print("wholerecoverypercent_obsDist22 = ", wholerecoverypercent_obsDist22, "wholerecoverypercent_obsDist22_03 = ", wholerecoverypercent_obsDist22_03, "wholerecoverypercent_obsDist22_1 = ", wholerecoverypercent_obsDist22_1, "wholerecoverypercent_obsDist22_10 = ", wholerecoverypercent_obsDist22_10, "wholerecoverypercent_obsDist22_30 = ", wholerecoverypercent_obsDist22_30, "wholerecoverypercent_obsDist22_100 = ", wholerecoverypercent_obsDist22_100, "wholerecoverypercent_obsDist22_1000 = ", wholerecoverypercent_obsDist22_1000)
print("sigmaobsDist22 = ", sigmaobsDist22, "sigmaobsDist22_03 = ", sigmaobsDist22_03, "sigmaobsDist22_1 = ", sigmaobsDist22_1, "sigmaobsDist22_10 = ", sigmaobsDist22_10, "sigmaobsDist22_30 = ", sigmaobsDist22_30, "sigmaobsDist22_100 = ", sigmaobsDist22_100, "sigmaobsDist22_1000 = ", sigmaobsDist22_1000)
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
print("overallrecoverypercent_obsDist22 = ", overallrecoverypercent_obsDist22, "overallrecoverypercent_obsDist22_03 = ", overallrecoverypercent_obsDist22_03, "overallrecoverypercent_obsDist22_1 = ", overallrecoverypercent_obsDist22_1, "overallrecoverypercent_obsDist22_10 = ", overallrecoverypercent_obsDist22_10, "overallrecoverypercent_obsDist22_30 = ", overallrecoverypercent_obsDist22_30, "overallrecoverypercent_obsDist22_100 = ", overallrecoverypercent_obsDist22_100, "overallrecoverypercent_obsDist22_1000 = ", overallrecoverypercent_obsDist22_1000)
print("overallsigmaobsDist22 = ", overallsigmaobsDist22, "overallsigmaobsDist22_03 = ", overallsigmaobsDist22_03, "overallsigmaobsDist22_1 = ", overallsigmaobsDist22_1, "overallsigmaobsDist22_10 = ", overallsigmaobsDist22_10, "overallsigmaobsDist22_30 = ", overallsigmaobsDist22_30, "overallsigmaobsDist22_100 = ", overallsigmaobsDist22_100, "overallsigmaobsDist22_1000 = ", overallsigmaobsDist22_1000)



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
print("wholerecoverypercent_obsDist195 = ", wholerecoverypercent_obsDist195, "wholerecoverypercent_obsDist195_03 = ", wholerecoverypercent_obsDist195_03, "wholerecoverypercent_obsDist195_1 = ", wholerecoverypercent_obsDist195_1, "wholerecoverypercent_obsDist195_10 = ", wholerecoverypercent_obsDist195_10, "wholerecoverypercent_obsDist195_30 = ", wholerecoverypercent_obsDist195_30, "wholerecoverypercent_obsDist195_100 = ", wholerecoverypercent_obsDist195_100, "wholerecoverypercent_obsDist195_1000 = ", wholerecoverypercent_obsDist195_1000)
print("sigmaobsDist195 = ", sigmaobsDist195, "sigmaobsDist195_03 = ", sigmaobsDist195_03, "sigmaobsDist195_1 = ", sigmaobsDist195_1, "sigmaobsDist195_10 = ", sigmaobsDist195_10, "sigmaobsDist195_30 = ", sigmaobsDist195_30, "sigmaobsDist195_100 = ", sigmaobsDist195_100, "sigmaobsDist195_1000 = ", sigmaobsDist195_1000)
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
overallsigmaobsDist195_1000 = ((N_totalrecoverableobsDist195_1000**(1/2))/N_totalobsDist195_1000)*100
print("overallrecoverypercent_obsDist195 = ", overallrecoverypercent_obsDist195, "overallrecoverypercent_obsDist195_03 = ", overallrecoverypercent_obsDist195_03, "overallrecoverypercent_obsDist195_1 = ", overallrecoverypercent_obsDist195_1, "overallrecoverypercent_obsDist195_10 = ", overallrecoverypercent_obsDist195_10, "overallrecoverypercent_obsDist195_30 = ", overallrecoverypercent_obsDist195_30, "overallrecoverypercent_obsDist195_100 = ", overallrecoverypercent_obsDist195_100, "overallrecoverypercent_obsDist195_1000 = ", overallrecoverypercent_obsDist195_1000)
print("overallsigmaobsDist195 = ", overallsigmaobsDist195, "overallsigmaobsDist195_03 = ", overallsigmaobsDist195_03, "overallsigmaobsDist195_1 = ", overallsigmaobsDist195_1, "overallsigmaobsDist195_10 = ", overallsigmaobsDist195_10, "overallsigmaobsDist195_30 = ", overallsigmaobsDist195_30, "overallsigmaobsDist195_100 = ", overallsigmaobsDist195_100, "overallsigmaobsDist195_1000 = ", overallsigmaobsDist195_1000)
