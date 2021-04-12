#NOTE: this is WAY too slow to do for all stars in my samples

import pandas as pd
import numpy as np
import os
from astropy import units, constants

import sys
sys.path.insert(0, '/Users/ageller/WORK/LSST/onGitHub/EBLSST/code')
from SED import SED

from dust_extinction.parameter_averages import F04


RV=3.1
wavelength = (552. + 691.)/2.
filterFilesRoot = '/Users/ageller/WORK/LSST/onGitHub/EBLSST/input/filters/'
#filterFilesRoot = '/projects/p30137/ageller/EBLSST/input/filters/'

def getrMagBinary(L1, T1, g1, r1, L2, T2, g2, r2, M_H, dist, AV, extVal):


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


	Lconst1 = SED1.getLconst()
	Lconst2 = SED2.getLconst()

	Ared = extVal*AV

	Fv1 = SED1.getFvAB(dist*units.kpc, 'r_', Lconst = Lconst1)
	Fv2 = SED2.getFvAB(dist*units.kpc, 'r_', Lconst = Lconst2)
	Fv = Fv1 + Fv2
	return -2.5*np.log10(Fv) + Ared #AB magnitude 


def getrMagSingle(L1, T1, g1, r1, M_H, dist, AV, extVal):


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

	Lconst1 = SED1.getLconst()

	Ared = extVal*AV
	Fv = SED1.getFvAB(dist*units.kpc, 'r_', Lconst = Lconst1)

	return -2.5*np.log10(Fv) + Ared #AB magnitude 


def file_len(fname):
	i = 0
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1


def updateMag(directory = 'output_files'):

	files = os.listdir(directory)
	IDs = []

	#one option for getting the extinction
	ext = F04(Rv=RV)
	extVal = ext(wavelength*units.nm)

	for index, f in enumerate(files):
		fl = file_len(os.path.join(directory,f))

		if (fl >= 4):
			#read in the header
			header = pd.read_csv(os.path.join(directory,f), nrows=1)

			data = pd.read_csv(os.path.join(directory,f), header = 2).fillna(-999)

			#swap locations so that m1 is always > m2
			check = data.loc[(data['m2'] > data['m1'])]
			if (len(check.index) > 0):
				for i, row in check.iterrows():
					m1tmp = row['m1']
					data.at[i, 'm1'] = row['m2']
					data.at[i, 'm2'] = m1tmp
					r1tmp = row['r1']
					data.at[i, 'r1'] = row['r2']
					data.at[i, 'r2'] = r1tmp
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

			#get the magnitudes if needed
			rMag = data['appMagMean_r'].values
			for i, row in data.iterrows():
				if (row['appMagMean_r'] == -999.):
					try:
						rMag[i] = getrMagBinary(row['L1'], row['Teff1'], logg1[i], row['r1'], row['L2'], row['Teff2'], logg2[i], row['r2'], row['[M/H]'], row['d'], row['Av'], extVal)
					except:
						try:
							rMag[i] = getrMagSingle(row['L1'], row['Teff1'], logg1[i], row['r1'],  row['[M/H]'], row['d'], row['Av'], extVal)
						except:
							print('bad mag',row['L1'], row['Teff1'], logg1[i], row['r1'], row['L2'], row['Teff2'], logg2[i], row['r2'], row['[M/H]'], row['d'], row['Av'])
					print(i, rMag[i])
				# if (i > 10):
				# 	break

			data['appMagMean_r'] = rMag

			fnew = os.path.join(directory,f+'.rMag')

			#print the header lines
			fread = open(os.path.join(directory,f),'r')
			lines = fread.readlines()
			fread.close()
			fwrite = open(fnew, 'w')
			fwrite.write(lines[0])
			fwrite.write(lines[1])
			fwrite.close()

			#now the rest of the data
			data.to_csv(fnew, mode='a', index=False)

		print(round(i/len(files),4), f)

if __name__ == "__main__":

	updateMag('/Users/ageller/WORK/LSST/fromQuest/clusters/GlobularClusters/withCrowding/TRILEGALrband/eccEclipse/baseline/output_files')
	#updateMag('output_files')

