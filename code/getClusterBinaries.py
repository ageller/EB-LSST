"""Class of getClusterBinaries that runs our HS-period finding code, uses calculated sigma values
Takes in these params: mass, Rhm, age, metallicity, velocity dispersion, and number of binaries requested
should calculate hard-soft binary for each binary drawn with cosmic and return # of binaries requested on input
output of this should be a numpy array of arrays
"""

# Importing needed mdoules
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import independent
from cosmic.sample.sampler import multidim
from cosmic.evolve import Evolve
import numpy as np
import pandas as pd

names_gc = ['ID_x', 'Name', 'RA', 'DEC', 'L','B','R_Sun','R_gc','X','Y', 'Z', 'key_0','[Fe/H]_x', 'wt', 'E(B-V)_x',\
 'V_HB','(m-M)V_x', 'V_t', 'M_V,t', 'U-B', 'B-V', 'V-R', 'V-I', 'spt', 'ellip', 'ID_y', 'v_r', '+/-', 'v_LSR' ,'sig_v' ,'+/-.1', 'c', 'r_c', 'r_h', 'mu_V',\
  'rho_', 'lg(tc)', 'lg(th)', 'Mcl[Msun]', 'rh[pc]', '[Fe/H]_y', 'age[Gyr]', '(m-M)V_y', 'E(B-V)_y', 'log10(rho[Msun]/pc^3)',\
 'rc', 'sigma0[km/s]', 'esigma0[km/s]', 'fb', 'efb', '[M/H]', 'Rgc[kpc]','Rsun[kpc]']
names_oc = ['Cluster_name', 'RA', 'DEC', 'l', 'b', 'Dist Mod', 'EB-V', 'Age', 'ST' ,'Z', 'Diam', 'Fe/H', 'MRV',\
 'pm RA', 'pm Dec', 'logM[Msun]', 'rtP[pc]', 'log(t[yr])K', 'rcK[pc]', 'rtK[pc]', 'Rhm[pc]',\
  '[Fe/H]K]', 'deltaV', 'sigdV', '[FeH]', 'sigFeH', 't', 'sigt', 'logt' ,'Rgc' ,'z' ,'Diam[pc]', 'd[pc]']


names_clusters = ['Name', 'RA', 'Dec', 'dist[kpc]', 'rh[pc]', 'r_c', 'mass[Msun]', 'Age',\
       'Age[Myr]', 'Z', 'sigma[km/s]', 'OpSim ID', 'OpSim RA', 'OpSim Dec',\
       'Source Flag', 'Cluster Type']

# Class that samples hard binaries for every cluster and takes in outputs from Aaron
class getClusterBinaries(object):
	"""
	Class for us to run all of our cosmic evolution of binaries within globular and open clusters.
	Will find the hard-soft cutoff and use either given or calculated velocity dispersions for both types of clusters
	Then will loop through each cluster (in our compiled cluster data file) and run this class on each

	"""

	#def __init__(self, mass, Rhm, age, Z, sigma, Nbin, cluster_file_path):#dont't think we'll need filepath stuff
	def __init__(self, age, Z, sigma, Nbin):#dont't think we'll need filepath stuff
		# Input data from clusters
		#self.mass = mass
		#self.Rhm = Rhm
		self.age = age
		self.Z = Z
		self.sigma = sigma
		self.Nbin = Nbin
		#self.cluster_file_path = cluster_file_path#Full files paths to gc and oc data


		# Class variables for later
		self.period_hardsoft = None
		self.output = None
		self.InitialBinaries = None
		self.bpp = None
		self.bcm = None
		self.bcmEvolved = None
		self.dist = None
		self.inc = None
		self.omega = None
		self.OMEGA = None
		self.random_seed = 0

		# BSE dictionary copied from cosmic's documentation (unchanged): https://cosmic-popsynth.github.io
		self.BSEDict = {'xi': 0.5, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 0, 'alpha1': 1.0, \
		'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 1.0, 'ck': -1000, 'bwind': 0.0, 'lambdaf': 1.0, \
		'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'nsflag': 3, 'ceflag': 0, 'eddfac': 1.0, 'merger': 0, 'ifflag': 0, \
		'bconst': -3000, 'sigma': 265.0, 'gamma': -2.0, 'ppsn': 1,\
		 'natal_kick_array' : [-100.0,-100.0,-100.0,-100.0,-100.0,-100.0], 'bhsigmafrac' : 1.0, 'polar_kick_angle' : 90,\
		  'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 0, \
		  'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsnp' : 2.5, 'ecsn_mlow' : 1.6, 'aic' : 1, 'sigmadiv' :-20.0}
# # Method to read in globular and open cluster files - same as before
# 	def cluster_readin(self, cluster_file_path):
# 		Clusters = pd.read_csv(self.cluster_file_path, sep = ' ', header = 0, names = names_clusters)

# 		self.Clusters = Clusters

# Method to calculate the hard-soft boundary for binaries
	def get_Phs(self, m1=1.0, m2=1.0,m3=0.5):
		"""
		Function to calculate the hard-soft period cutoff given by Eq 1 in Geller, Leigh 2015
    	Masses are in solar masses (converted later to kg), m1 and m2 are binary component masses,
    	and m3 is the mass of the incoming (disrupting) object, velocity dispersions are given in km/s
		
    	"""
    	#do this ahead of time when you create your table
    	# get sigma value (first see if it has one, if not, if mass then calculate sigma, else draw randomly)
    	# for random draw you will have predefined loc and scale for numpy.random.normal for OCs and GCs
    	#then calulate phs but use all the self.period_hardsoft

		G = 1.334 * (10 ** 11) # Gravitational Constant in units of km^3 M_sun ^ -1 s ^ -2 (consistent with cosmic output units)

		const = (np.pi*G/np.sqrt(2))
		sigma = self.sigma

		Phs = const * (((m1 * m2)/m3)**(3/2)) * np.sqrt(m1 + m2) * (np.sqrt(3) * sigma) ** -3
		Phs = Phs / (24 *3600)#Converting hard-soft period from seconds to days

		#print("hard-soft boundary", Phs, np.log10(Phs))
		self.period_hardsoft = np.round(np.log10(Phs),decimals=1)#rounding to 2 decimal places for cosmic




	# New sampler function - only pulls initial binaries, want them to be hard binaries so we set maximum period cutoff with porb_hi and porb_lo
	def Initial_Binary_Sample(self):
		"""

		Creates and evolves a set of binaries with given 
		age (to evolve to), number of binaries, metallicity, and velocity dispersion.

		"""
		# Initial (input) binares -- using sampler method from cosmic #1234 - random seed
		print("initial binary input:",self.random_seed, self.age, self.Z, self.Nbin, self.sigma, 	self.period_hardsoft)
		InitialBinaries, sampled_mass, n_sampled = InitialBinaryTable.sampler('multidim',\
		 [0,12], [0,12],self.random_seed,1, 'delta_burst', self.age, self.Z, self.Nbin, porb_lo = 0.15, porb_hi = self.period_hardsoft)



		# Input binary params
		p_i = InitialBinaries['porb']
		m1_i = InitialBinaries['mass1_binary']#In Msun units
		m2_i = InitialBinaries['mass2_binary']
		ecc_i = InitialBinaries['ecc']
		tphysf_i = InitialBinaries['tphysf']#Time to evolve to (Myr)
		Z_i = InitialBinaries['metallicity']
		

		self.InitialBinaries = InitialBinaries


	# Evolving hard binaries from our initial binary table above
	def EvolveBinaries(self):

		"""Takes Initial (hard) binaries from above and evolves them"""
		bpp, bcm, initC  = Evolve.evolve(initialbinarytable = self.InitialBinaries, BSEDict = self.BSEDict)

		self.bpp = bpp
		self.bcm = bcm
		self.bcmEvolved = self.bcm.loc[self.bcm['tphys'] == self.age]
		#print(self.bcmEvolved)


	# Method to generate final output array of arrays from Aaron
	def EB_output(self):
		#double check that we have the correct units

		##################
		#we need to grab only the final values at the age of the cluster
		###############
		Nvals = len(self.bcmEvolved['mass_1'].values)

		# Inclination and omega values
		self.inc = np.arccos(2.*np.random.uniform(0,1,Nvals) - 1.)
		self.omega = np.random.uniform(0,2*np.pi,Nvals)
		self.OMEGA = np.random.uniform(0,2*np.pi,Nvals)

		noneArray = np.array([None for x in range Nvals])
		output = np.array([self.bcmEvolved['mass_1'].values, self.bcmEvolved['mass_2'].values, \
			np.log10(self.bcmEvolved['porb'].values), self.bcmEvolved['ecc'].values, \
			10.**self.bcmEvolved['rad_1'].values, 10.**self.bcmEvolved['rad_2'].values,\
			10.**self.bcmEvolved['lumin_1'].values, 10.**self.bcmEvolved['lumin_2'].values, \
			noneArray, noneArray, noneArray, np.ones(Nvals)*self.dist, \
			self.inc, self.OMEGA, self.omega, \
			noneArray, np.ones(Nvals)*self.Z, \
			10.**self.bcmEvolved['teff_1'], 10.**self.bcmEvolved['teff_2'], \
			self.bcmEvolved['k1'], self.bcmEvolved['k2']])

		#print('original output array: ',output)


		#transposes the above array to make it look like original pandas dataframe - turns rows into columns and vice versa
		self.output = output.T

		#print('\'Transposed\' output matrix',self.output)

		# self.EB = np.empty(17)
		# self.EB[0] = self.bcm['mass_1'].values#Evolved mass 1, in units of Msun
		# self.EB[1] = self.bcm['mass_2'].values#Evolved mass 2, in units of Msun
		# self.EB[2] = self.bcm['porb'].values#Orbital period of evolved binaries
		# self.EB[3] = self.bcm['ecc'].values#Eccentricity of evolved binary
		# self.EB[4] = self.bcm['rad_1'].values#Evolved Binary radius 1 in solar units
		# self.EB[5] = self.bcm['rad_2'].values

		#print(EB)

	def runAll(self):
		#self.cluster_readin(self.cluster_file_path)
		self.get_Phs()
		self.Initial_Binary_Sample()
		self.EvolveBinaries()
		self.EB_output()


# ########################################## Test Instances #########################################################################

if __name__ == "__main__":

	# Test instances, creating instance of class to make sure everything's working right
	test_binary = getClusterBinaries(12344., 0.0, 11., 10)
	test_binary.random_seed = 1111
	#test_binary.period_hardsoft = 5.1
	#MyBinaries = test_binary.Initial_Binary_Sample(13000, 10, 0.01, np.random.randint(1,200))

	# # Evolving the initial binaries from out test class instance
	#e_binary = test_binary.EvolveBinaries()
	#binary = test_binary.EB_output()
	# print(e_binary)
	# print(binary)

	test_binary.runAll()
	print(test_binary.bcmEvolved)

	# EB.m1 = line[0] #Msun
	# EB.m2 = line[1] #Msun
	# EB.period = 10.**line[2] #days
	# EB.eccentricity = line[3]
	# EB.r1 = line[4] #Rsun
	# EB.r2 = line[5] #Rsun
	# EB.L1 = line[6] #Lsun
	# EB.L2 = line[7] #Lsun
	# EB.xGx = line[8] #unused kpc
	# EB.yGx = line[9] #unused kpc
	# EB.zGx = line[10] #unutsed kpc
	# EB.dist = line[11] #kpc
	# EB.inclination = line[12] #degrees
	# EB.OMEGA = line[13] #degrees
	# EB.omega = line[14] * #degrees
	# EB.AV = line[15]  #optional, if not available, make it None
	# EB.M_H = line[16]





