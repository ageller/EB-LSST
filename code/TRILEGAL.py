import numpy as np
import pandas as pd
import os
import scipy.stats

#######################
#3rd party codes
#for TRILEGAL and maybe also A_V
from vespa_update import trilegal as trilegal_update
p = os.environ['PATH']
pv = os.path.join(os.getcwd(),'vespa_update')
p2 = pv+':'+p
os.environ['PATH'] = p2


class TRILEGAL(object):
	def __init__(self, *args,**kwargs):
		self.area = np.pi*(3.5/2.)**2. #square degrees (LSST FoV)
		self.area0frac = 1.
		self.maglim = 25
		self.maglimFilter = 3 #rband in LSST
		self.sigma_AV = 0.1 #default
		self.binaries = False
		self.filterset = 'lsst' 
		self.tmpfname = 'TRILEGAL_model.h5'
		self.tmpdir = '.'
		self.archivedir = '.'
		self.deleteModel = True

		self.model = None
		self.KDE = None

		self.RA = None
		self.Dec = None
		self.fieldID = None
		self.Nstars = 0

		self.seeing = 0.5
		self.resEl = 0. #calculated below based on seeing
		self.starsPerResEl = 0.

		self.shuffle = True

	def downloadModel(self):
		passed = False
		self.area *= self.area0frac
		print("downloading model",self.tmpfname, self.area)
		while (not passed):
			passed = trilegal_update.get_trilegal(self.tmpfname, self.RA, self.Dec, folder=self.tmpdir, galactic=False, \
				filterset=self.filterset, area=self.area, maglim=self.maglim, maglimFilter=self.maglimFilter, binaries=self.binaries, \
				trilegal_version='1.6', sigma_AV=self.sigma_AV, convert_h5=True)
			if (not passed):
				self.area *= 0.9
				print(f"reducing TRILEGAL area to {self.area}...")

	def setModel(self, download = True):
		area0 = self.area
		filePath = os.path.join(self.archivedir,self.tmpfname)

		if (not os.path.exists(filePath)):
			download = True
			filePath = os.path.join(self.tmpdir,self.tmpfname)

		if (download):
			self.downloadModel()

		with pd.HDFStore(filePath) as store:
			attrs = store.get_storer('df').attrs
			if ('area' in attrs.trilegal_args):
				self.area = float(attrs.trilegal_args['area'])

		self.model = pd.read_hdf(filePath)
		self.Nstars = len(self.model) * (area0/self.area)

		#check for crowding
		#first see if we need to normalize this to the LSST FoV
		#LSSTFoV = np.pi*(3.5/2.)**2.
		#FoVratio = LSSTFoV/self.area #ideally this ratio is 1.

		#assume a uniform surface density stars/square degree
		#surfaceDensity = self.Nstars*FoVratio/LSSTFoV
		surfaceDensity = self.Nstars/self.area

		#stars/resolution element
		self.resEl = np.pi*(self.seeing/2./3600.)**2. 
		self.starsPerResEl = np.array(surfaceDensity*self.resEl)

		#add the distance
		logDist = np.log10( 10.**(self.model['m-M0'].values/5.) *10. / 1000.) #log(d [kpc])
		self.model['logDist'] = logDist

		if (self.shuffle):
			self.model = self.model.sample(frac=1).reset_index(drop=True)

		if (self.deleteModel):
			os.remove(filePath)

		data = np.vstack((self.model['logL'].values, self.model['logTe'].values, self.model['logg'].values, \
						self.model['logDist'].values, self.model['Av'].values, self.model['[M/H]'].values))
		self.KDE = scipy.stats.gaussian_kde(data)

		print(f'downloaded TRILEGAL model for ID={self.fieldID}, RA={self.RA}, DEC={self.Dec}, Nstars={self.Nstars}, Nstars/resEl={self.starsPerResEl}')
