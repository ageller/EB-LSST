#OpSim (in OpSimRun1) - http://ops2.lsst.org/docs/current/architecture.html

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
import sqlite3

class OpSim(object):

	def __init__(self, *args,**kwargs):
		self.dbFile = '../input/db/baseline2018a.db' #for the OpSim database

		self.verbose = False
		self.fieldCursor = None
		self.summaryCursor = None

		self.fieldID = [None]
		self.RA = [None]
		self.Dec = [None]
		self.Nobs = [None]
		self.m_5 = [None]
		self.obsDates = [None]
		self.NobsDates = [None]
		self.totalNobs = [None]

		self.obsDist = None

	#database manipulation
	def getCursors(self):
		#gets SQlite cursor to pull information from OpSim
		#https://www.lsst.org/scientists/simulations/opsim/summary-table-column-descriptions-v335
		#http://ops2.lsst.org/docs/current/architecture.html
		db = sqlite3.connect(self.dbFile)
		cursor = db.cursor()

		#cursor.execute("SELECT fieldid, expDate, filter, fiveSigmaDepth FROM summary")  #old, for minion
		cursor.execute("SELECT Field_fieldid, observationStartTime, filter, fiveSigmaDepth FROM ObsHistory")
		self.summaryCursor = np.array(cursor.fetchall()) #NOTE: this takes a LONG time
		print("have summary cursor.")

		#cursor.execute("SELECT fieldid, fieldra, fielddec FROM field") #old, for minion
		cursor.execute("SELECT fieldid, ra, dec FROM Field")
		self.fieldCursor = np.array(cursor.fetchall()) #NOTE: this takes a LONG time
		print("have field cursor.")


	#For OpSim database
	def setFieldID(self, myRA, myDec, deglim = 3.5/2.):
		#uses RA/Dec (from galactic coordinates) to return locatiom's fieldID according to OpSim
		#field-of-view == 3.5-degree diameter (also returned with fieldFov key)

		RA = self.fieldCursor[:,1].astype(float)
		Dec = self.fieldCursor[:,2].astype(float)
		dbCoord = SkyCoord(ra = RA*units.degree, dec = Dec*units.degree, frame='icrs')
		inCoord = SkyCoord(ra = myRA*units.degree, dec = myDec*units.degree, frame='icrs')

		imin, sep2d, dist3d = inCoord.match_to_catalog_sky(dbCoord)

		dbID = (self.fieldCursor[imin,0]).astype('int') 

		mask = np.where(sep2d.to(units.degree).value > deglim)
		#this check apparently isn't necessary because it looks like the entire sky is covered with fieldIDs, but I suppose some of these fieldIDs don't have any observation dates (in the northern hemisphere)
		if (len(mask[0]) > 0):
			print(mask[0])
			print("WARNING: coordinate outside LSST FOV", myRA[mask], myDec[mask])
			dbID[mask] = -999

		if (self.verbose):
			print("have Field ID", dbID)

		self.fieldID = [dbID]

	def getDates(self, ID, filtin):
		#matches FieldID to existing OpSim database ID and matches observation filters to get dates (in seconds since the start of the 
		# survey)
		FieldID = self.summaryCursor[:,0].astype('int')
		date = self.summaryCursor[:,1].astype('float')
		filt = self.summaryCursor[:,2]
		fiveSigmaDepth = self.summaryCursor[:,3].astype('float')

		posIDFilt = np.where(np.logical_and(FieldID == ID, filt == filtin[:-1]))
		if (self.verbose):
			print("posIDFilt = ", posIDFilt, filtin)

		OpSimdates = posIDFilt[0]

		if (len(OpSimdates) < 1):
			return [None], [None]
		else:
			if (self.obsDist == None): #default, use the OpSim dates
				if (self.verbose):
					print('OpSimdates =', OpSimdates)
				dates = np.array([float(d) for d in date[OpSimdates] ])/86400. #converting seconds to days\
				m_5 = np.array([float(x) for x in fiveSigmaDepth[OpSimdates] ])
			else: #alternatively, can use a CDF of dt values to construct dates
				N = round(self.obsDist[filtin]['Nobs'])
				dt = []
				for i in range(N):
					y = np.random.random()
					dt.append(10.**np.interp(y, self.obsDist[filtin]['cdf'], self.obsDist[filtin]['bins']))
				dates = np.cumsum(np.array(dt))
				mpos = np.random.randint(0, high=len(OpSimdates), size=N)
				m_5 = np.array([float(x) for x in fiveSigmaDepth[OpSimdates[mpos]] ])
				#print("in getDate", dates, m_5)
			return dates, m_5

	def setDates(self, i, filters):
		self.obsDates[i] = dict()
		self.NobsDates[i] = dict()
		self.m_5[i] = dict()
		self.totalNobs[i] = 0
		for filt in filters:
			self.obsDates[i][filt], self.m_5[i][filt] = self.getDates(self.fieldID[i], filt)
			#print("in setDates", i, filt, self.obsDates[i][filt])

			self.NobsDates[i][filt] = 0
			if (self.obsDates[i][filt][0] != None):
				self.NobsDates[i][filt] = len(self.obsDates[i][filt])
			self.totalNobs[i] += self.NobsDates[i][filt]
			if (self.verbose):
				print(f'observing with OpSim in filter {filt}, have {self.NobsDates[i][filt]} observations')

	def getAllOpSimFields(self):
		print("getting OpSim fields...")
		self.getCursors()
		FieldID = self.summaryCursor[:,0].astype('int')
		#date = self.summaryCursor[:,1].astype('float')

		self.fieldID = np.array([])
		self.RA = np.array([])
		self.Dec = np.array([])
		self.Nobs = np.array([])

		for x in self.fieldCursor:
			inS = np.where(FieldID == int(x[0]))[0]
			self.Nobs = np.append(self.Nobs, len(inS))
			self.fieldID = np.append(self.fieldID, x[0])
			self.RA = np.append(self.RA, x[1])
			self.Dec = np.append(self.Dec, x[2])
		self.obsDates = np.full_like(self.RA, dict(), dtype=dict)
		self.NobsDates = np.full_like(self.RA, dict(), dtype=dict)
		self.totalNobs = np.full_like(self.RA, 0)
		self.m_5 = np.full_like(self.RA, dict(), dtype=dict)

		print(f'returned {len(self.fieldID)} fields')