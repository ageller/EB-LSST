#!/software/anaconda3.6/bin/python

import csv
import argparse
import numpy as np
import pandas as pd
from mpi4py import MPI
import os
from astropy.coordinates import SkyCoord


######################
#my code
from LSSTEBClusterWorker import LSSTEBClusterWorker
from OpSim import OpSim


def define_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-n", "--n_bin", 		type=int, help="Number of binaries per process [100000]")
	parser.add_argument("-o", "--output_file", 	type=str, help="output file name")
	parser.add_argument("-i", "--input_file", 	type=str, help="input file name")
	parser.add_argument("-s", "--seed", 		type=int, help="random seed []")
	parser.add_argument("-v", "--verbose", 		action='store_true', help="Set to show verbose output [true]")

	#https://docs.python.org/2/howto/argparse.html
	args = parser.parse_args()
	#to print out the options that were selected (probably some way to use this to quickly assign args)
	opts = vars(args)
	options = { k : opts[k] for k in opts if opts[k] is not None }
	print(options)

	return args

def apply_args(worker, args):


	if (args.n_bin is not None):
		worker.n_bin = args.n_bin
		
	if (args.output_file is not None):
		worker.ofile = args.output_file


	worker.verbose = args.verbose

	#set the random seed
	if (args.seed is not None):
		worker.seed = args.seed

def file_len(fname):
	i = -1
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

def getFinishedIDs(d='output_files', Nbins = 40000):
	if (not os.path.exists(d)):
		return []
	files = os.listdir(d)
	IDs = []
	for f in files:
		n = file_len(os.path.join(d,f))
		done = False
		#if the file made it to the end (2 header rows, 1 line about OpSim)
		if (n >= Nbins + 3):
			done = True
		else:
			#if there were no OpSim observations
			if (n == 4):
				last = ' '
				with open(os.path.join(d,f), 'r') as fh:
					for line in fh:
						pass
					last = line
				if (last[0:2] == '-1'):
					done = True

		if done:
			p1 = f.rfind('__')
			IDs.append(f[0:p1])

	return IDs

if __name__ == "__main__":

	filters = ['u_', 'g_', 'r_', 'i_', 'z_', 'y_']

	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	sendbuf = None
	root = 0

	args = define_args()
	if (args.n_bin == None):
		args.n_bin = 4

	if (args.input_file == None):
		args.input_file = 'GCdataForEBLSST.csv'

	#these tables should contain (at least) the cluster Name, Mass, Distance, Metallicity, Rhm, Age, and OpSim ID, RA, Dec
	clusterDF = pd.read_csv(args.input_file).fillna(0.)

#########################################
#########################################
	nClusters = len(clusterDF) #total number of clusters

	finishedIDs = getFinishedIDs(Nbins = args.n_bin)
	nClusters -= len(finishedIDs)
	nClustersPerCore = int(np.floor(nClusters/size))
	print(f"nClusters={nClusters}, nClustersPerCore={nClustersPerCore}")

	nVals = 10
	sendbuf = np.empty((size, nVals*nClustersPerCore), dtype='float64')
	recvbuf = np.empty(nVals*nClustersPerCore, dtype='float64')

	galModelDir = 'TRILEGALmodels'
	if (rank == root):
		if not os.path.exists('output_files'):
			os.makedirs('output_files')
		if not os.path.exists(galModelDir):
			os.makedirs(galModelDir)

		OpS = OpSim()
		OpS.dbFile = '/projects/p30137/ageller/EBLSST/input/db/baseline2018a.db' #for the OpSim database	
		OpS.getAllOpSimFields()

		clusterName = []
		clusterMass = []
		clusterDistance = []
		clusterMetallicity = []
		clusterAge = []
		clusterRhm = []
		clusterVdisp = []
		clusterOpSimID = []
		clusterOpSimRA = []
		clusterOpSimDec = []
		for i, ID in enumerate(clusterDF['ID']):
			if ID.replace(" ","_") not in finishedIDs: 
				clusterOpSimID.append(clusterDF['OpSimID'][i])
				clusterOpSimRA.append(clusterDF['OpSimRA[deg]'].values[i])
				clusterOpSimDec.append(clusterDF['OpSimDec[deg]'].values[i])

				clusterName.append(clusterDF.index[i]) #because I can't include strings
				clusterMass.append(clusterDF['mass[Msun]'][i])
				clusterDistance.append(clusterDF['dist[pc]'][i]/1000.) #code wants kpc
				clusterMetallicity.append(clusterDF['[Fe/H]'][i])
				clusterAge.append(clusterDF['age[Myr]'][i])
				clusterRhm.append(clusterDF['rhm[pc]'][i])
				clusterVdisp.append(clusterDF['sigma_v0_z[km/s]'][i])


		nfields = len(clusterName)
		print(f"rank 0 nfields={nfields}")
		print('Names:',clusterDF['ID'].iloc[clusterName])

		#scatter the fieldID, RA, Dec 
		#get as close as we can to having everything scattered
		maxIndex = min(int(nClustersPerCore)*int(size), int(nfields))
		output = np.vstack((
			np.array(clusterOpSimID)[:maxIndex],
			np.array(clusterOpSimRA)[:maxIndex],
			np.array(clusterOpSimDec)[:maxIndex],
			np.array(clusterName)[:maxIndex], 
			np.array(clusterMass)[:maxIndex], 
			np.array(clusterDistance)[:maxIndex], 
			np.array(clusterMetallicity)[:maxIndex], 
			np.array(clusterAge)[:maxIndex], 
			np.array(clusterRhm)[:maxIndex], 
			np.array(clusterVdisp)[:maxIndex], 
		)).astype('float64').T

		print("check",maxIndex, nClustersPerCore, size, nfields, output.shape)
		print("reshaping to send to other processes")
		sendbuf = np.reshape(output, (size, nVals*nClustersPerCore))
		# print("OpSimID",clusterOpSimID)
		# print("OpSimRA",clusterOpSimRA)
		# print("OpSimDec",clusterOpSimDec)
		# print("Name",clusterName)
		# print("Mass",clusterMass)
		# print("Distance",clusterDistance)
		# print("Z",clusterMetallicity)
		# print("Age",clusterAge)
		# print("Rhm",clusterRhm)
		# print("Vdisp",clusterVdisp)
		# print("Type",clusterType)


	#scatter to the all of the processes
	comm.Scatter(sendbuf, recvbuf, root=root) 
	#now reshape again to get back to the right format
	fieldData = np.reshape(recvbuf, (nClustersPerCore, nVals))	

	#add on any extra fields to rank =0
	if (rank == root):
		if (nClustersPerCore*size < nClusters):
			print("adding to rank ", root)
			extra = np.vstack((
				np.array(clusterOpSimID)[maxIndex:],
				np.array(clusterOpSimRA)[maxIndex:],
				np.array(clusterOpSimDec)[maxIndex:],
				np.array(clusterName)[maxIndex:], 
				np.array(clusterMass)[maxIndex:], 
				np.array(clusterDistance)[maxIndex:], 
				np.array(clusterMetallicity)[maxIndex:], 
				np.array(clusterAge)[maxIndex:], 
				np.array(clusterRhm)[maxIndex:], 
				np.array(clusterVdisp)[maxIndex:], 
			)).astype('float64').T
			fieldData = np.vstack((fieldData, extra))


	#put this back in the right orientation
	fields = fieldData.T

	#define the worker
	worker = LSSTEBClusterWorker()
	
	#check for command-line arguments
	apply_args(worker, args)	
	if (worker.seed == None):
		worker.seed = 1234
	worker.seed += rank

	worker.filterFilesRoot = '/projects/p30137/ageller/EBLSST/input/filters/'
	worker.filters = filters
	worker.clusterName = clusterDF['ID'].iloc[fields[3].astype('int')].values
	worker.clusterMass = fields[4]
	worker.clusterDistance = fields[5]
	worker.clusterMetallicity = fields[6]
	worker.clusterAge = fields[7]
	worker.clusterRhm = fields[8]
	worker.clusterVdisp = fields[9]

	#os.environ['PYSYN_CDBS'] = '/projects/p30137/ageller/PySynphotData'
	#print(f"PYSYN_CDBS environ = {os.environ['PYSYN_CDBS']}")

	#redefine the OpSim fieldID, RA, Dec and the run through the rest of the code
	OpS = OpSim()
	OpS.dbFile = '/projects/p30137/ageller/EBLSST/input/db/baseline2018a.db' #for the OpSim database	
	OpS.getCursors()
	OpS.fieldID = fields[0]
	OpS.RA = fields[1]
	OpS.Dec = fields[2]
	OpS.obsDates = np.full_like(OpS.fieldID, dict(), dtype=dict)
	OpS.NobsDates = np.full_like(OpS.fieldID, dict(), dtype=dict)
	OpS.m_5 = np.full_like(OpS.fieldID, dict(), dtype=dict)
	OpS.totalNobs = np.full_like(OpS.fieldID, 0)
	#this will contain the distribution of dt times, which can be used instead of OpSim defaults
	#OpS.obsDist = pickle.load(open("OpSim_observed_dtDist.pickle", 'rb'))

	worker.OpSim = OpS
	#worker.OpSim.verbose = True

	galDir = os.path.join(galModelDir, str(rank))
	if not os.path.exists(galDir):
		os.makedirs(galDir)
	worker.galDir = galDir

	#add a delay here to help with the get_trilegal pileup?
	time.sleep(5*rank)

	ofile = worker.ofile
	for i in range(len(worker.clusterName)):
		worker.n_bin = args.n_bin
		worker.clusterName[i] = worker.clusterName[i].replace(' ','_')
		nme = worker.clusterName[i]
		if (nme not in finishedIDs and worker.clusterName[i] != -1):
			#initialize
			print(f"RANK={rank}, OpSimi={i}, ID={worker.OpSim.fieldID[i]}, cluster={worker.clusterName[i]}")
			passed = worker.initialize(OpSimi=i) #Note: this will not redo the OpSim class, because we've set it above
	
			#set up the output file
			worker.ofile = 'output_files/' + nme + '__' + ofile

			#check if this is a new file or if we are appending
			append = False
			if os.path.exists(worker.ofile):
				n = file_len(worker.ofile)
				#in this scenario, the first few lines were written but it died somewhere. In this case, we don't write headers.  Otherwise, just start over
				if (n >= 3):
					append = True


			if (append):
				worker.n_bin -= (n-3)
				print(f'appending to file {worker.ofile}, with n_bins = {n-3}')
				csvfile = open(worker.ofile, 'a')	
			else:
				print(f'creating new file {worker.ofile}')
				csvfile = open(worker.ofile, 'w')	

			worker.csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

			#write header
			if (not append):
				worker.writeOutputLine(OpSimi=i, header=True)
				csvfile.flush()

			if (passed):
				#run through ellc and gatspy
				worker.getGalaxy(i)

				#get the output from Andrew's cluster code
				clusterDat = worker.sampleCluster(i)

				print(f'Nlines in clusterDat={len(clusterDat)} for ID={worker.OpSim.fieldID[i]}')

				for j, line in enumerate(clusterDat):

					#define the binary parameters
					worker.getEB(line, OpSimi=i)
					print(f"RANK={rank}, OpSimi={i}, linej={j}, ID={worker.OpSim.fieldID[i]}, pb={worker.EB.period}")
	
					if (worker.EB.observable):
						worker.run_ellc()
						if (worker.EB.observable):
							worker.run_gatspy()
	
					worker.writeOutputLine()
					csvfile.flush()
			else:
				worker.writeOutputLine(OpSimi=i, noRun=True)
				csvfile.flush()
	
	
			csvfile.close()

		worker.Galaxy = None

		




