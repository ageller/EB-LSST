#!/software/anaconda3.6/bin/python

from newLSSTEBWorker import LSSTEBWorker 
from OpSim import OpSim
import csv
import argparse
import numpy as np
from mpi4py import MPI
import os
import time

import pickle



def define_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-c", "--n_bin", 		type=int, help="Number of binaries per process [100000]")
	parser.add_argument("-o", "--output_file", 	type=str, help="output file name")
	parser.add_argument("-a", "--n_band", 		type=int, help="Nterms_band input for gatspy [2]")
	parser.add_argument("-b", "--n_base", 		type=int, help="Nterms_base input for gatspy [2]")
	parser.add_argument("-s", "--seed", 		type=int, help="random seed []")
	parser.add_argument("-v", "--verbose", 		action='store_true', help="Set to show verbose output")
	parser.add_argument("-l", "--opsim", 		action='store_false', help="set to run LSST OpSim, else run nobs =")

	#https://docs.python.org/2/howto/argparse.html
	args = parser.parse_args()
	#to print out the options that were selected (probably some way to use this to quickly assign args)
	opts = vars(args)
	options = { k : opts[k] for k in opts if opts[k] != None }
	print(options)

	return args

def apply_args(worker, args):


	if (args.n_bin is not None):
		worker.n_bin = args.n_bin
		
	if (args.output_file is not None):
		worker.ofile = args.output_file

	if (args.n_band is not None):
		worker.n_band = args.n_band
	if (args.n_base is not None):
		worker.n_base = args.n_base

	worker.verbose = args.verbose
	worker.doOpSim = args.opsim

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
	#these are not all finished, but this is OK for now
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
			IDs.append(int(f[0:4]))

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

	nfields = 5292 #total number of fields from OpSim
	finishedIDs = getFinishedIDs()
	nfields -= len(finishedIDs)
	nfieldsPerCore = int(np.floor(nfields/size))
	print(f"nfields={nfields}, nfieldsPerCore={nfieldsPerCore}")

	sendbuf = np.empty((size, 3*nfieldsPerCore), dtype='float64')
	recvbuf = np.empty(3*nfieldsPerCore, dtype='float64')

	galModelDir = 'TRILEGALmodels'
	if (rank == root):
		if not os.path.exists(galModelDir):
			os.makedirs(galModelDir)
		if not os.path.exists('output_files'):
			os.makedirs('output_files')

		OpS = OpSim()
		OpS.dbFile = '/projects/p30137/ageller/EBLSST/input/db/baseline2018a.db' #for the OpSim database	
		OpS.getAllOpSimFields()

		unfin = []
		for i, ID in enumerate(OpS.fieldID):
			if ID not in finishedIDs: 
				unfin.append(i)
		OpS.fieldID = OpS.fieldID[unfin]
		OpS.RA = OpS.RA[unfin] 
		OpS.Dec = OpS.Dec[unfin]

		nfields = len(OpS.fieldID)
		print(f"rank 0 nfields={nfields}")

		#scatter the fieldID, RA, Dec 
		#get as close as we can to having everything scattered
		maxIndex = min(nfieldsPerCore*size, nfields)
		output = np.vstack((OpS.fieldID[:maxIndex], OpS.RA[:maxIndex], OpS.Dec[:maxIndex])).T

		print("check",maxIndex, nfieldsPerCore, size, nfields, output.shape)
		print("reshaping to send to other processes")
		sendbuf = np.reshape(output, (size, 3*nfieldsPerCore))




	#scatter to the all of the processes
	comm.Scatter(sendbuf, recvbuf, root=root) 
	#now reshape again to get back to the right format
	fieldData = np.reshape(recvbuf, (nfieldsPerCore, 3))	

	#print("rank", rank, fieldData)

	#add on any extra fields to rank =0
	if (rank == 0):
		if (nfieldsPerCore*size < nfields):
			print("adding to rank 0")
			extra = np.vstack((OpS.fieldID[maxIndex:], OpS.RA[maxIndex:], OpS.Dec[maxIndex:])).T
			fieldData = np.vstack((fieldData, extra))

	#define the worker
	worker = LSSTEBWorker()
	worker.filterFilesRoot = '/projects/p30137/ageller/EBLSST/input/filters/'
	worker.filters = filters
	#os.environ['PYSYN_CDBS'] = '/projects/p30137/ageller/PySynphotData'
	print(f"PYSYN_CDBS environ = {os.environ['PYSYN_CDBS']}")
	#check for command-line arguments
	apply_args(worker, args)	
	if (worker.seed == None):
		worker.seed = 1234
	worker.seed += rank

	#redefine the OpSim fieldID, RA, Dec and the run through the rest of the code
	fields = fieldData.T
	OpS = OpSim()
	OpS.dbFile = '/projects/p30137/ageller/EBLSST/input/db/baseline2018a.db' #for the OpSim database	
	OpS.getCursors()
	OpS.fieldID = fields[0]
	OpS.RA = fields[1]
	OpS.Dec = fields[2]
	OpS.obsDates = np.full_like(OpS.RA, dict(), dtype=dict)
	OpS.NobsDates = np.full_like(OpS.RA, dict(), dtype=dict)
	OpS.m_5 = np.full_like(OpS.RA, dict(), dtype=dict)
	OpS.totalNobs = np.full_like(OpS.RA, 0)
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
	k = 0
	for i in range(len(fields[0])):
		worker.n_bin = args.n_bin
		if (worker.OpSim.fieldID[i] not in finishedIDs and worker.OpSim.fieldID[i] != -1):
			#initialize
			print(f"RANK={rank}, OpSimi={i}, ID={worker.OpSim.fieldID[i]}")
			passed = worker.initialize(OpSimi=i) #Note: this will not redo the OpSim class, because we've set it above
	
			#set up the output file
			worker.ofile = 'output_files/'+str(int(worker.OpSim.fieldID[i])).zfill(4) + ofile

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

			#write header, this will not contain the galaxy Nstars number
			if (not append and not passed):
				worker.writeOutputLine(OpSimi=i, header=True)
				csvfile.flush()

			if (passed):


				#run through ellc and gatspy
				worker.getGalaxy(i)
				gxDat = worker.sampleGalaxy()

				#write header, this will  contain the galaxy Nstars number
				if (not append):
					worker.writeOutputLine(OpSimi=i, header=True)
					csvfile.flush()

				print(f'Nlines in gxDate={len(gxDat)} for ID={worker.OpSim.fieldID[i]}')

				for j, line in enumerate(gxDat):
					line = gxDat[j]
	
					#define the binary parameters
					try:
						worker.getEB(line, OpSimi=i)
						print(f"RANK={rank}, OpSimi={i}, linej={j}, ID={worker.OpSim.fieldID[i]}, pb={EB.period}")
		
						if (worker.EB.observable):
							worker.run_ellc(k)
							if (worker.EB.observable):
								worker.run_gatspy(k)
		
						worker.writeOutputLine()
						csvfile.flush()
					except:
						print("WARNING: bad input line", line)
			else:
				worker.writeOutputLine(OpSimi=i, noRun=True)
				csvfile.flush()
	
	
			csvfile.close()

		#get ready for the next field
		worker.Galaxy = None
		worker.BreivikGal = None
		




