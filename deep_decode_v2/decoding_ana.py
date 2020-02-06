

# Analyze latent decoding from deepDecode

import os
import statistics 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('seaborn-white')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Georgia'

from utils.helper import latent_models, latent_grid
from utils.visualize import plot_latent_boundary
from utils.data_utils import filter_trials


dataPath = os.path.join('..', 'data')
figPath = os.path.join('figs')
ratNames = ['Barat', 'Buchanan', 'Mitt', 'Stella', 'SuperChris']
ratName = ratNames[4]
latentInd = 0

# Load in the data
print('Loading data...')

# Build paths and load the trial and latent decoding data
trlPath = os.path.join(dataPath, ratName, ratName.lower() + '_trial_info.npy')
trial_info = np.load(trlPath)
trial_indices = filter_trials(trial_info)
target = trial_info[trial_indices, 3] - 1


latentName = os.path.join('computeData', '{name}_latent_{index}.npy'.format(name=ratName, index=latentInd))
latentData = np.load(latentName)

latentDataSelect = latentData[trial_indices, :, :]
print(latentDataSelect.shape)

# Run the latent model
pcaOut, clfOut = latent_models(latentDataSelect, target, 11)

def add_cluster(pca_model, clf_model, odor_latent):
	"""
	Within each time window, all the points/trials from all the rats are assigned to different odor clusters.
	"""
	n_tpts = 10
	count_matrix = np.zeros((n_tpts, 4))
	proba_matrix = np.zeros((n_tpts, 4))
	for i in range(n_tpts):
		test = clf_model.predict(pca_model.transform(odor_latent[:, i + 8, :]))
		proba = clf_model.predict_proba(pca_model.transform(odor_latent[:, i + 8, :]))

		# Plot the decoding output as a random scatter with ___ in "latent space"
		# Size is a function of decoding probability
		cluster_x = np.random.normal(0, 0.1, np.sum(test == 0))
		cluster_y = np.random.normal(0, 0.1, np.sum(test == 0))
		plt.scatter(cluster_x + i, cluster_y, s=50 * np.max(proba[test == 0], axis=1), alpha=0.5, c='deepskyblue')
		cluster_x = np.random.normal(0, 0.1, np.sum(test == 1))
		cluster_y = np.random.normal(0, 0.1, np.sum(test == 1))
		plt.scatter(cluster_x + i, cluster_y + 1, s=50 * np.max(proba[test == 1], axis=1), alpha=0.5, c='tan')
		cluster_x = np.random.normal(0, 0.1, np.sum(test == 2))
		cluster_y = np.random.normal(0, 0.1, np.sum(test == 2))
		plt.scatter(cluster_x + i, cluster_y + 2, s=50 * np.max(proba[test == 2], axis=1), alpha=0.5, c='mediumseagreen')
		cluster_x = np.random.normal(0, 0.1, np.sum(test == 3))
		cluster_y = np.random.normal(0, 0.1, np.sum(test == 3))
		plt.scatter(cluster_x + i, cluster_y + 3, s=50 * np.max(proba[test == 3], axis=1), alpha=0.5, c='purple')
		
		for j in range(4):
			count_matrix[i, j] = np.sum(test == j)
			proba_matrix[i, j] = np.sum(proba[:, j])
	return count_matrix, proba_matrix


def decode_trials(pca_model, clf_model, odor_latent):
	'''
		This bad boy looks at the latent output from the cnn model 
		use LR on PCA reduced dims to try and figure out what the hpc was thinking
		returns trls X time window X odor 
	'''
	trl, win, dim = odor_latent.shape 
	#print('Data Shape:', trl, win, dim)
	nIter = 10
	decodeMat = np.empty((trl, nIter, 4)) # trls, timepoints, decodings
	decodeMat[:] = np.nan 
	for winItr in range(0,nIter):
		for trlItr in range(0,trl):
			# Reshape the data to fit a single sample. Need to confirm this is the best approach... 
			test = clf_model.predict(pca_model.transform(odor_latent[trlItr, winItr + 8, :].reshape(-1, 1) ))
			proba = clf_model.predict_proba(pca_model.transform(odor_latent[trlItr, winItr + 8, :].reshape(-1, 1) ))


			for odorIter in range(4):
				decodeMat[trlItr, winItr, odorIter] = np.sum(test == odorIter)
	return decodeMat


# Select which odor to decode
if 1:
	odorTarg = 1
	odor_latent = latentDataSelect[target == odorTarg, :, :] # (41, 18, 10)
	print('odor_latent', odor_latent.shape)
	print('pTarg:', target[target==1].shape)
	#print(target.shape)

	print('decodeing trials...')
	decodeMat = decode_trials(pcaOut, clfOut, odor_latent)
	print('dm Shape:', decodeMat.shape) # (41, 10, 4)


if 1:
	# plot the decoding for each timepoint
	ntrl, nTpts, nOdors = decodeMat.shape
	fig = plt.figure(figsize=(10, 6))
	timeBins = ['-.3s', '-.1s', '.1s', '.3s', '.5s', '.7s', '.9s', '1.1s', '1.3', '1.5s']
	for i in range(0, nTpts):
		ax = plt.subplot(1, nTpts, i+1)
		plt.imshow(decodeMat[:, i, :])
		plt.title(timeBins[i])
		if i == 0:
			plt.xlabel('Odor')
			plt.ylabel('Trial')
		else:
			plt.yticks([])
		my_xticks = ['A','B','C','D']
		plt.xticks([0,1,2,3], my_xticks)
	t = ratName + ' odor = ' + str(odorTarg) + ' Decoding'
	fig.suptitle(t, fontsize="x-large")

	fName = ratName + '_trlDecode-' + str(odorTarg)
	sName = os.path.join(figPath, 'trlDecoding', fName)
	plt.savefig(sName, dpi=500)

	plt.show()



print()

def org_targ(target):
	'''
	Organize target (a list of odors 1-4) such that its easier to vizualize
	'''
	odors = set(target)
	targMat = np.zeros((len(target), len(odors))) # trls, timepoints, decodings
	for trl in range(len(target)):
		targMat[trl,target[trl]] = 1
	return targMat




if 0: # compare one timepoint to the trial
	odor_latent = latentDataSelect # (41, 18, 10)
	print()
	print('odor_latent', odor_latent.shape)
	#print(target.shape)

	print('decodeing trials...')
	decodeMat = decode_trials(pcaOut, clfOut, odor_latent)
	print('dm Shape:', decodeMat.shape) # (41, 10, 4)

	tm = org_targ(target)

	# plot the decoding for each timepoint
	ntrl, nTpts, nOdors = decodeMat.shape
	nPoint = 6
	fig = plt.figure(figsize=(10, 6))
	ax = plt.subplot(1, 2, 2)
	plt.imshow(decodeMat[:, nPoint, :])
	plt.title(str(nPoint))

	ax = plt.subplot(1, 2, 1)
	plt.imshow(tm)
	plt.title('Trls')

	t = ratName + ' odor = ' + str(nPoint)
	fig.suptitle(t, fontsize="x-large")

	plt.show()

if 0: # Can we decode ALL trials...? 
	odor_latent = latentDataSelect # (41, 18, 10)
	print()
	print('odor_latent', odor_latent.shape)
	#print(target.shape)

	print('decodeing trials...')
	decodeMat = decode_trials(pcaOut, clfOut, odor_latent)
	print('dm Shape:', decodeMat.shape) # (41, 10, 4)

	# plot the decoding for each timepoint
	ntrl, nTpts, nOdors = decodeMat.shape
	fig = plt.figure(figsize=(10, 6))
	for i in range(0, nTpts):
		ax = plt.subplot(1, nTpts, i+1)
		plt.imshow(decodeMat[:, i, :])
		plt.title(str(i))
	t = ratName + ' odor = all'
	fig.suptitle(t, fontsize="x-large")

	plt.show()


if 0: # Can we perdict trial based on decoding? 
	# trls X time window X odor 
	odor_latent = latentDataSelect
	decodeMat = decode_trials(pcaOut, clfOut, odor_latent)
	tWin = 3
	# As a simple first step, just grab the max indacy
	#maxDecode = np.argmax(decodeMat[:, tWin, :], axis=1)
	# given the weight of B, this won't work.. let's try variance? 
	varList = [statistics.variance(i) for i in decodeMat[:, tWin, :]]
	#plt.plot(varList)
	#plt.show()

	

	# Lets see if it checks out with the current one... 
	if 0:
		# Import the poke durations from the full trial list...
		# Need to be more elegant about this... later... 
		from scipy.io import loadmat
		print('XXXXXXX')
		dataPath = '/Users/k/Desktop/fortin_lab/computedMats/Superchris_extraction_odor2s_PokeIn.mat'
		#dataPath = '/Users/k/Desktop/fortin_lab/data/SuperChris/superchris_extraction_odor2s.mat'
		data_odor = loadmat(dataPath)
		trlfo = data_odor['trialInfo']
		print('ts', trlfo.shape)
		trial_indices1 = filter_trials(trial_info)

		trial_indices = filter_trials(trlfo)
		odorInfo = trlfo[trial_indices, 3] - 1
		oi = trial_info[trial_indices1, 3] - 1

		print(odorInfo.shape)
		print(oi.shape)
		

		# The two lists seem similar... But there's some weirdness
		# THIS IS A REALLY HACKY SOLUTION!!!
		#odorInfo = odorInfo[2:]
		#oi = oi[5:]
		print(len(odorInfo), len(oi))


		# check for the bad inds... [8, 136]?
		bad_inds = []
		c = 0
		for i in range(len(oi)):
			if i + c == len(oi):
				break
			if int(odorInfo[i]) != oi[i+c]:
				bad_inds.append(i)
				c += 1
		print('List fix:', c)
		goodInds = [i for i in range(len(oi)-len(bad_inds)) if i not in bad_inds]
		
		

		# grab the hold times... 
		holdTimes = trlfo[trial_indices, 6]
		print('hold:, good:')
		print(len(holdTimes), len(goodInds))
		holdTimes = holdTimes[goodInds]
		print(holdTimes)
		print(len(varList), len(holdTimes))
		print(len(trial_info[trial_indices1, 3]))

		if 1: #Check the lists side by side... 
			tifo = trial_info[trial_indices1, 3] - 1
			tifo = tifo[5:]
			oi = oi[goodInds]
			oi = oi[2:]
			for i,j in zip(oi, tifo):
				print(i,j)

		varList = varList[5:]


		plt.scatter(varList[1:80], holdTimes[1:80])
		plt.show()


	if 0:
		#fName = '/Users/k/Desktop/fortin_lab/computedMats/superChris_all_trial_info.npy'
		print('XXXXXXX')
		trlfo = np.load('/Users/k/Desktop/fortin_lab/computedMats/Superchris_PokeIn_trial_info.npy')
		#trlfo = np.load(fName)
		print('trlfo', trlfo.shape)
		print(trlfo)
		print(trlfo[:,6])
		trial_indices = filter_trials(trlfo)

		ofo = trlfo[trial_indices, 6] - 1
		print(ofo.shape)
		print(ofo)


if 0: # Plot aggregate trial visualizations... 
	count, proba = add_cluster(pcaOut, clfOut, odor_latent)

	if 0: # plot latent space cluser representations
		# test plot
		fig = plt.figure(figsize=(8, 4), dpi=500)
		plt.plot([1.5, 1.5], [-1, 4], '--', c='gray', alpha=0.5)
		plt.plot([7.5, 7.5], [-1, 4], '--', c='gray', alpha=0.5)
		plt.ylim(-0.5, 3.5)
		plt.xticks([1.5, 7.5], ['0s', '1.2s'])
		plt.yticks([0, 1, 2, 3], ['A', 'B', 'C', 'D'])

		plt.xlabel('Time')
		plt.ylabel('Odor')
		plt.title('Odor B Aggregate')
		plt.gca().invert_yaxis()

		fig.tight_layout()
		sName = os.path.join(figPath, 'odor_b_decode.png')
		plt.savefig(sName, dpi=500)

	# count the dots? 
	count_odor_B = count
	#print(count_odor_B / np.sum(count_odor_B[0]))

	plt.figure(figsize=(8, 2))
	prop_odor_B = count_odor_B / np.sum(count_odor_B[0])
	plt.plot(prop_odor_B[:, 0], color='deepskyblue')
	plt.plot(prop_odor_B[:, 1], color='tan')
	plt.plot(prop_odor_B[:, 2], color='mediumseagreen')
	plt.plot(prop_odor_B[:, 3], color='purple')
	plt.xticks([1.5, 7.5], ['0s', '1.2s'])
	plt.show()





print('fin\n')
