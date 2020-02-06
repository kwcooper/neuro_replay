
# Decode rat odors via CNN 
# LL & KWC 200131

# Data is processed hpc 2 second NPY arrays

import os
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Concatenate, Convolution2D, AveragePooling2D, convolutional
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import np_utils

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.data_utils import *
from utils.tetrode import *
from utils.visualize import *
from utils.helper import *

# Disable the TF warning re: AVX/FMA, 
# though should revisit for potential performance upgrades...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

viz = 0   # display the plots
ratNames = ['Barat', 'Buchanan', 'Mitt', 'Stella', 'SuperChris']
ratName = ratNames[4]

# Build the paths and load the data
dataPath = os.path.join('..', 'data')
spkPath = os.path.join(dataPath, ratName, ratName.lower() + '_spike_data_binned.npy')
lfpPath = os.path.join(dataPath, ratName, ratName.lower() + '_lfp_data_sampled.npy')
trlPath = os.path.join(dataPath, ratName, ratName.lower() + '_trial_info.npy')

print('Loading and cleaning data for network...')
spike_data_binned = np.load(spkPath)
lfp_data_sampled = np.load(lfpPath)
lfp_data_sampled = np.swapaxes(lfp_data_sampled, 1, 2)
trial_info = np.load(trlPath)

#print(spike_data_binned.shape) # SC (245, 46, 400) # Checks out
#print(lfp_data_sampled.shape)  # SC (245, 21, 400)

# Grab and select trial info
select = filter_trials(trial_info)

decoding_start = 210
decoding_end = decoding_start + 25

# Select the trials and the time segments of our decoding data
decoding_data_lfp = lfp_data_sampled[select, :, decoding_start:decoding_end]
decoding_data_spike = spike_data_binned[select, :, decoding_start:decoding_end]
decoding_data_spike = (decoding_data_spike - np.mean(decoding_data_spike)) / np.std(decoding_data_spike) # z - score

decoding_target = np_utils.to_categorical((trial_info[select, 3] - 1).astype(int))


# Index tetrode id's and units for the analysis
tetrode_ids, tetrode_units = fetch_tet(ratName.lower())


# debugging...
print('decoding_data_spike: ', decoding_data_spike.shape) # (168, 46, 25)
print('spike_data_binned: ', spike_data_binned.shape)     # (245, 46, 400)




def valid_tetrodes(tetrode_ids, tetrode_units):
	"""
	Only keep valid tetrodes with neuron units so that there is corresponding spike train data.
	:param tetrode_ids: (list) of tetrode ids in the order of LFP data
	:param tetrode_units: (dict) number of neuron units on each tetrode
	:return: (list) of tetrode ids with neuron units
	"""
	return [x for x in tetrode_ids if tetrode_units[x] > 0]

def build_tetrode_model(tetrode_ids, tetrode_units):
	"""
	Build tetrode convolutional neural network model for odor decoding.
	:param tetrode_ids: (list) of tetrode ids in the order of LFP data
	:param tetrode_units: (dict) number of neuron units on each tetrode
	:return: (keras) compiled decoding model
	"""
	input_tetrodes = valid_tetrodes(tetrode_ids, tetrode_units)

	input_layers = []
	for t in input_tetrodes:
		k = tetrode_units[t]
		input_layers.append(Input(shape=(k + 1, 25, 1)))

	convolution_layers = []
	for i, input_layer in enumerate(input_layers):
		t = input_tetrodes[i]
		k = tetrode_units[t]
		# Need to update call
		
		#convolution_layers.append(Convolution2D(5, k + 1, 1, activation='relu')(input_layer))
		#Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
		#Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

		convolution_layers.append(convolutional.Conv2D(5, (k + 1, 1), activation='relu')(input_layer))


	combo = Concatenate(axis=-1)(convolution_layers)
	pooling = AveragePooling2D(pool_size=(1, 25))(combo)

	x = Flatten()(pooling)
	x = Dense(10, activation='relu')(x)
	x = Dropout(rate=0.1)(x)

	prediction = Dense(4, activation='softmax')(x)

	model = Model(inputs=input_layers, outputs=prediction)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	return model

def organize_tetrode(spike_data, lfp_data, tetrode_ids, tetrode_units, verbose=True):
	"""
	Organize spike and LFP data by tetrode.
	:param spike_data: (3d numpy array) spike train data of format [trial, neuron, time]
	:param lfp_data: (3d numpy array ) LFP data of format [trial, tetrode, time]
	:param tetrode_ids: (list) of tetrode ids in the order of LFP data
	:param tetrode_units: (dict) number of neuron units on each tetrode
	:param verbose: (bool) whether to print each tetrode data shape
	:return: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
	"""
	all_tetrode_data = []
	i = 0

	for j, t in enumerate(tetrode_ids):
		k = tetrode_units[t]
		if k == 0:
			continue

		tetrode_lfp = np.expand_dims(lfp_data[:, j, :], axis=1)
		tetrode_spike = spike_data[:, i:(i + k), :]
		if len(tetrode_spike.shape) == 2:
			tetrode_spike = np.expand_dims(tetrode_spike, axis=1)

		tetrode_data = np.concatenate([tetrode_lfp, tetrode_spike], axis=1)
		tetrode_data = np.expand_dims(tetrode_data, axis=-1)

		all_tetrode_data.append(tetrode_data)

		if verbose:
			print('Current tetrode {t} with {k} neurons/units'.format(t=t, k=k))
			print(tetrode_data.shape)

		i += k
	return all_tetrode_data


def select_data(all_tetrode_data, index):
	"""
	Select tetrode data by trial indices.
	:param all_tetrode_data: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
	:param index: (1d numpy array) trial indices
	:return: (list of 4d numpy arrays) selected subset of tetrode data
	"""
	current_data = []
	for x in all_tetrode_data:
		current_data.append(x[index, :, :, :])
	return current_data


def cross_validate(all_tetrode_data, target, tetrode_ids, tetrode_units, verbose=True):
	"""
	Perform cross-validation with tetrode convolutional neural network model.
	:param all_tetrode_data: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
	:param tetrode_ids: (list) of tetrode ids in the order of LFP data
	:param tetrode_units: (dict) number of neuron units on each tetrode
	:param target: (2d numpy array) classification labels
	:param verbose: (bool) whether to print each validation fold accuracy
	:return: (2d numpy array) true and predicted labels
	"""
	kf = StratifiedKFold(n_splits=5)
	y_true = np.zeros(target.shape)
	y_hat = np.zeros(target.shape)
	i = 0

	for train_index, test_index in kf.split(np.zeros(target.shape[0]), target.argmax(axis=-1)):
		X_train, X_test = select_data(all_tetrode_data, train_index), select_data(all_tetrode_data, test_index)
		y_train, y_test = target[train_index, :], target[test_index, :]

		model = build_tetrode_model(tetrode_ids, tetrode_units)

		saveName = os.path.join('savedModels', 'temp_model.h5')
		checkpointer = ModelCheckpoint(saveName, verbose=0, save_best_only=True)
		hist = model.fit(X_train, y_train, epochs=200, batch_size=20, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=0)
		best_model = load_model(saveName)

		n = y_test.shape[0]
		y_true[i:(i + n), :] = y_test
		y_hat[i:(i + n), :] = best_model.predict(X_test)
		i += n

		if verbose:
			#print(hist.history.keys()) -> 'val_loss', 'val_accuracy', 'loss', 'accuracy'
			accuracy = max(hist.history['val_accuracy'])
			print('Current fold validation accuracy: {acc}'.format(acc=accuracy))

	return y_true, y_hat


# Build the model!
print('Building Model...')
model = build_tetrode_model(tetrode_ids, tetrode_units)
#model.summary()

# Organize LFP and Tetrode data
print('Organizing LFP and Tetrode data...')
tetrode_data = organize_tetrode(decoding_data_spike, decoding_data_lfp, tetrode_ids, tetrode_units, verbose=False)

# Run the model and test the accuracy
print('Running cross validation of network decoding:')
s_time = time.time()
y_true, y_hat = cross_validate(tetrode_data, decoding_target, tetrode_ids, tetrode_units)
print('Took:', time.time()-s_time)

print('Building confusion matrix...')
matrix = confusion_matrix(y_true.argmax(-1), y_hat.argmax(-1))

if viz: 
	fig = plt.figure(figsize=(6, 6))
	plot_confusion_matrix(matrix, classes=['A', 'B', 'C', 'D'], normalize=True,
						  title='Confusion matrix')
	plt.show()


# Extract latent representation of the model, store and save them... 
print('Running model and saving latent fits...')
for i in range(10):
	#'tetrode_data len 14, decoding_target len 168
	model = build_tetrode_model(tetrode_ids, tetrode_units)
	hist = model.fit(tetrode_data, decoding_target, epochs=100, batch_size=20, verbose=0, validation_split=0.1, shuffle=True)

	intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(model.layers[-3].name).output)
		
	all_lower = extract_latent(intermediate_layer_model, spike_data_binned, lfp_data_sampled, tetrode_ids, tetrode_units, 25, 20)
	#all_lower = extract_latent(intermediate_layer_model, decoding_data_spike, decoding_data_lfp, tetrode_ids, tetrode_units, 25, 20)
	
	#print('all_lower', all_lower.shape)

	fName = os.path.join('computeData', '{name}_latent_{index}.npy'.format(name=ratName, index=i))
	np.save(fName, all_lower)


print('fin\n')






