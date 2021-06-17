from braindecode.datasets.moabb import MOABBDataset

from braindecode.datautil.preprocess import (exponential_moving_standardize, preprocess, Preprocessor)

import numpy as np
from braindecode.datautil.windowers import create_windows_from_events

import torch
from braindecode.util import set_random_seeds
from braindecode.models import * #ShallowFBCSPNet, EEGNetv1

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

import mne
mne.set_log_level(False)
moabb.set_log_level(False)

######################################################################

def train(subject_id):

	subject_range = [subject_id]
	##### subject_range = [x for x in range(1, 10)]

	dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_range)


	######################################################################
	# Preprocessing

	low_cut_hz = 4.  # low cut frequency for filtering
	high_cut_hz = 38.  # high cut frequency for filtering
	# Parameters for exponential moving standardization
	factor_new = 1e-3
	init_block_size = 1000

	preprocessors = [
		Preprocessor('pick_types', eeg=True, eog=False, meg=False, stim=False),  # Keep EEG sensors
		Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
		Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
		Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'),
		Preprocessor('resample', sfreq=125),

		## Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
		## factor_new=factor_new, init_block_size=init_block_size)
		## Preprocessor('pick_channels', ch_names=short_ch_names, ordered=True),
	]

	# Transform the data
	preprocess(dataset, preprocessors)


	######################################################################
	# Cut Compute Windows
	# ~~~~~~~~~~~~~~~~~~~

	trial_start_offset_seconds = -0.0
	# Extract sampling frequency, check that they are same in all datasets
	sfreq = dataset.datasets[0].raw.info['sfreq']
	assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
	# Calculate the trial start offset in samples.
	trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

	# Create windows using braindecode function for this. It needs parameters to define how
	# trials should be used.
	windows_dataset = create_windows_from_events(
		dataset,
		# picks=["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"],
		trial_start_offset_samples=trial_start_offset_samples,
		trial_stop_offset_samples=0,
		preload=True,
	)

	######################################################################
	# Split dataset into train and valid

	splitted = windows_dataset.split('session')
	train_set = splitted['session_T']
	valid_set = splitted['session_E']

	######################################################################
	# Create model

	cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
	device = 'cuda' if cuda else 'cpu'
	if cuda:
		torch.backends.cudnn.benchmark = True
	seed = 20200220  # random seed to make results reproducible
	# Set random seed to be able to reproduce results
	set_random_seeds(seed=seed, cuda=cuda)

	n_classes = 4
	# Extract number of chans and time steps from dataset
	n_chans = train_set[0][0].shape[0]
	input_window_samples = train_set[0][0].shape[1]

	
	model = ShallowFBCSPNet(
		n_chans,
		n_classes,
		input_window_samples=input_window_samples,
		final_conv_length='auto')
	

	"""
	model = EEGNetv1(
			n_chans,
			n_classes,
			input_window_samples=input_window_samples,
			final_conv_length="auto",
			pool_mode="mean",
			second_kernel_size=(2, 32),
			third_kernel_size=(8, 4),
			drop_prob=0.25)
	"""

	"""
	model = HybridNet(n_chans, n_classes,
					input_window_samples=input_window_samples)
	"""

	"""
	model = TCN(n_chans, n_classes,
				n_blocks=6,
				n_filters=32,
				kernel_size=9,
				drop_prob=0.0,
				add_log_softmax=True)
	"""

	"""
	model = EEGNetv4(n_chans,
					n_classes,
					input_window_samples=input_window_samples,
					final_conv_length="auto",
					pool_mode="mean",
					F1=8,
					D=2,
					F2=16,  # usually set to F1*D (?)
					kernel_length=64,
					third_kernel_size=(8, 4),
					drop_prob=0.2)
	"""

	if cuda:
		model.cuda()


	######################################################################
	# Training

	# These values we found good for shallow network:
	lr = 0.01 # 0.0625 * 0.01
	weight_decay = 0.0005

	# For deep4 they should be:
	# lr = 1 * 0.01
	# weight_decay = 0.5 * 0.001

	batch_size = 64
	n_epochs = 80

	clf = EEGClassifier(
		model,
		criterion=torch.nn.NLLLoss,
		optimizer=torch.optim.SGD, #AdamW,
		train_split=predefined_split(valid_set),  # using valid_set for validation
		optimizer__lr=lr,
		optimizer__momentum=0.9,
		optimizer__weight_decay=weight_decay,
		batch_size=batch_size,
		callbacks=[
			"accuracy", #("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
		],
		device=device,
	)
	# Model training for a specified number of epochs. `y` is None as it is already supplied
	# in the dataset.
	clf.fit(train_set, y=None, epochs=n_epochs)

	results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
	df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
					  index=clf.history[:, 'epoch'])

	val_accs = df['valid_accuracy'].values
	max_val_acc = 100.0 * np.max(val_accs)

	return max_val_acc

if __name__=='__main__':

	accs = []
	for subject_id in range(1,10):
		acc = train(subject_id)
		accs.append(acc)
	accs = np.array(accs)

	print('\n\nValidation accuracy: {:.2f} +- {:.2f}\n\n'.format(np.mean(accs), np.std(accs)))




















######################################################################
# Plot Results

"""
# Extract loss and accuracy values for plotting from history object
results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
				  index=clf.history[:, 'epoch'])

# get percent of misclass for better visual comparison to loss
df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
			   valid_misclass=100 - 100 * df.valid_accuracy)

plt.style.use('seaborn')
fig, ax1 = plt.subplots(figsize=(8, 3))
df.loc[:, ['train_loss', 'valid_loss']].plot(
	ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

df.loc[:, ['train_misclass', 'valid_misclass']].plot(
	ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
ax1.set_xlabel("Epoch", fontsize=14)

# where some data has already been plotted to ax
handles = []
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
plt.tight_layout()

plt.ioff()
plt.show()
"""