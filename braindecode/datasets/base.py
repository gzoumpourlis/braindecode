"""
Dataset classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import os
import json
import warnings
from glob import glob

import pandas as pd

from torch.utils.data import Dataset, ConcatDataset


def _create_description(description):
	if description is not None:
		if (not isinstance(description, pd.Series)
				and not isinstance(description, dict)):
			raise ValueError(f"'{description}' has to be either a "
							 f"pandas.Series or a dict.")
		if isinstance(description, dict):
			description = pd.Series(description)
	return description


class BaseDataset(Dataset):
	"""Returns samples from an mne.io.Raw object along with a target.

	Dataset which serves samples from an mne.io.Raw object along with a target.
	The target is unique for the dataset, and is obtained through the
	`description` attribute.

	Parameters
	----------
	raw: mne.io.Raw
		Continuous data.
	description: dict | pandas.Series | None
		Holds additional description about the continuous signal / subject.
	target_name: str | None
		Name of the index in `description` that should be used to provide the
		target (e.g., to be used in a prediction task later on).
	transform : callable | None
		On-the-fly transform applied to the example before it is returned.
	"""
	def __init__(self, raw, description=None, target_name=None,
				 transform=None):
		self.raw = raw
		self.description = _create_description(description)
		self.transform = transform

		# save target name for load/save later
		self.target_name = target_name
		if target_name is None:
			self.target = None
		elif target_name in self.description:
			self.target = self.description[target_name]
		else:
			raise ValueError(f"'{target_name}' not in description.")

	def __getitem__(self, index):
		X, y = self.raw[:, index][0], self.target
		if self.transform is not None:
			X = self.transform(X)
		return X, y

	def __len__(self):
		return len(self.raw)

	@property
	def transform(self):
		return self._transform

	@transform.setter
	def transform(self, value):
		if value is not None and not callable(value):
			raise ValueError('Transform needs to be a callable.')
		self._transform = value


class WindowsDataset(BaseDataset):
	"""Returns windows from an mne.Epochs object along with a target.

	Dataset which serves windows from an mne.Epochs object along with their
	target and additional information. The `metadata` attribute of the Epochs
	object must contain a column called `target`, which will be used to return
	the target that corresponds to a window. Additional columns
	`i_window_in_trial`, `i_start_in_trial`, `i_stop_in_trial` are also
	required to serve information about the windowing (e.g., useful for cropped
	training).
	See `braindecode.datautil.windowers` to directly create a `WindowsDataset`
	from a `BaseDataset` object.

	Parameters
	----------
	windows : mne.Epochs
		Windows obtained through the application of a windower to a BaseDataset
		(see `braindecode.datautil.windowers`).
	description : dict | pandas.Series | None
		Holds additional info about the windows.
	transform : callable | None
		On-the-fly transform applied to a window before it is returned.
	"""
	def __init__(self, windows, description=None, transform=None):
		self.windows = windows
		self.description = _create_description(description)
		self.transform = transform

		self.y = self.windows.metadata.loc[:, 'target'].to_numpy()
		self.crop_inds = self.windows.metadata.loc[
			:, ['i_window_in_trial', 'i_start_in_trial',
				'i_stop_in_trial']].to_numpy()
		self.sample_weight_list = list()

	def __getitem__(self, index):
		X = self.windows.get_data(item=index)[0].astype('float32')
		if self.transform is not None:
			X = self.transform(X)
		y = self.y[index]
		# necessary to cast as list to get list of three tensors from batch,
		# otherwise get single 2d-tensor...
		crop_inds = self.crop_inds[index].tolist()
		######################
		self.sample_weight_list.append( self.trialwise_weights[index] )
		######################
		return X, y, crop_inds

	def __len__(self):
		return len(self.windows.events)

	@property
	def transform(self):
		return self._transform

	@transform.setter
	def transform(self, value):
		if value is not None and not callable(value):
			raise ValueError('Transform needs to be a callable.')
		self._transform = value

	def compute_covariances(self):
		import numpy as np
		from pyriemann.utils.mean import mean_riemann
		from pyriemann.utils.distance import distance_riemann
		data = self.windows.get_data()
		if len(data.shape)==3:
			trialwise_covs_list = list()
			for i_window in range(data.shape[0]):
				trial_cov = np.cov(data[i_window])
				trialwise_covs_list.append(trial_cov)
			trialwise_covs = np.array(trialwise_covs_list)
		else:
			raise ValueError('Shape of windows.get_data() is not 3D')

		global_cov = mean_riemann(trialwise_covs, tol=1e-08, maxiter=50, init=None, sample_weight=None)

		unique_targets = np.unique(self.y)
		classwise_covs = dict()
		for target in unique_targets:
			indices_target = np.where(self.y==target)[0]
			target_covs_list = list()
			for index in indices_target:
				target_covs_list.append(trialwise_covs[index])
			target_covs = np.array(target_covs_list)
			# https://github.com/pyRiemann/pyRiemann/issues/65
			# if you can not fix the root cause of the problem, then add regularization :
			# cov = Covariances('oas').fit_transform(X_train)
			riemann_mean_cov = mean_riemann(target_covs, tol=1e-08, maxiter=50, init=None, sample_weight=None)
			classwise_covs[target] = riemann_mean_cov
		self.trialwise_covs = trialwise_covs
		self.classwise_covs = classwise_covs
		self.global_cov = global_cov

		distances = np.zeros(data.shape[0])
		for i_window in range(data.shape[0]):
			trial_cov = trialwise_covs[i_window]
			distances[i_window] = distance_riemann(trial_cov, global_cov)
			# covar_target = classwise_covs[self.y[i_window]]
			# distances[i_window] = distance_riemann(covar_trial, covar_target)
		# print('Min: {:.3f} | Max: {:.3f}'.format(np.min(distances), np.max(distances)))

		alpha = 1.0
		beta = 0.5

		# self.trialwise_weights = np.ones(data.shape[0])

		distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 0.00001)
		# beta==0 | range = [0, alpha]
		# beta!=0 | range = [beta, alpha+beta]
		self.trialwise_weights = beta + alpha*(1 - distances)

class BaseConcatDataset(ConcatDataset):
	"""A base class for concatenated datasets. Holds either mne.Raw or
	mne.Epoch in self.datasets and has a pandas DataFrame with additional
	description.

	Parameters
	----------
	list_of_ds: list
		list of BaseDataset, BaseConcatDataset or WindowsDataset
	"""
	def __init__(self, list_of_ds):
		# if we get a list of BaseConcatDataset, get all the individual datasets
		if list_of_ds and isinstance(list_of_ds[0], BaseConcatDataset):
			list_of_ds = [d for ds in list_of_ds for d in ds.datasets]
		super().__init__(list_of_ds)
		self.description = pd.DataFrame([ds.description for ds in list_of_ds])
		self.description.reset_index(inplace=True, drop=True)
		self.sample_weight_list = list()

	def split(self, by=None, property=None, split_ids=None):
		"""Split the dataset based on information listed in its description
		DataFrame or based on indices.

		Parameters
		----------
		by: str | list(int) | list(list(int))
			If by is a string, splitting is performed based on the description
			DataFrame column with this name.
			If by is a (list of) list of integers, the position in the first
			list corresponds to the split id and the integers to the
			datapoints of that split.
		property: str
			Some property which is listed in info DataFrame.
		split_ids: list(int) | list(list(int))
			List of indices to be combined in a subset.

		Returns
		-------
		splits: dict{str: BaseConcatDataset}
			A dictionary with the name of the split (a string) as key and the
			dataset as value.
		"""
		args_not_none = [
			by is not None, property is not None, split_ids is not None]
		if sum(args_not_none) != 1:
			raise ValueError("Splitting requires exactly one argument.")

		if property is not None or split_ids is not None:
			warnings.warn("Keyword arguments `property` and `split_ids` "
						  "are deprecated and will be removed in the future. "
						  "Use `by` instead.")
			by = property if property is not None else split_ids
		if isinstance(by, str):
			split_ids = {
				k: list(v)
				for k, v in self.description.groupby(by).groups.items()
			}
		else:
			# assume list(int)
			if not isinstance(by[0], list):
				by = [by]
			# assume list(list(int))
			split_ids = {split_i: split for split_i, split in enumerate(by)}

		return {str(split_name): BaseConcatDataset(
			[self.datasets[ds_ind] for ds_ind in ds_inds])
			for split_name, ds_inds in split_ids.items()}

	def get_metadata(self):
		"""Concatenate the metadata and description of the wrapped Epochs.

		Returns
		-------
		pd.DataFrame:
			DataFrame containing as many rows as there are windows in the
			BaseConcatDataset, with the metadata and description information
			for each window.
		"""
		if not all([isinstance(ds, WindowsDataset) for ds in self.datasets]):
			raise TypeError('Metadata dataframe can only be computed when all '
							'datasets are WindowsDataset.')

		all_dfs = list()
		for ds in self.datasets:
			df = ds.windows.metadata
			for k, v in ds.description.items():
				df[k] = v
			all_dfs.append(df)

		return pd.concat(all_dfs)

	@property
	def transform(self):
		return [ds.transform for ds in self.datasets]

	@transform.setter
	def transform(self, value):
		for i in range(len(self.datasets)):
			self.datasets[i].transform = value

	def save(self, path, overwrite=False):
		"""Save dataset to files.

		Parameters
		----------
		path: str
			Directory to which .fif / -epo.fif and .json files are stored.
		overwrite: bool
			Whether to delete old files (.json, .fif, -epo.fif) in specified directory 
			prior to saving.
		"""
		assert len(self.datasets) > 0, "Expect at least one dataset"
		assert (hasattr(self.datasets[0], 'raw') + hasattr(
			self.datasets[0], 'windows') == 1), (
			"dataset should have either raw or windows attribute")
		files = ['target_name.json', 'description.json']
		file_names_ = ["{}-raw.fif", "{}-epo.fif"]
		description_file_name = os.path.join(path, 'description.json')
		target_file_name = os.path.join(path, 'target_name.json')
		if not overwrite:
			if (os.path.exists(description_file_name) or 
				os.path.exists(target_file_name)):
				raise FileExistsError(
					f'{description_file_name} or {target_file_name} exist in {path}.')
		else:
			for file_name in file_names_:
				file_names = glob(os.path.join(path, f"*{file_name.lstrip('{}')}"))
				_ = [os.remove(f) for f in file_names]
			if os.path.isfile(target_file_name):
				os.remove(target_file_name)
			if os.path.isfile(description_file_name):
				os.remove(description_file_name)

		concat_of_raws = hasattr(self.datasets[0], 'raw')
		file_name = file_names_[0] if concat_of_raws else file_names_[1]
		if concat_of_raws:
			# for checks that all have same target name and for
			# saving later
			target_name = self.datasets[0].target_name
		for i_ds, ds in enumerate(self.datasets):
			full_file_path = os.path.join(path, file_name.format(i_ds))
			if concat_of_raws:
				ds.raw.save(full_file_path, overwrite=overwrite)
				assert ds.target_name == target_name, "All datasets should have same target name"
			else:
				ds.windows.save(full_file_path, overwrite=overwrite)

		if concat_of_raws:
			json.dump({'target_name': target_name}, open(target_file_name, 'w'))
		self.description.to_json(description_file_name)

	def compute_covariances_concat(self):
		"""Compute covariances (only for WindowsDataset) of the wrapped Epochs.

		Returns
		-------
		Returns nothing, but computes self.covariances:
			Numpy array containing covariances for each window.
		"""
		if not all([isinstance(ds, WindowsDataset) for ds in self.datasets]):
			raise TypeError('Metadata dataframe can only be computed when all '
							'datasets are WindowsDataset.')
		
		for ds in self.datasets:
			ds.compute_covariances()