import os, contextlib
import torch
import shutil
import numpy as np
from inspect import getfullargspec

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def get_reward_all(label_all, num_all):
	num_all = [0] + num_all
	cumsum_all = np.cumsum(num_all)
	reward_all = []
	for dist_ind in range(1, len(cumsum_all)):
		start_ind = cumsum_all[dist_ind-1]
		end_ind = cumsum_all[dist_ind]
		reward = np.mean(label_all[start_ind:end_ind])
		reward_all += [reward]
	return reward_all


def get_label_from_mu(mu_all, mu_list):
	"""
	Assume the value in mu_list is ascending. Assume mu=one is failure
	"""
	mu_list.insert(-1, 1.0) # add one to end; end means lowest reward
	num_label_level = len(mu_list)
	label_level_list = np.linspace(1.0, 0.0, num_label_level)

	# Convert list to dic with key as mu and value as label
	label_dic = {}
	for mu, label in zip(mu_list, label_level_list):
		label_dic[mu] = label

	# Assign label based on mu
	label_all = []
	for mu in mu_all:
		label_all += [label_dic[mu]]
	return label_all

def swap_halves(x):
	a, b = x.split(x.shape[0]//2)
	return torch.cat([b, a])

# torch.lerp only support scalar weight
def lerp(start, end, weights):
	return start + weights * (end - start)

###########################################################################333

def ensure_directory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def ensure_directory_hard(directory):
	if os.path.exists(directory):
		shutil.rmtree(directory)
	os.mkdir(directory)

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper


def save__init__args(values, underscore=False, overwrite=False, subclass_only=False):
    """
    Use in `__init__()` only; assign all args/kwargs to instance attributes.
    To maintain precedence of args provided to subclasses, call this in the
    subclass before `super().__init__()` if `save__init__args()` also appears
    in base class, or use `overwrite=True`.  With `subclass_only==True`, only
    args/kwargs listed in current subclass apply.
    """
    prefix = "_" if underscore else ""
    self = values['self']
    args = list()
    Classes = type(self).mro()
    if subclass_only:
        Classes = Classes[:1]
    for Cls in Classes:  # class inheritances
        if '__init__' in vars(Cls):
            args += getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])


def weighted_sample_without_replacement(population, weights, k=1):
    from random import choices
    weights = list(weights)
    positions = range(len(population))
    indices = []
    while True:
        needed = k - len(indices)
        if not needed:
            break
        for i in choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0.0
                indices.append(i)
    return [population[i] for i in indices], indices
