import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


def load_coords_from_file(filename, hand=1, data_path='data/labels'):
	path = os.path.join(data_path, filename)
	data = pd.read_csv(path, header=None, names=['x1', 'y1', 'x2', 'y2'])
	coord_cols = ['{}{}'.format(n, hand) for n in ['x', 'y']]
	coords = data [coord_cols].values 
	return coords 

def create_accumulator_matrix(coords, block_size=None, m_size=(500, 500)):
	'''Applies 2-d coordinates to build accumulator matrix. Optional to provide block size.'''
	if block_size is None:
		block_size = coords.shape[0]

	# format coordinates for use as indices
	adj_coords = np.round(coords[:block_size, :] * m_size[0]).astype(np.int)
	# create empty matrix
	acc_matrix = np.zeros(m_size)
	# update values in adjusted coordinates
	acc_matrix[adj_coords[:, 1], adj_coords[:, 0]] = 1
	return acc_matrix 

def show_acc_matrix(acc_matrix, fig_size=(8, 8)):
	plt.figure(figsize=fig_size)
	plt.imshow(acc_matrix, 'gray')
	plt.show()


def create_block_windows(num_coords, block_size):
	num_windows = num_coords // block_size
	r = num_coords % block_size
	# create lower bound for idx
	lwr_bnd = np.arange(num_windows) * block_size
	# create upper bound for idx
	upr_bnd = (np.arange(num_windows) + 1) * block_size
	# add remainder as reduced window
	if r > 0:
		lwr_bnd = np.append(lwr_bnd, upr_bnd[-1])
		upr_bnd = np.append(upr_bnd, num_coords)
	return np.vstack((lwr_bnd, upr_bnd)).T

def create_multiple_acc_matrices(coords, blocks):
	acc_matrices = []
	for i in range(blocks.shape[0]):
		block = blocks[i, :]
		coords_block = coords[block[0]:block[1], :]
		acc_matrix = create_accumulator_matrix(coords_block)
		acc_matrices.append(acc_matrix)
	return acc_matrices