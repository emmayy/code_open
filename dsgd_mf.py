'''
Code for implemetation of Distributed SGD for Matrix Factorization on Spark
By Zhengyang Ruan
Andrew id: zruan
April 17th, 2015
'''

import os
import sys
from pyspark import SparkContext, SparkConf
import numpy as np
import random

__author__ = "Zhengyang Ruan"

V_matrix = []  # a list of (user_id, movie_id, rating)
W_matrix = []  # global variable for W matrix
H_matrix = []  # global variable for H matrix
parameters = {} # global variable for num_factor, num_user, num_movie, beta_value, lambda_value, etc.


'''
do the main RDD splition and iteration
'''
def partition_v_matrix(inputV_filepath, num_factors, num_iterations, num_workers):
	sc = SparkContext(appName="dsgd_partition")

	if os.path.isfile(inputV_filepath):   # read in V matrix with format of each line "user_id, movie_id, rate"
		txt_source = sc.textFile(inputV_filepath)
		v_rdd = txt_source.map(lambda s: tuple(map(int,s.split(','))))

	# pick out max_user_id and max_movie_id to init the W and H matrix
	[max_user,max_movie] = v_rdd.reduce(lambda a,b: max((max(a[0],b[0]),max(a[1],b[1])),b[:2]))

	# here, use ini_divide to ensure that each W_i * H_j is in range of [0,1], to avoid inf or nan case
	# W_i and H_j are all row matrix for similicity. Will use getT() for the transformation
	# also, use numpy.matrix to do the matrix operation
	ini_divide = pow(1.0/num_factors,0.5)
	W_matrix.append(np.matrix([[random.random()*ini_divide for x in xrange(num_factors)] for y in xrange(max_user)]))
	H_matrix.append(np.matrix([[random.random()*ini_divide for x in xrange(num_factors)] for y in xrange(max_movie)]))
	parameters['max_user'] = max_user
	parameters['Ni'] = max_user
	parameters['max_movie'] = max_movie
	parameters['Nj'] = max_movie



	# calculate the matrix block size for each workers in one iteration
	user_block = parameters['max_user'] / num_workers
	movie_block = parameters['max_movie'] / num_workers
	parameters['user_block'] =  user_block
	parameters['movie_block'] =  movie_block
	parameters['n_iter'] = 0   # the one needed to calculate step size


	# do the iteration
	for iter_i in range(num_iterations):
		# devide the V into blocks in interchangeable diagonal matrix
		key_v_matrix = v_rdd.keyBy(lambda x: x[0]/user_block if (((iter_i+x[0]/user_block)%num_workers) == ((x[1]/movie_block)%num_workers)) else -1)
		
		# just pick out useful matrix with key != -1, where key is the index of the diagonal matrix used to do the partition
		key_v_matrix = key_v_matrix.subtractByKey(sc.parallelize([(-1,(0,0,0))]))

		# calculate n in sigma = ( tao_0 + n )^-beta for the step size
		parameters['n_iter'] += key_v_matrix.count()

		# partite the remaining V_ij matrix and do the mapPartition to calculate the updating w and h martix in parallel
		rdd_v_matrix = key_v_matrix.partitionBy(num_workers)
		re_w_h_matrix = rdd_v_matrix.mapPartitions(sgd_mf_seq).collect()  # a list of (0/1, {id:matrix} ) , 0-w_matrix, 1-h_matrix

		# update W and H matrix
		for item in re_w_h_matrix:
			if item[0] == 0:  # update W_matrix, 0 is the key value for new W_matrix
				w_new = item[1]
				for w_item in w_new:
					W_matrix[0][w_item] = w_new[w_item]

			elif item[0] == 1: # update H_matrix, 1 is the key value for new H_matrix
				h_new = item[1]
				for h_item in h_new:
					H_matrix[0][h_item] = h_new[h_item]



'''
do the sgd calculation for range of W and H
@input iteration of
(user_id, movie_id, rate) 
'''
def sgd_mf_seq(iteration):

	# init the W and H matrix, and I use dictionary to store them
	w_matrix = {}  
	h_matrix = {}

	number_vij = 0  # used to calculate n for the step size
	for vij in iteration:  # vij in format of (key,(user_id, movie_id, rate))
		number_vij += 1
		vij = vij[1]
		usr_id = int(vij[0])-1
		movie_id = int(vij[1])-1
		rate = int(vij[2])

		# if (user_id, movie_id) not in existing updating w or h matrix, just create in using the old ones
		if usr_id not in w_matrix:
			w_matrix[usr_id] = W_matrix[0][usr_id].copy()
		if movie_id not in h_matrix:
			h_matrix[movie_id] = H_matrix[0][movie_id].copy()

		# do the updating algorithm
		w_matrix[usr_id] += 2*( pow((parameters['n_iter']+number_vij),-1*parameters['beta_value'])* ( (rate - float(np.dot(W_matrix[0][usr_id],H_matrix[0][movie_id].getT())))*H_matrix[0][movie_id] - parameters['lambda_value']* W_matrix[0][usr_id]/parameters['Ni']) )
		h_matrix[movie_id] += 2*( pow((parameters['n_iter']+number_vij),-1*parameters['beta_value'])* ( (rate - float(np.dot(W_matrix[0][usr_id],H_matrix[0][movie_id].getT())))*W_matrix[0][usr_id] - parameters['lambda_value']* H_matrix[0][movie_id]/parameters['Nj']) )

	# return w and h martix, and use 0/1 key to distinguish them
	return [(0,w_matrix),(1,h_matrix)]



'''
output w_matrix and h_matrix into the out file
@output csv file with comma separated
'''
def out_put_w_h_matrix(outputW_filepath, outputH_filepath):

	# here, for just use getA() to get the array of the matrix, and use join() to convert each row into string for the output.
	out_w_file = open(outputW_filepath,'w')
	w_matrix = W_matrix[0].getA()
	for wi in range(parameters['max_user']):
		out_w_file.write(','.join(map(str,w_matrix[wi]))+'\n')
	out_w_file.close()

	# first use getT() to convert H_j from row into column matrix, and then same way as W_matrix to output into csv file
	out_h_file = open(outputH_filepath,'w')
	h_matrix = H_matrix[0].getT().getA()
	for hi in range(parameters['num_factors']):
		out_h_file.write(','.join(map(str,h_matrix[hi]))+'\n')
	out_h_file.close()

	

'''
main function, read in parameters, call specific functions
@input: spark-submit dsgd_mf.py num_factors num_workers num_iterations beta lambda V.csv w.csv h.csv
'''
if __name__ == '__main__':
	args = sys.argv
	if len(args) != 9:  # the input should match the default one
		print 'wrong number of parameters'
		sys.exit()

	# read in parameters
	num_factors = int(args[1])
	parameters['num_factors'] = num_factors

	num_workers = int(args[2])
	num_iterations = int(args[3])
	beta_value = float(args[4])
	parameters['beta_value'] = beta_value

	lambda_value = float(args[5])
	parameters['lambda_value'] = lambda_value

	inputV_filepath = args[6]
	outputW_filepath = args[7]
	outputH_filepath = args[8]

	# Do the DSGD calculation 
	partition_v_matrix(inputV_filepath, num_factors, num_iterations, num_workers)
	# Output W and H matrix into csv files
	out_put_w_h_matrix(outputW_filepath, outputH_filepath)
	

	






