#!/usr/local/bin/python
import numpy as np
import random

filename="156Project/votes.csv"
array=np.genfromtxt(filename,dtype="int",delimiter=",")

population_size=array.shape[0]
population_indices=np.arange(population_size)
training_indices=random.sample(population_indices,int(population_size*0.8))
testing_indices=list(set(population_indices)-set(training_indices))

training_data=array[training_indices,:]
testing_data=array[testing_indices,:]
np.savetxt("testing_data.csv",testing_data,delimiter=",",fmt="%d")
np.savetxt("training_data.csv",training_data,delimiter=",",fmt="%d")





