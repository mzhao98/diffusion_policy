import pickle
import numpy as np


# open likelihoods.pkl file
with open('likelihoods.pkl', 'rb') as f:
    likelihoods = pickle.load(f)

# print the likelihoods
print(likelihoods)
likelihoods_as_list = list(likelihoods.values())

# turn the log likelihoods into likelihoods
likelihoods_as_list = np.exp(likelihoods_as_list)

# print the likelihoods
print(likelihoods_as_list)

# normalize
likelihoods_as_list = likelihoods_as_list / np.sum(likelihoods_as_list)


# print the normalized likelihoods
print(likelihoods_as_list)


