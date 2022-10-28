from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import torch
a = np.random.rand(1, 512)
b = np.random.rand(512, 10000)
b = b.T

c = cosine_distances(a, b)
print(c.shape)