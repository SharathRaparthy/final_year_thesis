import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("left.txt", "rb") as fo:
    left = pickle.load(fo)
with open("right.txt", "rb") as fo:
    right = pickle.load(fo)
left = [-1 for i in left]
right = [1 for i in right]
plt.hist(right,bins=1)
plt.show()
