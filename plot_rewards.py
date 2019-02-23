import numpy as np
import matplotlib.pyplot as plt

directory = '../final_year_thesis'
filename = '/rewards_udem1_stacked.npy'
scores = np.load(directory + filename)

plot = True
if plot == True:

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
