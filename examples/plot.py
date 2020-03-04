import matplotlib.pyplot as plt
import numpy as np
import pickle

results_path = './results/'
plot_filename = results_path + 'reward_plot.png'

reward_function = []
for i in range(22):
    if i > 3:
        reward_function.append(1/(i-3))

plt.plot([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],reward_function)
plt.plot([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],np.negative(reward_function))
plt.plot(21,0,'o',markersize=2)
plt.xlabel('Züge')
plt.ylabel('Reward')
plt.xticks(np.arange(4, 22, step=1))
plt.show()
plt.savefig(plot_filename)
average_reward = reward_function
pickle.dump( average_reward, open( results_path+"save.p", "wb" ) )
average_reward = pickle.load( open( results_path+"save.p", "rb" ) )