import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot
import math

mat_contents = sio.loadmat('Proj1rundata_v6.mat')
my_struct = mat_contents['newData1']
my_data = my_struct[0,0]
robotarium_data = my_data[0]
robotarium_data = np.array(robotarium_data)
robotarium_data = np.transpose(robotarium_data)

#print(robotarium_data.shape)  # (1938, 5)

# for i in robotarium_data:
#     print(i)

target2 = (0.5, 0)

distances_to_target2 = [] #np.linalg.norm(( robotarium_data[:, 0], robotarium_data[:, 1]) - target2)

for row in robotarium_data:
    distance = math.sqrt((row[0] - target2[0])**2 + (row[1] - target2[1])**2)
    distances_to_target2.append(distance)

plt = matplotlib.pyplot
#plt.plot(robotarium_data[:, 0], robotarium_data[:, 1])

# all tasks
#plt.plot(distances_to_target2)

# task 3
plt.plot(distances_to_target2[370:800])

# task 5
plt.plot(distances_to_target2[1390:1900])

plt.grid(True)
#plt.xticks(np.arange(0, 2000, 100.0) )
#plt.yticks(np.arange(0, 1, .1))
plt.ylabel('Distance to target2')
plt.xlabel('iterations')
plt.legend(('Step 3', 'Step 5'))
plt.show()


