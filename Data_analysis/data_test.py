import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt

trial_number = 22
tracker = "Center Of Mass"
mat_data = sci.loadmat('./data/Trial22 Kinematics.mat')

# Printing the keys in our .mat file. 
# print(mat_data.keys())  
keys = list(mat_data.keys())
print(keys)
for key in keys: 
    print(key)

center_of_mass_key = keys[-1]

data = mat_data[center_of_mass_key][0][0]
# print(data)
print(len(data))
print("Shape", data.shape)
time_passed = len(data)
time = range(time_passed)

# print(time)

plt.figure(1)
plt.style.use('seaborn')
plt.plot(time, data)
plt.xlabel("Time")
plt.ylabel("Center Of Mass")
plt.title(f"Trial {trial_number} {tracker}")
plt.legend(['x', 'y', 'z'])
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Trial {trial_number} {tracker}")
plt.show()

# # # Accessing Center Of Mass Data
# print(mat_data[keys[-1]])


# # # # Printing the data raw. 
# # # print(mat_data)

