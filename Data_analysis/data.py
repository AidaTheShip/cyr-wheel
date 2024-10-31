import scipy.io as sci

mat_data = sci.loadmat('Trial22_Kinematics.mat')

# Printing the keys in our .mat file. 
# print(mat_data.keys())  
keys = list(mat_data.keys())
print(keys)
for key in keys: 
    print(key)

# Accessing specific data entries
print(mat_data[keys[-1]])

# # Printing the data raw. 
# print(mat_data)

