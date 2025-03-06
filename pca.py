import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import scipy.signal as signal
'''

Frame rate: 200 fps
Time: 25.365 secs



'''


trial = '19'
wheel_data = sci.loadmat(f'data/Contact_point/Trial00{trial}.mat')
wheel_keys = list(wheel_data.keys())
#rotations
rot_data = wheel_data[wheel_keys[-1]] #'Trial0001' key
rotations = rot_data[0][0][-1][0][0]['Rotations'][0]
wheel_COM = rot_data[0][0][-1][0]['Positions'][0][0].T / 1000

def extract_rotation_matrices(rotation_array):
    t = rotation_array.shape[1]  # Number of time steps
    rotation_matrices = [rotation_array[:, i].reshape(3, 3) for i in range(t)]
    return rotation_matrices
rot_mat = np.array(extract_rotation_matrices(rotations))

rot_mat = np.transpose(rot_mat, axes=(0, 2, 1))

# wheel_COM = wheel_data['Trial0001'][0][0][5][0][0]['Positions'][0] #(3,t)

segment_data = sci.loadmat(f'./data/Kinematics + EMG/Trial {trial} Segment CoM.mat') #body segments - arm, leg, ect.

# axis = 0 #0 for x, 1, for y, 2 for z
axis_labels = ['horizontal in wheel plane','perpendicular to wheel plane','vertical in wheel plane']
# Printing the keys in our .mat file. 
# print(mat_data.keys())  
keys = list(segment_data.keys())[6:]


time_passed = len(segment_data[keys[-1]][0][0])
time = range(time_passed)

# Wheel center of mass coordinates
def COM_coordinates(v, rot, com):
    array = []
    for t in range(np.shape(v)[0]):
        
        array.append(np.dot(rot[t].T,  v[t]- com[t]))
    return np.array(array)
def standardize(array):
     return (array - array.mean(axis = 0)) / array.std(axis = 0)
def downsample_average(arr, M):
    N = len(arr)
    factor = N // M  # How many elements to average per bin
    return np.array([arr[i*factor:(i+1)*factor].mean() for i in range(M)])
def PCA():
    data_matrix = []
    for axis in range(3):
        for i in range(6,len(keys)):
        
        
            # print(keys[i])
            data = segment_data[keys[i]][0][0]
            # print(data)
            # print(len(data))
            # print("Shape", data.shape)
            data_matrix.append(COM_coordinates(data, rot_mat, wheel_COM).T[axis])

    print(np.shape(data_matrix))

    data_mat = np.array(data_matrix).T

    #PERFORM PCA

    ### Step 1: Standardize the Data along the Features
    standardized_data = (data_mat - data_mat.mean(axis = 0)) / data_mat.std(axis = 0)


    ### Step 2: Calculate the Covariance Matrix
    # use `ddof = 1` if using sample data (default assumption) and use `ddof = 0` if using population data
    covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)


    ### Step 3: Eigendecomposition on the Covariance Matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


    ### Step 4: Sort the Principal Components
    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1] 

    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns


    ### Step 5: Calculate the Explained Variance
    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)


    ### Step 6: Reduce the Data via the Principal Components
    k = 2 # select the number of principal components
    reduced_data = np.matmul(standardized_data, sorted_eigenvectors[:,:k]) # transform the original data


    ### Step 7: Determine the Explained Variance
    total_explained_variance = sum(explained_variance[:k])

    ##labels
    xlabel = [item + 'x' for item in keys[6:]]
    ylabel = [item + 'y' for item in keys[6:]]
    zlabel = [item + 'z' for item in keys[6:]]
    label = xlabel + ylabel +zlabel

    return explained_variance, sorted_eigenvectors, np.array(label)

def pca_svd(data_mat, k=0):
    # Step 1: Standardize the data
    if 0 in np.std(data_mat, axis=0):
        raise ValueError("Division by 0 when standardizing data")
    
    standardized_data = (data_mat - np.mean(data_mat, axis=1)[:,np.newaxis]) #/ np.std(data_mat, axis=1) #mean in time not in sample space
    
 
    if np.isnan(standardized_data).any():
        print("NaN in data")
    if np.isinf(standardized_data).any():
        print("Inf in data")
    # Step 2: Perform SVD
    U, S, Vt = np.linalg.svd(standardized_data, full_matrices=False)

    
    # Step 4: Project the data onto the top k components (using sorted Vt)
    projected = standardized_data.dot(Vt.T)  
    
    n_samples = standardized_data.shape[0]  # Number of rows (samples)
    explained_variance = (S ** 2) / (np.sum(S**2))
    

    S_reduced = np.zeros(np.shape(S))
    S_reduced[k] = S[k]
    reduced = S_reduced[:,np.newaxis]*Vt
    reduced = U.dot(reduced) + np.mean(data_mat, axis=1)[:,np.newaxis]
    #Reduced data

    return projected, explained_variance, Vt, reduced
def PCA2(): #each row now includes x y z

    #First step: Put everything in COM, then join x,y,z for every segment
    num_segs = 12
    data_matrix = []
    for i in range(num_segs):
        
            
            data = segment_data[keys[i]][0][0][::8,:] #slice every 8th time step to reduce compute
            # print(data)
            # print(len(data))
            # print("Shape", data.shape)
            segment_COM = COM_coordinates(data, rot_mat, wheel_COM).T
            combined_segment = segment_COM.flatten(order='C')
            data_matrix.append(combined_segment)
            

    print("Shape of data matrix " + str(np.shape(data_matrix)))

    data_mat = np.array(data_matrix)

    #PERFORM PCA

    ### Step 1: Standardize the Data along the Features
    standardized_data = (data_mat - data_mat.mean(axis = 0)) / data_mat.std(axis = 0)
    print('done')

    ### Step 2: Calculate the Covariance Matrix
    # use `ddof = 1` if using sample data (default assumption) and use `ddof = 0` if using population data
    covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)
    rank = np.linalg.matrix_rank(covariance_matrix)
    print(rank)
    print(np.allclose(covariance_matrix, covariance_matrix.T, atol=1e-10))

    regularization_factor = 1e-6
    covariance_matrix += np.eye(covariance_matrix.shape[0]) * regularization_factor


    ### Step 3: Eigendecomposition on the Covariance Matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    print('eigenvcs')
    print(np.max(np.imag(eigenvectors)))
    print(np.max(np.real(eigenvectors)))


    ### Step 4: Sort the Principal Components
    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1] 
    print('done')

    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
    print('done')


    ### Step 5: Calculate the Explained Variance
    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    print('done')


    ### Step 6: Reduce the Data via the Principal Components
    k = 2 # select the number of principal components
    reduced_data = np.matmul(standardized_data, sorted_eigenvectors[:,:k]) # transform the original data
    print('done')


    ### Step 7: Determine the Explained Variance
    total_explained_variance = sum(explained_variance[:k])
    print('done')

    return reduced_data, explained_variance, sorted_eigenvectors

from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth low-pass filter to the input array.

    Args:
        data (np.ndarray): Input array to filter.
        cutoff (float): Cutoff frequency of the filter (Hz).
        fs (float): Sampling frequency of the data (Hz).
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered array.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)
'''

Plot explained variance and principle eigenvectors

'''
# # fig1, ax1 = plt.subplots(1,3, figsize = (14,10))
# fig2, ax2 = plt.subplots(1,1, figsize = (12,4))


# explained_variance, sorted_eigenvectors, label = PCA()

# ax2.plot(np.cumsum(explained_variance))
# ax2.set_xlabel("PC")
# ax2.set_xticks(np.arange(0, len(explained_variance), 1))

# # plt.show()





# print(len(label), len(sorted_eigenvectors[0]))
# for i in range(3):
#     #sort from high to low
#     sorted_indices = np.argsort(-sorted_eigenvectors[i])

#     # Reorder both arrays
#     sorted_vector = sorted_eigenvectors[i][sorted_indices]
    
#     sorted_labels = label[sorted_indices]

#     ax1[i].scatter(range(len(label)),sorted_vector)
#     ax1[i].set_xticks(ticks=range(len(label)), labels=sorted_labels, rotation = 80)
#     ax1[i].set_title(f'PC {i}  ')

    
    
    # plt.show()

# fig2.suptitle("Explained Variance", fontweight='bold')
# # fig1.tight_layout()
# fig2.tight_layout()
# # fig1.savefig(f'pca/Trial1/PC')
# fig2.savefig(f'pca/Trial1/explained_variance')




'''

FIRST PC

'''

# fig, ax = plt.subplots(1,1, figsize = (12,5))
# explained_variance, sorted_eigenvectors, label = PCA()
# abs_eigenvector = abs(sorted_eigenvectors[0])
# sorted_indices = np.argsort(-abs_eigenvector)

# # Reorder both arrays
# sorted_vector = abs_eigenvector[sorted_indices]

# sorted_labels = label[sorted_indices]

# ax.scatter(range(len(label)),sorted_vector)
# ax.set_xticks(ticks=range(len(label)), labels=sorted_labels, rotation = 90)
# ax.set_title(f'')


# fig.suptitle("First Principle Component", fontweight='bold')
# fig.tight_layout()
# fig.savefig('pca/Trial1/first_PC')

# np.save("sorted_labels1.npy", sorted_labels)
# np.save("sorted_vector1.npy", sorted_vector)

'''

Second PC

'''
# plt.clf()
# fig, ax = plt.subplots(1,1, figsize = (12,5))
# explained_variance, sorted_eigenvectors, label = PCA()

# abs_eigenvector = abs(sorted_eigenvectors[1])
# sorted_indices = np.argsort(-abs_eigenvector)

# # Reorder both arrays
# sorted_vector = abs_eigenvector[sorted_indices]

# sorted_labels = label[sorted_indices]

# ax.scatter(range(len(label)),sorted_vector)
# ax.set_xticks(ticks=range(len(label)), labels=sorted_labels, rotation = 90)
# ax.set_title(f'')


# fig.suptitle("Second Principle Component", fontweight='bold')
# fig.tight_layout()
# fig.savefig('pca/Trial1/second_PC')

# np.save("sorted_labels2.npy", sorted_labels)
# np.save("sorted_vector2.npy", sorted_vector)


'''

Weigh all segments by mass

'''


# body_segment_masses = np.array([ #Body mass estimates in kg from chat GPT 'These masses are for a 45-year-old woman with a mass of 70 kg and approximately 30% body fat'
#     0.4545,  # LeftHand
#     1.2727,  # LeftForearm
#     2.0909,  # LeftUpperArm
#     0.4545,  # RightHand
#     1.2727,  # RightForearm
#     2.0909,  # RightUpperArm
#     10.9091, # LeftThigh
#     3.6364,  # LeftShank
#     1.2727,  # LeftFoot
#     10.9091, # RightThigh
#     3.6364,  # RightShank
#     1.2727   # RightFoot
# ])

# normalized_masses = body_segment_masses / np.sum(body_segment_masses)
# explained_variance, sorted_eigenvectors, label = PCA()
# limb_num = len(body_segment_masses) #number of limbs, should be 12
# reshaped_firstPC = sorted_eigenvectors[0].reshape(-1, limb_num).T   #should be 12x3
# weights = reshaped_firstPC * body_segment_masses[:, np.newaxis]
# print(weights)




'''

Motion PCA

'''
num_segs = 12
data_matrix = []

for i in range(num_segs):
    
        
        data = segment_data[keys[i]][0][0]
        segment_COM = COM_coordinates(data, rot_mat, wheel_COM).T
        data_matrix.append(segment_COM[0][1000:4400]) #Remove getting on and getting off
        data_matrix.append(segment_COM[1][1000:4400])
        data_matrix.append(segment_COM[2][1000:4400])

        #Add EMG data

print("Shape of data matrix " + str(np.shape(data_matrix)))



data_mat = np.array(data_matrix)


projected, explained_variance, Vt, reduced = pca_svd(data_mat)
print(explained_variance[0],explained_variance[1],explained_variance[2])
plt.bar(range(1, len(explained_variance)+1), explained_variance)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.savefig(f'Trial{trial}_analysis/PCA/motion/Explained variance')
plt.clf()


fig, ax = plt.subplots(5,1, figsize = (16,16))
for i in range(5):
    ax[i].plot(np.linspace(1000/200, 4400/200, len(Vt[i])),Vt[i])
    ax[i].set_title(f"Component {i+1}")

plt.savefig(f"Trial{trial}_analysis/PCA/motion/Principal Components")
plt.clf()

#FFT on PCs
fs = 200  # Motion capture sampling rate in Hz (frames per second)

fig, ax = plt.subplots(4, 1, figsize=(16, 16))
for i in range(4):
    pc = Vt[i] - np.mean(Vt[i])
    fft_result = np.fft.fft(pc)  # FFT of the principal component
    N = len(pc)  # Number of samples
    frequencies = np.fft.fftfreq(N, 1 / fs)  # Frequency in Hz (1/seconds)
    magnitude = np.abs(fft_result)
    
    half_N = N // 2  # Only take the positive frequencies
    ax[i].plot(frequencies[:half_N], magnitude[:half_N])  
    ax[i].set_xlabel("Frequency (Hz)")
    ax[i].set_ylabel("Magnitude")
    ax[i].set_title(f"FFT of Principal Component {i+1}")
    ax[i].set_xlim(0,4)
    ax[i].axvline(0.88, linestyle = '--',label = "rotation freq")

ax[-1].legend()
plt.savefig(f"Trial{trial}_analysis/PCA/motion/PC_fft")
plt.clf()

# Compute the magnitude spectrum



#Generate ordered labels
ordered_labels = []
for i in range(12):
    ordered_labels.append(keys[i] + 'x')
    ordered_labels.append(keys[i] + 'y')
    ordered_labels.append(keys[i] + 'z')

# Get sorted indices based on absolute values in descending order
sorted_indices = np.argsort(np.abs(projected[:, 0]))[::-1]

# Reorder both the data and labels based on sorted indices
sorted_data = np.abs(projected[:, 0])[sorted_indices]
sorted_labels = np.array(ordered_labels)[sorted_indices]

plt.barh(sorted_labels, sorted_data)  # Horizontal bar chart
plt.xticks(rotation=0)  # Keep x-ticks horizontal for readability
plt.yticks(rotation=0)  # Ensure labels stay readable
plt.title("Data Projected onto PC 1 ")
plt.tight_layout()  # Adjust layout to fit everything
plt.savefig(f"Trial{trial}_analysis/PCA/motion/PC1_proj")


#Reduced data
print(np.shape(reduced))
plt.clf()
plt.plot(reduced[5], label = "reduced")
plt.plot(data_mat[5], label = 'original')
plt.legend()
plt.show()

# np.save(f'Trial{trial}_analysis/PCA/reduced.npy', reduced)


'''

PCA Artificial test

Test the PCA on artificial data to make sure it is doing what is needs to be doing

'''

# ## Generate Test Data: 2D Gaussian Distribution ##
# mean = [0, 0]  # Mean vector (center of the Gaussian)
# cov = [[4, 3.9],  # Covariance matrix (controls spread and correlation)
#        [3.9, 4]]  

# # Generate N samples
# N = 500
# data_matrix = np.random.multivariate_normal(mean, cov, N)

# print("Shape of data matrix:", np.shape(data_matrix))

# # Apply PCA
# reduced_data, explained_variance, Vt = pca_svd(data_matrix)

# # Compute data center
# center = np.mean(data_matrix, axis=0)

# # Plot results
# fig, ax = plt.subplots(2, 1, figsize=(7, 8))

# # Scatter plot of original data
# ax[0].scatter(data_matrix[:, 0], data_matrix[:, 1], alpha=0.5)
# ax[0].scatter(*center, color="black", label="Mean")

# # Plot PCA vectors
# pc1 = Vt[0]  # First principal component
# pc2 = Vt[1]  # Second principal component

# ax[0].quiver(*center, *pc1, scale=0.5, color="red", label="First PC")
# ax[0].quiver(*center, *pc2, scale=0.5, color="green", label="Second PC")

# ax[0].set_title("Test Data: 2D Gaussian with PCA Vectors")
# ax[0].set_xlabel("X")
# ax[0].set_ylabel("Y")
# ax[0].legend()
# ax[0].axis("equal")

# # Plot the transformed data in PCA space
# ax[1].plot(reduced_data[:, 0], label="First PC", color="red")
# ax[1].plot(reduced_data[:, 1], label="Second PC", color="green")
# ax[1].set_title("Projected Data onto Principal Components")
# ax[1].legend()

# plt.show()







'''

Plot all segments at once

'''

# for i in range(6,len(keys),1):
#     data = segment_data[keys[i]][0][0]
#     plt.plot(data.T[0])#- wheel_COM[0]/1000)
# plt.xlabel("t")
# plt.ylabel("x")
# plt.savefig("pca/all_segments_xaxis.png")
# plt.clf()
# for i in range(6,len(keys),1):
#     data = segment_data[keys[i]][0][0]
#     plt.plot(data.T[1])#- wheel_COM[1]/1000)

# plt.xlabel("t")
# plt.ylabel("y")
# plt.savefig("pca/all_segments_yaxis.png")

''' 

EMG PCA 

'''

# EMG_data = sci.loadmat(f'data/Trial1 EMG.mat')
# data_matrix = []
# time = np.linspace(5, 22, 3400)

# fs = 2200  # Sampling frequency in Hz (adjust based on your data)
# low_cutoff = 400  # Lower cutoff frequency (Hz)
# high_cutoff = 500  # Upper cutoff frequency (Hz)
# order = 4  # Filter order

# #Optional: test filter
# # EMG_test = EMG_data['EMG_1'][0][0].T[0][1000:4400]
# # b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs)
# # filtered_data = signal.filtfilt(b, a, EMG_test)  # Apply the filter

# # fig, ax = plt.subplots(2,1)
# # ax[0].plot(time, EMG_test)
# # ax[1].plot(time, filtered_data)
# # plt.show()

# for i in range(1,17):
#     EMG = EMG_data[f'EMG_{i}'][0][0].T[0][1000:4400]
#     b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs)
#     filtered_data = signal.filtfilt(b, a, EMG)  # Apply the filter
#     data_matrix.append(filtered_data)

# data_mat = np.array(data_matrix)


# reduced_data, explained_variance, Vt = pca_svd(data_mat)
# plt.bar(range(1, len(explained_variance)+1), explained_variance)
# plt.xlabel('Principal Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance by Principal Components')
# plt.savefig('Trial1_analysis/PCA/EMG/Explained variance')



# fig, ax = plt.subplots(5,1, figsize = (16,16))
# for i in range(5):
#     b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs)
#     filtered_data = signal.filtfilt(b, a, Vt[i])  # Apply the filter
#     ax[i].plot(np.linspace(5, 22, len(Vt[i])),Vt[i])
#     ax[i].set_title(f"Component {i+1}")
# plt.savefig("Trial1_analysis/PCA/EMG/Principal Components")

# #FFT on PCs
# fs = 2200  # Motion capture sampling rate in Hz (frames per second)

# fig, ax = plt.subplots(4, 1, figsize=(16, 16))
# for i in range(4):
#     pc = Vt[i] - np.mean(Vt[i])
#     fft_result = np.fft.fft(pc)  # FFT of the principal component
#     N = len(pc)  # Number of samples
#     frequencies = np.fft.fftfreq(N, 1 / fs)  # Frequency in Hz (1/seconds)
#     magnitude = np.abs(fft_result)
    
#     half_N = N // 2  # Only take the positive frequencies
#     ax[i].plot(frequencies[:half_N], magnitude[:half_N])  
#     ax[i].set_xlabel("Frequency (Hz)")
#     ax[i].set_ylabel("Magnitude")
#     ax[i].set_title(f"FFT of Principal Component {i+1}")
#     # ax[i].set_xlim(0,4)
#     ax[i].axvline(0.647, linestyle = '--',label = "rotation freq")

# ax[-1].legend()
# plt.savefig("Trial1_analysis/PCA/EMG/PC_fft")
# plt.clf()

# #Generate ordered labels
# muscle_names = np.array(['Posterior DeltoidL', 'Medial DeltoidL', 'Internal ObliqueL',
# 'External ObliqueL' ,'Gluteus MaximusL' ,'Hip FlexorsL',
#  'Medial GastrocnemiusL' ,'Biceps FemorisL' ,'Posterior DeltoidR',
#  'Medial DeltoidR', 'Internal ObliqueR' ,'External ObliqueR',
#  'Gluteus MaximusR' ,'Hip FlexorsR' ,'Medial GastrocnemiusR',
#  'Biceps FemorisR']) #ASK AIDAAA

# # Get sorted indices based on absolute values in descending order
# sorted_indices = np.argsort(np.abs(reduced_data[:, 0]))[::-1]

# # Reorder both the data and labels based on sorted indices
# sorted_data = np.abs(reduced_data[:, 0])[sorted_indices]
# sorted_labels = np.array(muscle_names)[sorted_indices]

# plt.barh(sorted_labels, sorted_data)  # Horizontal bar chart
# plt.xticks(rotation=0)  # Keep x-ticks horizontal for readability
# plt.yticks(rotation=0)  # Ensure labels stay readable
# plt.title("Data Projected onto PC 1 ")
# plt.tight_layout()  # Adjust layout to fit everything
# plt.savefig("Trial1_analysis/PCA/EMG/PC1_proj")
'''

Combined PCA

'''
# num_segs = 12
# EMG_data = sci.loadmat(f'data/Trial1 EMG.mat')
# motion_mat= []
# fs = 2200  # Sampling frequency in Hz (adjust based on your data)
# low_cutoff = 400  # Lower cutoff frequency (Hz)
# high_cutoff = 500  # Upper cutoff frequency (Hz)
# order = 4  # Filter order


# for i in range(num_segs):

        
#         data = segment_data[keys[i]][0][0]

#         segment_COM = COM_coordinates(data, rot_mat, wheel_COM).T
#         # combined_segment = segment_COM.flatten(order='C')
#         motion_mat.append(segment_COM[0][1000:4400])
#         motion_mat.append(segment_COM[1][1000:4400])
#         motion_mat.append(segment_COM[2][1000:4400])

# #Downsample ratio
# new_length = np.shape(motion_mat)[1]  # Compute new length
# EMG_mat = []
# for i in range(1,17):
#         #Add EMG data
#         EMG = EMG_data[f'EMG_{i}'][0][0].T[0][1000:4400]
#         b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs)
#         filtered_data = signal.filtfilt(b, a, EMG)  # Apply the filter
#         downsampled_signal = downsample_average(filtered_data, new_length)
#         #Downsample
#         EMG_mat.append(downsampled_signal)

# data_matrix = np.vstack((standardize(np.array(motion_mat)), standardize(np.array(EMG_mat))))

# print("Shape of data matrix " + str(np.shape(data_matrix)))



# data_mat = np.array(data_matrix)
# standardized_data = (data_mat - np.mean(data_mat, axis=0)) / np.std(data_mat, axis=0)
# #plot example downsampled EMG
# # plt.plot(standardized_data[-1])
# # plt.show()
# # plt.clf()

# projected, explained_variance, Vt, reduced = pca_svd(data_mat)
# plt.bar(range(1, len(explained_variance)+1), explained_variance)
# plt.xlabel('Principal Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance by Principal Components')
# plt.savefig('Trial1_analysis/PCA/motion&EMG/Explained variance')


# firstPC = Vt[0]
# fig, ax = plt.subplots(5,1, figsize = (16,16))
# for i in range(5):
#     ax[i].plot(np.linspace(0, 5073/200, len(Vt[i])),Vt[i])
#     ax[i].set_title(f"Component {i+1}")
# plt.savefig("Trial1_analysis/PCA/motion&EMG/Principal Components")
# plt.clf()
# #Generate ordered labels
# ordered_labels = []
# muscle_names = np.array(['Posterior DeltoidL', 'Medial DeltoidL', 'Internal ObliqueL',
# 'External ObliqueL' ,'Gluteus MaximusL' ,'Hip FlexorsL',
#  'Medial GastrocnemiusL' ,'Biceps FemorisL' ,'Posterior DeltoidR',
#  'Medial DeltoidR', 'Internal ObliqueR' ,'External ObliqueR',
#  'Gluteus MaximusR' ,'Hip FlexorsR' ,'Medial GastrocnemiusR',
#  'Biceps FemorisR'])
# for i in range(12):
#     ordered_labels.append(keys[i] + 'x')
#     ordered_labels.append(keys[i] + 'y')
#     ordered_labels.append(keys[i] + 'z')
# for i in range(16): #ASK AIDA
#     ordered_labels.append(muscle_names[i])
# # Get sorted indices based on absolute values in descending order
# sorted_indices = np.argsort(np.abs(projected[:, 1]))[::-1]

# # Reorder both the data and labels based on sorted indices
# sorted_data = np.abs(projected[:, 1])[sorted_indices]
# sorted_labels = np.array(ordered_labels)[sorted_indices]

# plt.barh(sorted_labels, sorted_data)  # Horizontal bar chart
# plt.xticks(rotation=0)  # Keep x-ticks horizontal for readability
# plt.yticks(rotation=0)  # Ensure labels stay readable
# plt.title("Data Projected onto PC 2 ")
# plt.tight_layout()  # Adjust layout to fit everything
# plt.savefig('Trial1_analysis/PCA/motion&EMG/PC2_proj')