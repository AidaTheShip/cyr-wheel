import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt
import scipy.spatial.transform as scirot
import os

trials = ['01']

def extract_rotation_matrices(rotation_array):
        t = rotation_array.shape[1]  # Number of time steps
        rotation_matrices = [rotation_array[:, i].reshape(3, 3) for i in range(t)]
        return rotation_matrices

def rotation_matrix_to_euler_angles(rotations, t):
        R = np.array([
        [rotations[0, t], rotations[1, t], rotations[2, t]],
        [rotations[3, t], rotations[4, t], rotations[5, t]],
        [rotations[6, t], rotations[7, t], rotations[8, t]]
    ])
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # roll (sagittal plane)
            y = np.arctan2(-R[2, 0], sy)      # pitch (frontal plane)
            z = np.arctan2(R[1, 0], R[0, 0])  # yaw (transverse plane)
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.degrees(x), np.degrees(y), np.degrees(z)  # Return in degrees








# Wheel center of mass coordinates
def COM_coordinates(v, rot, com):
    array = []
    for t in range(np.shape(v)[1]):
        array.append(np.dot(v[:,t]- com[:,t]/1000, np.linalg.inv(rot[:,:,t])))
    return np.array(array).T


# fig, ax = plt.subplots(1, 4, figsize = (12,4)) 


# ax[0].plot(wheel_COM[0] / 1000, wheel_COM[1] / 1000)
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
# ax[0].set_title('Wheel COM')

# ax[1].plot(cp[0] / 1000, cp[1] / 1000)
# ax[1].set_xlabel('x')
# ax[1].set_ylabel('y')
# ax[1].set_title('Wheel contact point')



# # print(data)
# #print(len(data))
# #print("Shape", data.shape)
# ax[2].plot(body_data.T[0],body_data.T[1] )
# ax[2].set_xlabel('x')
# ax[2].set_ylabel('y')
# ax[2].set_title('Body COM')
# relative_motion = COM_coordinates(body_data.T, rot_mat, wheel_COM)
# ax[3].plot(relative_motion[0], relative_motion[1])
# ax[3].set_xlabel('x')
# ax[3].set_ylabel('y')
# ax[3].set_title('Body COM - wheel COM')

# plt.tight_layout()
# plt.savefig(f"Trial_plots/Trial{trial}/motion.png")
# plt.clf()




# # motion in plane of wheel
# fig, ax = plt.subplots(3, 2, figsize = (9,9)) 

# time = np.linspace(1000/200, 4400 /200, len(relative_motion[1][1000:4400]) )
# ax[1,0].plot(time, relative_motion[1][1000:4400])
# ax[1,0].set_xlabel("t")
# ax[1,0].set_ylabel("Distance")
# ax[1,0].set_title("Normal to wheel plane")

# # Perform the FFT
# fs = 200
# signal = relative_motion[1][1000:4400] - np.mean(relative_motion[1][1000:4400])
# fft_result = np.fft.fft(signal)  # FFT of the signal
# frequencies = np.fft.fftfreq(len(signal), 1 / fs)  # Corresponding frequencies

# # Compute the magnitude spectrum
# magnitude = np.abs(fft_result)
# ax[1,1].plot(frequencies[:fs // 2], magnitude[:fs // 2])
# ax[1,1].set_title("Magnitude Spectrum")
# ax[1,1].set_xlabel("Frequency")
# ax[1,1].set_xlim(0,4)
# ax[1,1].set_ylabel("Amplitude")
# ax[1,1].axvline(0.647, linestyle = '--')


# ax[0,0].plot(time, relative_motion[0][1000:4400])
# ax[0,0].set_xlabel("t")
# ax[0,0].set_ylabel("Distance")
# ax[0,0].set_title("Horizontal in wheel plane")
# # Perform the FFT
# fs = 200
# signal = relative_motion[0][1000:4400] - np.mean(relative_motion[0][1000:4400])
# fft_result = np.fft.fft(signal)  # FFT of the signal
# frequencies = np.fft.fftfreq(len(signal), 1 / fs)  # Corresponding frequencies

# # Compute the magnitude spectrum
# magnitude = np.abs(fft_result)
# ax[0,1].plot(frequencies[:fs // 2], magnitude[:fs // 2])
# ax[0,1].set_title("Magnitude Spectrum")
# ax[0,1].set_xlabel("Frequency ")
# ax[0,1].set_xlim(0,4)
# ax[0,1].set_ylabel("Amplitude")
# ax[0,1].axvline(0.647, linestyle = '--')


# ax[2,0].plot(time, relative_motion[2][1000:4400])
# ax[2,0].set_xlabel("t")
# ax[2,0].set_ylabel("Distance")
# ax[2,0].set_title("Vertical in wheel plane")
# # Perform the FFT
# fs = 1000
# signal = relative_motion[2][1000:4400] - np.mean(relative_motion[2][1000:4400])
# fft_result = np.fft.fft(relative_motion[2])  # FFT of the signal
# frequencies = np.fft.fftfreq(len(relative_motion[2][1000:4400]), 1 / fs)  # Corresponding frequencies

# # Compute the magnitude spectrum
# magnitude = np.abs(fft_result)
# ax[2,1].plot(frequencies[:fs // 2], magnitude[:fs // 2])
# ax[2,1].set_title("Magnitude Spectrum")
# ax[2,1].set_xlabel("Frequency")

# ax[2,1].set_ylabel("Amplitude")
# ax[2,1].axvline(0.647, linestyle = '--', label = "wheel rot freq")
# ax[2,1].set_xlim(0,50)
# ax[2,1].legend()
# plt.tight_layout()
# plt.savefig("Trial_plots/Trial01/relative_motion_fft.png")






# 

'''Angles'''
def Angles(trial,rot_mat, start_frame, end_frame):
    fps = 200
    R = scirot.Rotation.from_matrix(rot_mat)
    euler = R.as_euler("zxy")
    euler_cut = euler[start_frame:end_frame,:]
    fig, ax = plt.subplots(2,3, figsize = (12,4))
    time = np.linspace(int(start_frame / fps),np.shape(euler_cut)[0]/fps,np.shape(euler_cut)[0] )
   
    for i in range(3):
        ax[0,i].plot(time,(360/ (2*np.pi))*euler_cut[:,i])
        ax[0 ,i].set_title(['Spin (ψ)','Tilt(θ)','Roll(ϕ)'][i])
        centered  =euler_cut[:,i] - np.mean(euler_cut[:,i])
        fft_result = np.fft.fft(centered)  # FFT of the principal component
        N = len(centered)  # Number of samples
        frequencies = np.fft.fftfreq(N, 1 / fps)  # Frequency in Hz (1/seconds)
        magnitude = np.abs(fft_result)
        half_N = N // 2  # Only take the positive frequencies
        
        
        max_index= np.argmax(magnitude[:half_N])
        max_freq = frequencies[max_index]

 
        ax[1, i].plot(frequencies[:half_N], magnitude[:half_N]) 
        ax[1, i].axvline(max_freq, label = str(np.round(max_freq,2))) 
        ax[1, i].set_xlabel("Frequency (Hz)")
        ax[1, i].set_ylabel("Magnitude")
        
        ax[1 ,i].set_xlim(0,4)
        ax[1 ,i].legend()
    
    plt.tight_layout()
    directory = f"/Users/adampearl/Documents/_Soft_Math/cyr_wheel_local/Trial_plots/Trial{trial}"
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + "/wheel_angles.png")


    plt.show()
def Anglesplot():
    fps = 200
    # waltz_trials = {'01':[5,22], '03':[4,21], '04':[10,24],'06':[3,12], '07': [5,17], '09': [9,24], '19': [1,12] } # Trial number and start/stop times
    waltz_trials = {'01':[5,22], '07': [5,17], '09': [9,24], '19': [1,12], '24' :[1,8]} # Trial number and start/stop times
    for trial in waltz_trials:
        wheel_data = sci.loadmat(f'data/Contact_point/Trial00{trial}.mat')
        wheel_keys = list(wheel_data.keys())
        rot_data = wheel_data[wheel_keys[-1]]
        rotations = rot_data[0][0][-1][0][0]['Rotations'][0]
        rot_mat = np.array(extract_rotation_matrices(rotations))

        start_frame = np.round( waltz_trials[trial][0] * fps )
        end_frame = np.round( waltz_trials[trial][1] * fps )
        Angles(trial,rot_mat, start_frame, end_frame)


# fig, ax = plt.subplots(2, 3, figsize = (10,4)) 
# for i in range(3):
#     ax[0,i].plot(time, wheel_COM[i][1000:4400] /1000)
    
#     meanCOM = wheel_COM[i][1000:4400]/1000 - np.mean(wheel_COM[i][1000:4400]/1000)
#     fft_result = np.fft.fft(meanCOM)  # FFT of the principal component
#     N = len(meanCOM)  # Number of samples
#     frequencies = np.fft.fftfreq(N, 1 / fs)  # Frequency in Hz (1/seconds)
#     magnitude = np.abs(fft_result)
    
#     half_N = N // 2  # Only take the positive frequencies
#     ax[1,i].plot(frequencies[:half_N], magnitude[:half_N])  
#     ax[1,i].set_xlabel("Frequency (Hz)")
#     ax[1,i].set_ylabel("Magnitude")
#     ax[1,i].set_title(f"FFT")
#     ax[1,i].set_xlim(0,4)
#     max_index = np.argmax(magnitude[:half_N])
#     max_freq = frequencies[:half_N][max_index]
#     print(f"Max Freq: {max_freq}")
# plt.savefig('Trial_plots/Trial01/wheel_COM.png')

# #Plot all segments
# fig, ax = plt.subplots(12,3)
# time = np.linspace(5,22,3400)
# for i in range(3):
#     for j in range(12):
#         data = segment_data[keys[j]][0][0]
#         segment_COM = COM_coordinates(data, rot_mat, wheel_COM).T
#         ax[i,j].plot(time, segment_COM[i][1000:4400])

# plt.show()



'''

Check Daoyuan Ansatz

'''
def Ansatz(save = False, show = False):


    # parameters
    Radius = 1
    g = 9.81
    m = 6
    omega_s = 0.88 * 2*np.pi
    omega_r = 0.735312 * 2*np.pi
    scaled_omega_s = omega_s * np.sqrt(Radius/g)
    scaled_omega_r = omega_r * np.sqrt(Radius/g)
    B_c = -1.7
    scaled_B_c = B_c / Radius
    epsilon = 0.1

    B_theta = ((-2 * m * (1 + scaled_omega_s**2)) / ((2 * m + 3) * (1 + scaled_omega_s**2) - 1)) * scaled_B_c
    A_psi = 0
    A_theta = - (2 * m * (1 + scaled_omega_r**2 - scaled_omega_s**2)) / ((2 * m + 3) * (1 + scaled_omega_r**2 - scaled_omega_s**2) - 1)
    A_phi = (2 * m * scaled_omega_s) / ((2 + m) * scaled_omega_r * ((2 * m + 3) * (1 + scaled_omega_r**2 - scaled_omega_s**2) - 1))
    Delta_psi = 0  # Irrelevant
    Delta_theta = 0
    Delta_phi = np.pi / 2
    Delta_match = 0.2 #phase shift to match data


    print(A_phi)
    print(A_theta)
    #Equations 
    def c(t):
        return scaled_B_c*epsilon + epsilon*np.sin(omega_r * t)


    def psi(t):
        expr = omega_s * t + A_psi * epsilon * np.sin(omega_r * t + Delta_psi)
        return (expr+np.pi) % (2*np.pi) - np.pi

    def theta(t):
        return  B_theta * epsilon + A_theta * epsilon * np.sin(omega_r * t + Delta_theta)  #+ np.pi / 2 #(removed to fit data, added negative sign)

    def phi(t):
        return A_phi * epsilon * np.sin(omega_r * t + Delta_phi)


    # fps = 200
    # start_frame = 1000
    # end_frame = 4400
    # total_time = (end_frame - start_frame) / fps
    # time_theory = np.linspace(0, total_time, 1000)

    


    # radtodeg = (360/ (2*np.pi))
    # R = scirot.Rotation.from_matrix(rot_mat)
    # euler = R.as_euler("zxy")

    # euler_cut = euler[start_frame:end_frame,:]
    # time_data = np.linspace(0,total_time, np.shape(euler_cut)[0] )

    # fig, ax = plt.subplots(1,4, figsize = (16,4))

    # ax[0].plot(time_data,radtodeg*euler_cut[:,0], color = 'red', alpha = 0.7)
    # ax[0].plot(time_theory,radtodeg*psi(time_theory), color = 'green', alpha = 0.7)
    # ax[0].set_title('Spin (psi)')

    # ax[1].plot(time_data,radtodeg*euler_cut[:,1], color = 'red', alpha = 0.7)
    # ax[1].plot(time_theory,radtodeg*theta(time_theory), color = 'green', alpha = 0.7)
    # ax[1].set_title('Tilt (theta)')   

    # ax[2].plot(time_data,radtodeg*euler_cut[:,2], color = 'red', alpha = 0.7)
    # ax[2].plot(time_theory,radtodeg*phi(time_theory), color = 'green', alpha = 0.7)
    # ax[2].set_title('Rock (phi)') 

    # ax[3].plot(time_theory,c(time_theory), color = 'green', alpha = 0.7)
    # ax[3].set_title('c(t)') 




    # plt.show()

    # plt.clf()
    trials = {'1': [0.88, 0.65],'7': [0.92,0.5], '9':[0.8,0.4], '19':[0.55,0.64], '24':[0.57,0.43]}
    uncertainties = {'1': [0.055,0.05], '7':[0.08,0.13], '9':[0.15,0.1], '19':[0.055,0.21], '24':[0.34,0.25]} #ws, wr
    ''' Plot Atheta '''
   

    
    radius = 1
    g = 9.81
    scale = 2*np.pi *np.sqrt(radius/g) #multiply to remove dimensions

    def A_thetaf(dw2):
        raddeg = 360/(2*np.pi)
        dw2scale = dw2 * scale**2
        return - raddeg*(2 * m * (1 + dw2scale)) / ((2 * m + 3) * (1 + dw2scale) - 1)
    def A_phif(ws,wr):
        raddeg = 360/(2*np.pi)
        scaled_omega_s = ws * 2*np.pi *np.sqrt(radius/g)
        scaled_omega_r = wr * 2*np.pi *np.sqrt(radius/g)
        A_phi = raddeg*(2 * m * scaled_omega_s) / ((2 + m) * scaled_omega_r * ((2 * m + 3) * (1 + scaled_omega_r**2 - scaled_omega_s**2) - 1))
        return A_phi
    # for trial in trials:
    #     directory = f'daoyuan/Trial{trial}'
    #     os.makedirs(directory, exist_ok=True)

    #     #Atheta
    #     dw2_range = np.linspace(-1, 1, 1000)
    #     plt.xlabel(r'$w_r^2 - w_s^2 / 4\pi^2$')
    #     plt.ylabel('Amplitude (degs)')
    #     plt.title(r'$A_\theta$')
    #     plt.ylim(-100,100)
    #     plt.plot(dw2_range, A_thetaf(dw2_range))
        
    #     point = trials[trial]
    #     dw2 = point[1]**2 - point[0]**2 #wr^2 - ws^2
    #     dw2one = uncertainties[trial][1]**2 - uncertainties[trial][0]**2 + 2*(point[1]*uncertainties[trial][1] - point[0]*uncertainties[trial][0])
    #     dw2two = uncertainties[trial][1]**2 - uncertainties[trial][0]**2 - 2*(point[1]*uncertainties[trial][1] - point[0]*uncertainties[trial][0])
    #     minerror = np.abs(min([dw2one,dw2two]))
    #     maxerror = np.abs(max([dw2one,dw2two]))
    #     print(minerror)
    #     plt.errorbar([dw2], [A_thetaf(dw2)], xerr=([[minerror], [maxerror]]), fmt='o', capsize = 3)
    #     if save:
    #         plt.savefig(directory+"/Atheta")
    #     if show:
    #         plt.show()
    #     plt.clf()

    #     #Aphi
    #     wr_range = np.linspace(-1, 1, 1000)
    #     plt.xlabel(r'$w_r / 2\pi$')
    #     plt.ylabel('Amplitude (degs)')
    #     plt.title(r'$A_\phi$')
    #     plt.plot(wr_range, A_phif(point[0], wr_range))
    #     plt.errorbar(point[1], A_phif(point[0],point[1]), xerr = uncertainties[trial][1], fmt='o', capsize = 3)
    #     plt.xlim(0, 1)
    #     plt.ylim(-50,50)
    #     if save:
    #         plt.savefig(directory + "/Aphi")
    #     if show:
    #         plt.show()
    #     plt.clf()

    def wrf(ws):
        radius = 1
        g = 9.81
        scale = 2*np.pi *np.sqrt(radius/g) #[T]
        
        scaledws = ws * scale
        #return np.sqrt( (-2*(1+m)/(-3+2*m)) +scaledws**2 ) / scale #dimensionful
        return np.sqrt( (-2*(1+m)/(3+2*m)) +scaledws**2 ) / scale #dimensionful
    def stability(ws,eps):
        radius = 1
        g = 9.81
        scale = 2*np.pi *np.sqrt(radius/g) #[T]
        ws2 = ws*scale
        
        num = m*(eps*(380*ws2**2 - 380) -400*ws2**3 +31.0974*ws2**2 +400*ws2 -31.0974)-600*ws2**3+46.6461*ws2**2+400*ws2-31.0974
        denom = m*(380*eps-400*ws2+31.0974)-600*ws2+46.6461
        result = np.sqrt(num/denom)
        return result / scale
    
    '''All trials together'''

    # Singularity
    ws_range = np.linspace(0.4, 1, 1000)
    
    markers = {'1': 'o', '7':'s', '9':'o', '19':'s', '24':'o'}
    plt.xlabel(r'$w_s/ 2\pi$')
    plt.ylabel(r'$w_r/ 2\pi$')
    plt.title('Singularity')
    plt.plot(ws_range, wrf(ws_range))
    for trial in trials:
        ws = trials[trial][0]
        wr = trials[trial][1]
        plt.errorbar(ws, wr, xerr = uncertainties[trial][0],yerr = uncertainties[trial][1], fmt = markers[trial], capsize=5, label = "Trial " + trial)
    #Try adding stability contour
    eps = 0.1
    result = stability(ws_range, eps)
    plt.plot(ws_range,result, color='red', linestyle = "--", label = r"$\epsilon$ = " + str(eps))  # Shade above the line

    plt.legend()
    if save:
        plt.savefig("daoyuan/singularity")
    if show:
        plt.show()
    plt.clf()
    # Atheta

    dw2_range = np.linspace(-1, 1, 1000)
    plt.xlabel(r'$w_r^2 - w_s^2 / 4\pi^2$')
    plt.ylabel('Amplitude (degs)')
    plt.title(r'$A_\theta$')
    plt.ylim(-100,100)
    plt.plot(dw2_range, A_thetaf(dw2_range))
    for trial in trials:
        point = trials[trial]
        dw2 = point[1]**2 - point[0]**2 #wr^2 - ws^2
        dw2one = uncertainties[trial][1]**2 - uncertainties[trial][0]**2 + 2*(point[1]*uncertainties[trial][1] - point[0]*uncertainties[trial][0])
        dw2two = uncertainties[trial][1]**2 - uncertainties[trial][0]**2 - 2*(point[1]*uncertainties[trial][1] - point[0]*uncertainties[trial][0])
        minerror = np.abs(min([dw2one,dw2two]))
        maxerror = np.abs(max([dw2one,dw2two]))
        print(minerror)
        plt.errorbar([dw2], [A_thetaf(dw2)], xerr=([[minerror], [maxerror]]), fmt=markers[trial], capsize = 3, label = "Trial " + trial)
    plt.legend()
    if save:
        plt.savefig("daoyuan/Atheta")
    if show:
        plt.show()
    plt.clf()

    #Aphi
    wr_range = np.linspace(0, 1.0, 5000)
    plt.xlabel(r'$w_r / 2\pi$')
    plt.ylabel('Amplitude (degs)')
    plt.title(r'$A_\phi$')
    plt.plot(wr_range, A_phif(point[0], wr_range))
    for trial in trials:
        point = trials[trial]
        plt.errorbar(point[1], A_phif(point[0],point[1]), xerr = uncertainties[trial][1], fmt=markers[trial], capsize = 3,label = "Trial " + trial, alpha = 0.7)
    


    plt.ylim(-30,30)
    plt.legend()
    if save:
        plt.savefig("daoyuan/Aphi")
    if show:
        plt.show()
    plt.clf()



''' Processing (Copy from here)'''

# wheel_data = sci.loadmat(f'data/Contact_point/Trial00{trial}.mat')
# segment_data = sci.loadmat('./data/Kinematics + EMG/Trial 1 Segment CoM.mat')
# wheel_keys = list(wheel_data.keys())
# rot_data = wheel_data[wheel_keys[-1]]
# cp_data = sci.loadmat(f'data/Contact_point/Trial00{trial}_ContactPoint.mat')
# mat_data = sci.loadmat(f'./data/Trial1 Kinematics.mat')#this is hard coded because we dont have data for 24
# keys = list(segment_data.keys())[6:]

# center_of_mass_key = keys[-1]

# # rot_data

# wheel_COM = wheel_data[f'Trial00{trial}'][0][0][5][0][0]['Positions'][0]
# cp = cp_data['contact_points']

# # body_data = mat_data[center_of_mass_key][0][0]
# rotations = rot_data[0][0][-1][0][0]['Rotations'][0]
# rot_mat = np.array(extract_rotation_matrices(rotations))

''' Execution '''



Ansatz(save=True)