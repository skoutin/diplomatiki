"""this text has methods for manipulating the .mat files and more generally the LFP siganals of the .mat files"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal as sn
# import sys
# sys.path.insert(0, 'D:\Files\peirama_dipl\project_code')
import statsmodels.api as sm
from scipy.stats import shapiro


def sdirs (name:str):
    """This function contains the directories of all the .mat LFP signal files named accordingly"""
    pc_path = 'D:/Files/peirama_dipl/ALL_DATA/final_data/'

    ### WT1_1in6= 'D:/Files/peirama_dipl/ALL_DATA/final_data/wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_4_channel1.mat'
    WT1_1in6 = pc_path + 'wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_control_channel1.mat' 
    WT1_2in6 = pc_path + 'wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_first 20 min in 0 Mg_channel1.mat'
    WT1_3in6 = pc_path + 'wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_second 20 min in 0 Mg_channel1.mat'
    WT1_4in6 = pc_path + 'wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_third 20 min in 0 Mg_channel1.mat'
    WT1_5in6 = pc_path + 'wild_type_0Mg/1376.7 channel 1 (B2R)/satb1_1376.7_fourth_channel1.mat'
    WT1_6in6 = pc_path + 'wild_type_0Mg/1376.7 channel 1 (B2R)/satb1__1376.7_fifth 20 min in 0 Mg_channel1.mat'

    WT2_1in5 = pc_path + 'wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_control_channel2.mat'
    WT2_2in5 = pc_path + 'wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_first 20 min in 0 Mg_channel2.mat'
    WT2_3in5 = pc_path + 'wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_second 20 min in 0 Mg_channel2.mat'
    WT2_4in5 = pc_path + 'wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_third 20 min in 0 Mg_channel2.mat'
    WT2_5in5 = pc_path + 'wild_type_0Mg/1376.7 channel 2 (B2R)/satb1_1376.7_fourth 20 min in 0 Mg_channel2.mat'

    WT3_1in5 = pc_path + 'wild_type_0Mg/1451.4 channel2/satb1_1451.4_control_channel2.mat'
    WT3_2in5 = pc_path + 'wild_type_0Mg/1451.4 channel2/satb1_1451.4_first 20 min in 0 Mg_channel2.mat'
    WT3_3in5 = pc_path + 'wild_type_0Mg/1451.4 channel2/satb1_1451.4_second 20 min in 0 Mg_channel2_Nikos.mat'
    WT3_4in5 = pc_path + 'wild_type_0Mg/1451.4 channel2/satb1_1451.4_third 20 min in 0 Mg_channel2.mat'
    WT3_5in5 = pc_path + 'wild_type_0Mg/1451.4 channel2/satb1_1451.4_fourth 20 min in 0 Mg_channel2.mat'

    WT4_1in5 = pc_path + 'wild_type_0Mg/1451.4 channel4/satb1_1451.4_control_channel4.mat'
    WT4_2in5 = pc_path + 'wild_type_0Mg/1451.4 channel4/satb1_1451.4_first 20 min in 0 Mg_channel4.mat'
    WT4_3in5 = pc_path + 'wild_type_0Mg/1451.4 channel4/satb1_1451.4_second 20 min in 0 Mg_channel4_Nikos.mat'
    WT4_4in5 = pc_path + 'wild_type_0Mg/1451.4 channel4/satb1_1451.4_third 20 min in 0 Mg_channel4.mat'
    WT4_5in5 = pc_path + 'wild_type_0Mg/1451.4 channel4/satb1_1451.4_fourth 20 min in 0 Mg_channel4.mat'

    WT5_1in5 = pc_path + 'wild_type_0Mg/1494.2 channel1/satb1_1494.2_control_channel1.mat'
    WT5_2in5 = pc_path + 'wild_type_0Mg/1494.2 channel1/satb1_1494.2_first 20 min in 0 Mg_channel1.mat'
    WT5_3in5 = pc_path + 'wild_type_0Mg/1494.2 channel1/satb1_1494.2_second 20 min in 0 Mg_channel1.mat'
    WT5_4in5 = pc_path + 'wild_type_0Mg/1494.2 channel1/satb1_1494.2_third 20 min in 0 Mg_channel1.mat'
    WT5_5in5 = pc_path + 'wild_type_0Mg/1494.2 channel1/satb1_1494.2_fourth 20 min in 0 Mg_channel1.mat'

    WT6_1in5 = pc_path + 'wild_type_0Mg/1494.2 channel2/satb1_1494.2_control_channel2.mat'
    WT6_2in5 = pc_path + 'wild_type_0Mg/1494.2 channel2/satb1_1494.2_first 20 min in 0 Mg_channel2.mat'
    WT6_3in5 = pc_path + 'wild_type_0Mg/1494.2 channel2/satb1_1494.2_second 20 min in 0 Mg_channel2.mat'
    WT6_4in5 = pc_path + 'wild_type_0Mg/1494.2 channel2/satb1_1494.2_third 20 min in 0 Mg_channel2.mat'
    WT6_5in5 = pc_path + 'wild_type_0Mg/1494.2 channel2/satb1_1494.2_fourth 20 min in 0 Mg_channel2.mat'

    Mu1_1in5 = pc_path + 'mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_control_channel1.mat'
    Mu1_2in5 = pc_path + 'mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_first 20 min in 0 Mg_channel1.mat'
    Mu1_3in5 = pc_path + 'mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_second 20 min in 0 Mg_channel1.mat'
    Mu1_4in5 = pc_path + 'mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_third 20 min in 0 Mg_channel1.mat'
    Mu1_5in5 = pc_path + 'mutant_0Mg/1375.5 channel 1 (F2R)/satb1_1375.5_fourth 20 min in 0 Mg_channel1.mat'

    Mu2_1in5 = pc_path + 'mutant_0Mg/1473.1 channel4/satb1_1473.1_control_channel4.mat'
    Mu2_2in5 = pc_path + 'mutant_0Mg/1473.1 channel4/satb1_1473.1_first 20 min in 0 Mg_channel4.mat'
    Mu2_3in5 = pc_path + 'mutant_0Mg/1473.1 channel4/satb1_1473.1_second 20 min in 0 Mg_channel4.mat'
    Mu2_4in5 = pc_path + 'mutant_0Mg/1473.1 channel4/satb1_1473.1_third 20 min in 0 Mg_channel4.mat'
    Mu2_5in5 = pc_path + 'mutant_0Mg/1473.1 channel4/satb1_1473.1_fourth 20 min in 0 Mg_channel4.mat'

    Mu3_1in5 = pc_path + 'mutant_0Mg/1476.7 channel4/satb1_1476.7_control_channel4.mat'
    Mu3_2in5 = pc_path + 'mutant_0Mg/1476.7 channel4/satb1_1476.7_first 20 min in 0 Mg_channel4.mat'
    Mu3_3in5 = pc_path + 'mutant_0Mg/1476.7 channel4/satb1_1476.7_second 20 min in 0 Mg_channel4.mat'
    Mu3_4in5 = pc_path + 'mutant_0Mg/1476.7 channel4/satb1_1476.7_third 20 min in 0 Mg_channel4.mat'
    Mu3_5in5 = pc_path + 'mutant_0Mg/1476.7 channel4/satb1_1476.7_fourth 20 min in 0 Mg_channel4.mat'

    WT1_4AP_1in5 = pc_path + 'wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_control_channel2.mat'
    WT1_4AP_2in5 = pc_path + 'wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_1_channel2.mat'
    WT1_4AP_3in5 = pc_path + 'wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_2_channel2.mat'
    WT1_4AP_4in5 = pc_path + 'wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_3_channel2.mat'
    WT1_4AP_5in5 = pc_path + 'wild_type_4ΑΡ/1580.4 channel 2/4AP_1580.4_3_channel2.mat'

    WT2_4AP_1in5 = pc_path + 'wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_control_channel4.mat'
    WT2_4AP_2in5 = pc_path + 'wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_1_channel4.mat'
    WT2_4AP_3in5 = pc_path + 'wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_2_channel4.mat'
    WT2_4AP_4in5 = pc_path + 'wild_type_4ΑΡ/1677.3 channel4/4AP_1677.3_3_channel4.mat'
    WT2_4AP_5in5 = pc_path + 'wild_type_4ΑΡ/1677.3 channel4/per.mat'  # αυτό είναι όντως το 5ο αρχείο; κάντο plot για να δεις

    test1 = pc_path + 'TEST/1619.6 channel4/satb1_1619.6_control_channel4.mat'
    test2 = pc_path + 'TEST/1619.6 channel4/satb1_1619.6_1_channel4.mat'
    test3 = pc_path + 'TEST/1619.6 channel4/satb1_1619.6_3_channel4.mat'
    test4 = pc_path + 'TEST/1619.6 channel4/satb1_1619.6_4_channel4.mat'

    dirs={'WT1_1in6':WT1_1in6, 'WT1_2in6':WT1_2in6, 'WT1_3in6':WT1_3in6, 'WT1_4in6':WT1_4in6, 'WT1_5in6':WT1_5in6, 'WT1_6in6':WT1_6in6, 
          'WT2_1in5':WT2_1in5, 'WT2_2in5':WT2_2in5, 'WT2_3in5':WT2_3in5, 'WT2_4in5':WT2_4in5, 'WT2_5in5':WT2_5in5,
          'WT3_1in5':WT3_1in5, 'WT3_2in5':WT3_2in5, 'WT3_3in5':WT3_3in5,'WT3_4in5':WT3_4in5, 'WT3_5in5':WT3_5in5,
          'WT4_1in5':WT4_1in5, 'WT4_2in5':WT4_2in5, 'WT4_3in5':WT4_3in5, 'WT4_4in5':WT4_4in5, 'WT4_5in5':WT4_5in5,
          'WT5_1in5':WT5_1in5, 'WT5_2in5':WT5_2in5, 'WT5_3in5':WT5_3in5, 'WT5_4in5':WT5_4in5, 'WT5_5in5':WT5_5in5,
          'WT6_1in5':WT6_1in5, 'WT6_2in5':WT6_2in5, 'WT6_3in5':WT6_3in5, 'WT6_4in5':WT6_4in5, 'WT6_5in5':WT6_5in5,
          'Mu1_1in5':Mu1_1in5, 'Mu1_2in5':Mu1_2in5, 'Mu1_3in5':Mu1_3in5, 'Mu1_4in5':Mu1_4in5, 'Mu1_5in5':Mu1_5in5,
          'Mu2_1in5':Mu2_1in5, 'Mu2_2in5':Mu2_2in5, 'Mu2_3in5':Mu2_3in5, 'Mu2_4in5':Mu2_4in5, 'Mu2_5in5':Mu2_5in5,
          'Mu3_1in5':Mu3_1in5, 'Mu3_2in5':Mu3_2in5, 'Mu3_3in5':Mu3_3in5, 'Mu3_4in5':Mu3_4in5, 'Mu3_5in5':Mu3_5in5,
          'WT1_4AP_1in5':WT1_4AP_1in5, 'WT1_4AP_2in5':WT1_4AP_2in5, 'WT1_4AP_3in5':WT1_4AP_3in5, 'WT1_4AP_4in5':WT1_4AP_4in5, 'WT1_4AP_5in5':WT1_4AP_5in5,
          'WT2_4AP_1in5':WT2_4AP_1in5, 'WT2_4AP_2in5':WT2_4AP_2in5, 'WT2_4AP_3in5':WT2_4AP_3in5, 'WT2_4AP_4in5':WT2_4AP_4in5,'WT2_4AP_5in5':WT2_4AP_5in5,
          'test1':test1, 'test2':test2, 'test3':test3, 'test4':test4}

    return dirs[name]





def lists_of_names(selector:str):
    """This function can return easily the most basic groupings of the LFP signals. e.g. list_of_names('WT1') wiil return a list of 'WT1_1in6' to 'WT1_6in6' string in a list"""
    WT1 = ['WT1_1in6', 'WT1_2in6', 'WT1_3in6', 'WT1_4in6', 'WT1_5in6', 'WT1_6in6']
    WT2 = ['WT2_1in5', 'WT2_2in5', 'WT2_3in5', 'WT2_4in5', 'WT2_5in5']
    WT3=['WT3_1in5', 'WT3_2in5', 'WT3_3in5', 'WT3_4in5', 'WT3_5in5']
    WT4 = ['WT4_1in5', 'WT4_2in5', 'WT4_3in5', 'WT4_4in5', 'WT4_5in5']
    WT5 = ['WT5_1in5', 'WT5_2in5', 'WT5_3in5', 'WT5_4in5', 'WT5_5in5']
    WT6 = ['WT6_1in5', 'WT6_2in5', 'WT6_3in5', 'WT6_4in5', 'WT6_5in5']
    Mu1= ['Mu1_1in5', 'Mu1_2in5', 'Mu1_3in5', 'Mu1_4in5', 'Mu1_5in5']
    Mu2= ['Mu2_1in5', 'Mu2_2in5', 'Mu2_3in5', 'Mu2_4in5', 'Mu2_5in5']
    Mu3= ['Mu3_1in5', 'Mu3_2in5', 'Mu3_3in5', 'Mu3_4in5', 'Mu3_5in5']
    WT1_4AP = ['WT1_4AP_1in5', 'WT1_4AP_2in5', 'WT1_4AP_3in5', 'WT1_4AP_4in5', 'WT1_4AP_5in5']
    WT2_4AP = ['WT2_4AP_1in5', 'WT2_4AP_2in5', 'WT2_4AP_3in5', 'WT2_4AP_4in5', 'WT2_4AP_5in5']
    test = ['test1', 'test2', 'test3', 'test4']
    All_WT_0Mg = WT1 + WT2 + WT3 + WT4 + WT5 + WT6
    All_Mu_0Mg = Mu1 + Mu2 + Mu3
    All_EA = [WT1[0], WT2[0], WT3[0], WT4[0], WT5[0], WT6[0], Mu1[0], Mu2[0], Mu3[0], WT1_4AP[0], WT2_4AP[0]]
    All_EA_WT_0Mg = [WT1[0], WT2[0], WT3[0], WT4[0], WT5[0], WT6[0]]
    All_after_EA = WT1[1:] + WT2[1:] + WT3[1:] + WT4[1:] + WT5[1:] + WT6[1:] + Mu1[1:] + Mu2[1:] + Mu3[1:] + WT1_4AP[1:] + WT2_4AP[1:]
    All_after_EA_WT_0Mg = WT1[1:] + WT2[1:] + WT3[1:] + WT4[1:] + WT5[1:] + WT6[1:]
    All = All_WT_0Mg + All_Mu_0Mg + WT1_4AP + WT2_4AP
    dict= {'WT1':WT1, 'WT2':WT2, 'WT3':WT3, 'WT4':WT4, 'WT5':WT5, 'WT6':WT6, 'Mu1':Mu1, 'Mu2':Mu2,'Mu3':Mu3, 'WT1_4AP':WT1_4AP, 'WT2_4AP':WT2_4AP,
           'All_WT_0Mg':All_WT_0Mg, 'All_Mu_0Mg':All_Mu_0Mg, 'All_EA':All_EA, 'All_after_EA':All_after_EA, 'All_EA_WT_0Mg':All_EA_WT_0Mg, 
           'All_after_EA_WT_0Mg':All_after_EA_WT_0Mg, 'All':All, 'test':test}
    return dict[selector]





def time_series (name:str, downsampling:int):
    """This function takes as input the name of an LFP signal as they are in the above sdirs() function and returns a 2-line ndarray, with the LFP recording in the first
    line and the time stamps of each recorded value in the second line.
    The signal can also be downsampled. If downsampling ='None', no downsampling takes place, else downsampling must be an integer."""
    dr=sdirs(name)
    f=h5py.File(dr, 'r') # h5py.File can open any '.mat' file. Here it opens a struct as a dictionary
    s=list(f.keys())[1]
    struct=f['s']
    # signal=np.array(struct['signal']) # μάλλον δε χρειάζεται αν η κάτω σειρα με το np.squeeze λειτουργεί καλά
    signal=np.squeeze(np.array(struct['signal']))
    # time=np.array(struct['time']) # μάλλον δε χρειάζεται αν η κάτω σειρα με το np.squeeze λειτουργεί καλά
    time=np.squeeze(np.array(struct['time']))
    if downsampling =='None':
        time_ser=np.vstack((signal.T, time.T))
    else:
        signal=sn.decimate(signal, q=downsampling, axis=0)
        # signal=signal[0::downsample] # alternative way to simply take the 10th element without sn.decimate which also performs an anti-aliasing filter.
        time=time[0::downsampling]
        time_ser=np.vstack((signal.T, time.T))
        time_ser.astype(np.float32) # without this command data are extracted as float64. This turns them into float32 for saving memory
    return time_ser




## παλιά συνάρτηση -> μάλλον για διαγραφή
def extract_data_old (tag:str, downsampling:int, save_path:str, float_type_32:bool):
    """This method extracts the time series from many 20-minute files and creates a 2D ndarray of time_series that can be used for machine learning. 
    The signals can also be downsampled. If downsampling ='None', no downsampling takes place, else downsampling must be an integer.
    The save_path must be given as a string and the method will save the ndarray  in an .npy file
    The file can be created and saves as the original ndarray float64 dtype if the float_type_32 variable is False, or as a float64 dtype if the float_type_32 variable is 
    False. This option was created to save memory, and also becuase theoriginal torch tensor dtype is tensor32 and not tesnor64"""
    name_list = lists_of_names(tag)
    n=18461538 # make all time_series samples the same length, by zero-paddding, in order to take same length results
    data_list=[]
    for name in name_list:
        ts=time_series(name, downsampling)
        signal=ts[0,:].copy()
        time=ts[1,:]
        if downsampling != 'None': signal.resize((int(n/downsampling),), refcheck=False) #gia na einai ola ta simata isou mikous
        else: signal.resize((int(n),), refcheck=False) #gia na einai ola ta simata isou mikous
        # print(signal.shape)
        data_list.append(signal)
    if not(float_type_32): data_list = np.array(data_list)   
    if float_type_32: data_list = np.array(data_list, dtype=np.float32) # without this command data are extracted as float64. This turns them into float32 for saving memory
    print('The shape of extracted data is', data_list.shape)
    np.save(save_path, data_list)
    return data_list





def extract_data(tag:str, downsampling:int, save_path:str):
    """This method extracts the time series from many 20-minute files and creates a 2D ndarray of time_series that can be used for machine learning. 
    The signals can also be downsampled. If downsampling ='None', no downsampling takes place, else downsampling must be an integer.
    The save_path must be given as a string and the method will save the ndarray  in an .npy file
    The file can be created and saves as the original ndarray float64 dtype if the float_type_32 variable is False, or as a float64 dtype if the float_type_32 variable is 
    False. This option was created to save memory, and also becuase theoriginal torch tensor dtype is tensor32 and not tesnor64"""
    name_list = lists_of_names(tag)
    # n=18461538 # constant length make all time_series samples the same length, by zero-paddding, in order to take same length results (palios tropos me stathero n)
    # if downsampling != 'None': signal.resize((int(n/downsampling),), refcheck=False) # gia na einai ola ta simata isou mikous
    # else: signal.resize((int(n),), refcheck=False) # gia na einai ola ta simata isou mikous
    resize_list = []
    data_list=[]
    for name in name_list:
        ts=time_series(name, downsampling)
        signal=ts[0,:].copy()
        # time=ts[1,:]
        resize_list.append(signal.shape[0])
        # print(signal.shape)
        data_list.append(signal)
    print('extract data -> lfp_signal lengths before resize are: ', resize_list)
    # n = max(resize_list) # gia na epektinei me zero-padding oso to megisto mikos lfp
    n = min(resize_list) # gia na kovei oso to elaxisto mikos lfp
    for signal in data_list:
        if downsampling != 'None': signal.resize((int(n),), refcheck=False) # gia na einai ola ta simata isou mikous
        else: signal.resize((int(n),), refcheck=False) # gia na einai ola ta simata isou mikous
    data_list = np.array(data_list)      
    print('The shape of extracted data is', data_list.shape)
    np.save(save_path, data_list)
    return data_list    





def windowing (signal:np.ndarray, window_length:int, window_movement:int, overlap:float):
    """This function takes a 1D time-series signal as ndarray and cuts it into windows, returning as output a 2D ndarray with one window-segment in every line
    'window_length' variable determines the length of windows that will be created (all windows will be of equal length)
    'window movement' variable must ne an integer and determines how many steps the sliding window will make in order to extract the next window
    As an exception if the window_movent variable = "by overlapping" the the window movement is determined by the ovelrap degree and the window length
     the 'overlap' variable must take values between 0and 1 as a percentage e.g. if overlap = 0.7 the 70 % of thw last values of any window, will be the same as the
       70% of the first values of the next window """
    if window_movement == "by overlapping": window_movement=int(window_length * (1-overlap))
    windows=signal[0:window_length]
    sp=np.arange(window_movement,len(signal),window_movement)
    for i in sp:
        segm=signal[i:i+window_length]
        if len(segm) < window_length:
            break
        # print(f'{i} iteration')
        windows = np.vstack((windows, segm))
    return windows





def combine (list_of_names:list[str], downsampling:int):
    """This method combine signals to greater one. It's purpose is to take different 20-minute parts of an expiriment and combine them to the whole signal from the 
    beggining of the expiriment until its end. This way the whole signal contains the transition to epilepsy. As an input take a list of names of the lfp signals and
    returns the whole signal. The list can be extracted easilly from the lists_of_names() method { e.g. combine(lists_of_names('WT1)),10) } that exists in this module. 
    The whole signal can also be downsampled. If downsampling ='None', no downsampling takes place, else downsampling must be an integer."""
    signal = time_series(list_of_names[0], downsampling)
    lfp=signal[0,:]
    time= signal[1,:]
    fs=time[1]-time[0]
    for name in list_of_names[1:]:
        signal = time_series(name, downsampling)
        lfp_new=signal[0,:]
        time_new= signal[1,:]
        lfp = np.hstack((lfp, lfp_new))
        time = np.hstack((time, time_new + len(time)*fs))
    signal = np.vstack((lfp,time))
    #print(lfp.shape)
    #print(time.shape)
    return signal





def cut_to_events (name:str, downsampling:int, include_down_states:bool):
    """"This method cuts the signal to events with their type of event. It takes as an input the name of an lfp_signal and returns a 2D list. Every line of this list contains
    a ndarray which is the event recording and a string which is the type of this event. 
    - If the 'include_down_states variable' is True, the list will contain also the parts of the down/quiescent states between the events, while if the 'include_down_states' 
    variable is False the list will contain only the events by ommiting the down/quiescent states. 
    - The 'downsamplping' variable downsamples the events. If downsampling ='None', no downsampling takes place, else downsampling must be an integer."""
    dr=sdirs(name)
    file= h5py.File(dr, 'r')

    signal=np.squeeze(np.array(file['s/signal']))
    upStateStart=np.squeeze(np.array(file['s/upStateStart']))
    upStateEnd=np.squeeze(np.array(file['s/upStateEnd']))

    upStateFlg = []
    cell = file.get('s/upStateFlg')
    for i in np.arange(len(cell)):
        obj_ref= cell[i][0]
        refered = file[obj_ref] # αυτή η εντολή παίρνει το <HDF5 object reference> και επιστρέφει το περιεχώμενο του
        num = np.squeeze(np.array(refered))
        upStateFlg.append(num)
    upStateFlg = np.array (upStateFlg)

    struct_of_fields=file.get('s/classes')
    name_field = struct_of_fields['className']
    classes = []
    for i in np.arange(len(name_field)):
        obj_ref = name_field[i][0]
        refered = file[obj_ref]
        word =''
        for j in np.arange(len(refered)):
            try:
                letter = chr(refered[j][0])
                word = word + letter
            except IndexError:
                pass
        if word == 'Interictal' or word == 'Interictal Activity': word = 'Interictal activity' # για να υπάρχει κοινή ονομασία σε όλα τα αρχεία
        if word in ['SLE', 'Seizure like Event', 'Seizure Like Event'] : word = 'Seizure like event' # για να υπάρχει κοινή ονομασία σε όλα τα αρχεία
        if word != '': classes.append(word)
    
    num_of_events = len(upStateFlg)

    if not(include_down_states):
        cutted_to_events_signal = [[] for i in range(num_of_events)] # without down_states or quiescent activity
        for i in (np.arange(num_of_events-1)):
            i_event = signal[int(upStateStart[i]):int(upStateEnd[i])+1]
            if downsampling != 'None': i_event=sn.decimate(i_event, q=downsampling, axis=0)
            i_class_tag = classes[int(upStateFlg[i])]
            i_pair = [i_event, i_class_tag]
            cutted_to_events_signal[i] = i_pair
        return cutted_to_events_signal
    
    if include_down_states:
        cutted_to_events_signal = [[] for i in range(2*num_of_events+1)] # without down_states and quiescent activity
        zero_quiescent = signal[0 : int(upStateStart[0])]
        if downsampling != 'None': zero_quiescent=sn.decimate(zero_quiescent, q=downsampling, axis=0)
        zero_quiescent_pair = [zero_quiescent, 'Quiescent state']
        cutted_to_events_signal[0] = zero_quiescent_pair
        for i in (np.arange(num_of_events-1)):
            i_event = signal[int(upStateStart[i]):int(upStateEnd[i])+1]
            if downsampling != 'None': i_event=sn.decimate(i_event, q=downsampling, axis=0)
            i_class_tag = classes[int(upStateFlg[i])]
            i_event_pair = [i_event, i_class_tag]
            cutted_to_events_signal[2*i+1] = i_event_pair
            i_quiescent = signal[int(upStateEnd[i])+1:int(upStateStart[i+1])]
            if downsampling != 'None': i_quiescent=sn.decimate(i_quiescent, q=downsampling, axis=0)
            i_quiescent_pair = [i_quiescent, 'Quiescent state']
            cutted_to_events_signal[2*i+2] = i_quiescent_pair

        last_event = signal[int(upStateStart[num_of_events-1]):int(upStateEnd[num_of_events-1])+1]
        if downsampling != 'None': last_event=sn.decimate(last_event, q=downsampling, axis=0)
        last_class_tag = classes[int(upStateFlg[num_of_events-1])]
        last_event_pair = [last_event, last_class_tag]
        cutted_to_events_signal[2*(num_of_events-1)+1] = last_event_pair
        last_quiescent = signal[int(upStateEnd[num_of_events-1])+1:]
        if downsampling != 'None': last_quiescent=sn.decimate(last_quiescent, q=downsampling, axis=0)
        last_quiescent_pair = [last_quiescent, 'Quiescent state']
        cutted_to_events_signal[2*(num_of_events-1)+2] = last_quiescent_pair
        return cutted_to_events_signal
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def visualise_lfp_signal(name:str, downsampling:int, save_name:str, file_or_whole='file'):
    path = 'D:/Files/peirama_dipl/project_files/'
    if file_or_whole=='file': lfp_time_series = time_series(name, downsampling)
    if file_or_whole=='whole': lfp_time_series = combine (lists_of_names(name), downsampling)
    signal=lfp_time_series[0,:]
    time=lfp_time_series[1,:]
    plt.plot(time, signal)
    plt.title(f'File name:{name}, downsampling:{downsampling}')
    if save_name != 'None': # input save_name='None' if you dont want to save the plot
        plt.savefig(path + save_name + '.png') # if you want to change the saved file type, chnge '.svg' e.g. '.png'
    plt.show()
    plt.close()




def plot_PSD(name:str, downsampling:int, plot:str):
    lfp_time_series=time_series(name, downsampling)
    signal=lfp_time_series[0,:]
    time=lfp_time_series[1,:]
    d=time[4]-time[3]
    fs=1/d
    n=len(signal)

    if plot=='with numpy and mannual freqs' or plot=='all_methods':
        fhat=np.fft.fft(signal,n=n)
        amp=np.abs(fhat) # amplitude of the DFT
        Pxx=(amp**2)/n #power of the DFT
        freq=fs/n #frequency increment
        f = np.arange(n)*(fs/n) #Create x-axis of frequencies in Hz
        plt.plot(f,Pxx)
        plt.suptitle('plot with np.fft & manual f')
        plt.title(f'File name:{name}, downsampling:{downsampling}')
        plt.show()

    if plot=='with numpy and np computed freqs' or plot=='all_methods':
        fhat=np.fft.fft(signal,n=n)
        amp=np.abs(fhat) # amplitude of the DFT
        Pxx=(amp**2)/n #power of the DFT
        freq_np=np.fft.fftfreq(len(signal),d=d)
        idx=np.argsort(freq_np)
        plt.plot(freq_np[idx],Pxx[idx])
        plt.suptitle('plot with np.fftfreq')
        plt.title(f'File name:{name}, downsampling:{downsampling}')
        plt.show()

    if plot=='with numpy and mannual freqs on log scale' or plot=='all_methods': # plots the y axis in logarithmic scale
        fhat=np.fft.fft(signal,n=n)
        amp=np.abs(fhat) # amplitude of the DFT
        Pxx=(amp**2)/n #power of the DFT
        freq=fs/n #frequency increment
        f = np.arange(n)*(fs/n) #Create x-axis of frequencies in Hz
        plt.semilogy(f,Pxx)
        plt.suptitle('log plot with np.fft & manual f')
        plt.title(f'File name:{name}, downsampling:{downsampling}')
        plt.show()

    if plot=='with matplotlib' or plot=='all_methods':
        f1, Pxx = plt.psd(signal, Fs=fs,scale_by_freq=True)
        plt.suptitle('plot with plt.psd')
        plt.show()

    if plot == 'with scipy' or plot=='all_methods':
        f2, Pxx = sn.periodogram(signal,fs=fs,return_onesided=True,scaling='density')
        plt.plot(f2,Pxx)
        plt.suptitle('plot with scypi')
        plt.title(f'File name:{name}, downsampling:{downsampling}')
        plt.show()

    if plot == 'log with scipy on log scale' or plot=='all_methods': #plots the y axis in logarithmic scale
        f2, Pxx = sn.periodogram(signal,fs=fs,return_onesided=True,scaling='density')
        plt.semilogy(f2,Pxx)
        plt.suptitle('log plot with scipy')
        plt.title(f'File name:{name}, downsampling:{downsampling}')
        plt.show() 




def spectrogram(name:str, downsampling:int, method:str, log_scale:bool):
    # κλήση του αρχείου σήματος lfp
    lfp_time_series = time_series(name, downsampling)
    signal=lfp_time_series[0,:]
    time=lfp_time_series[1,:]
    d=time[4]-time[3]
    fs=1/d

    # Σχεδιασμός του spectrogram με τη χρήση του matplotlib
    if method=='matplotlib':
        plt.specgram(signal, Fs=fs, scale='dB', NFFT=4096)

    # Υπολογισμός των δεδομένων του spctrogram κατευθείαν χωρίς την ενδιάμεση χρήση του STFT
    if method=='scipy without stft':
        f, t, Sxx = sn.spectrogram(signal, fs=fs, window='hann')   
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.pcolormesh(t, f, 10*np.log10(Sxx/Sxx.max())) # 10*np.log10(Sxx/Sxx.max()) is the decibell normilization

    # Υπολογισμός των δεδομένων του spectrogram μέσω υπολογισμού του STFT
    if method=='scipy throught stft':
        f, t, Zxx = sn.stft(signal, fs=fs, nfft=256)           # Ο αλγόριθμος short time fourier transform, τα Zxx είναι οι μιγαδικοί αριθμοί του
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud') # without dB normalisation
        plt.pcolormesh(t, f, 10*np.log10(np.abs(Zxx)/np.abs(Zxx).max()), shading='gouraud') # with dB normalisation

    # λοιπές ρυθμίσεις του plot
    plt.colorbar(label = 'power/frequency;;;')
    plt.ylim(bottom=0, top=300)
    if log_scale:
        plt.yscale('symlog') # κάνει τον άξονα των συχνοτήτων να είναι λογαριθμικός αντί για γραμμικός (αντι γαι 50-100-150 κλπ, γίνεται 10-10**2-10**3 κλπ)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()




def visualize_events(name:str, downsampling:int):
    import signal_handler
    events = signal_handler.cut_to_events(name, downsampling, include_down_states=True)
    num_of_events = len(events)
    for i in np.arange(num_of_events):
        i_event = events[i]
        i_signal = i_event[0]
        i_tag = i_event[1]
        plt.plot(i_signal)
        plt.title(i_tag)
        plt.show()
        if i%5 == 0: 
            cont = int(input('Do you want to continue to the next file; (Yes = 1, No=0)\n'))
            if not(cont): break
    plt.close('all')


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def check_signal_chars(time_ser:np.ndarray, mat_file_name:str, downsampling:int, check_basic_stats:bool, check_time_chars:bool, check_stat_diagrams:bool, check_normality:bool):
    if mat_file_name != 'None':
        time_ser = time_series(mat_file_name, downsampling)
        signal = time_ser[0,:]
        time = time_ser[1,:]
        plt.plot(time, signal); plt.title(f'Signal name: {mat_file_name}, downsampling = {downsampling}'); plt.show(); plt.close()
    if mat_file_name == 'None':
        if time_ser.shape[0] == 2: # αν η time_series έχει και το χρόνο μαζί της
            signal = time_ser[0,:]
            time = time_ser[1,:]
            plt.plot(time, signal); plt.title(f'Signal name: {mat_file_name}'); plt.show(); plt.close()
        else:
            signal = time_ser
            plt.plot(signal); plt.title('The signal plotted'); plt.show(); plt.close()

    if check_basic_stats:
        print(f'length of signal: {signal.size}')
        print('signal mean is', signal.mean())
        print('signal standard deviation is', signal.std())
        print(f'signal range is [{signal.min()}, {signal.max()}]')

    if check_time_chars and time_ser.shape[0] == 2:
        d = time[1] - time[0]
        print(f'time intervals is {d} seconds (multiplied with {downsampling} due to downsampling)')
        print(f'sampling frequency is {1/d} (divided by {downsampling} due to downsampling)')
        intervals = np.zeros(signal.size - 1)
        for idx in range(signal.size-1): intervals[idx] = time[idx + 1] - time[idx]
        accurate_testing = (d==intervals)
        accurate_testing = accurate_testing.all()
        error = 10**(-10)
        approximate_testing = (abs(d-intervals) < error)
        approximate_testing = approximate_testing.all()
        print(f'All time intervals are exactly equal -> {accurate_testing}')
        print(f'All time intervals are almost exactly equal (difference < {error}) -> {approximate_testing}')
    elif check_time_chars and time_ser.shape[0] != 2: print('No time array has been provided. Time chars cannot be computed')

    # signal statistic diagrams (they are plotted seperatly because they are memory consuming)
    if check_stat_diagrams:
        plt.hist(signal, bins = 1000)
        plt.title ('signal histogram')
        plt.show()
        plt.close()

        plt.boxplot(signal)
        plt.suptitle('Boxplot (for checking outliers)')
        plt.title('If 2 black lines appear they are all outliers. the boxplot still appears in the midlle', fontsize=10)
        plt.show()
        plt.close()

    # cheks if signal is normal with qq plot & Shapiro_Wilk test
    if check_normality:
        sm.qqplot(signal, line='45')
        plt.title('Q-Q plot (for checking normality of data)')
        plt.show()
        plt.close()

        _, p = shapiro(signal)
        if p > 0.05: print('Acoording to Shapiro_Wilk test data are distributed normally (p>0.05)')
        else: print('Acoording to Shapiro_Wilk test data are NOT distributed normally (p<0.05)')




def check_scaling_results(tag:str, downsampling:int, norm_method:str, scaling_power = 4):
    """This method just checks visually the impact of normalization on a group of signals"""
    #### check normalization results in all signals
    norm_method_list = ['min_max', 'max_abs', 'z_normaliyzation', 'robust_scaling', 'decimal_scaling', 'log_normalization', 'None']
    scaler =lfp_scaler(norm_method, scaling_power)

    for idx, name in enumerate(lists_of_names(tag)):
        signal = time_series(name, downsampling)[0,:]
        print("Unscaled signal characteristics:")
        check_signal_chars(signal, 'None', downsampling, check_basic_stats =1, check_time_chars=0, check_stat_diagrams=1, check_normality=1)
        plt.plot(signal); plt.title(f'File name:{name}, Unscaled signal'); plt.show(); plt.close()
        signal = scaler.fit_transform1d(signal)
        print("\n\nScaled signal characteristicks:")
        check_signal_chars(signal, 'None', downsampling, check_basic_stats =1, check_time_chars=0, check_stat_diagrams=1, check_normality=1)
        plt.plot(signal); plt.title(f'File name:{name}, scaling method: {norm_method}'); plt.show(); plt.close()
        if idx%3 == 0: 
            cont = int(input('Do you want to continue to the next file; (Yes = 1, No=0)\n'))
            if not(cont): break




def check_signal_stats(tag:str, downsampling:int):
    """This method just prints the statistical characteristics of each signal on a group of signals"""
    for idx, name in enumerate(lists_of_names(tag)):
        print(f'Characteristics of {name}')
        check_signal_chars(np.array(0), name, downsampling, check_basic_stats =1, check_time_chars=0, check_stat_diagrams=1, check_normality=0)
        print('\n\n')
        if idx%3 == 0: 
            cont = int(input('Do you want to continue to the next file; (Yes = 1, No=0)\n'))
            if not(cont): break




def check_signal_time_chars(tag:str, downsampling:int):
    """This method just prints the time characteristics of each signal on a group of signals"""
    for idx, name in enumerate(lists_of_names(tag)):
        print(f'Characteristics of {name}')
        check_signal_chars(np.array(0), name, downsampling, check_basic_stats =0, check_time_chars=1, check_stat_diagrams=0, check_normality=0)
        print('\n\n')
        if idx%3 == 0: 
            cont = int(input('Do you want to continue to the next file; (Yes = 1, No=0)\n'))
            if not(cont): break




def check_signal_normality(tag:str, downsampling:int):
    """This method just checks each signal for normality on a group of signals"""
    for idx, name in enumerate(lists_of_names(tag)):
        print(f'Check normality for {name}')
        check_signal_chars(np.array(0), name, downsampling, check_basic_stats =1, check_time_chars=0, check_stat_diagrams=0, check_normality=1)
        print('\n\n')
        if idx%3 == 0: 
            cont = int(input('Do you want to continue to the next file; (Yes = 1, No=0)\n'))
            if not(cont): break

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class lfp_scaler:
    ### This class is an objects that scales/normalizes the signal
    ### αυτή η κλάσση δεν περιέχει inverse2d και inverse3d επειδή τα test input και test output του LSTM θα είναι μονοδιάστατα σήματα (βεβαία αν είχε inverse2d και inverse3d
    # ή θα γίνονταν με ενιαιες παραμέτρους κανονικοποίησης, ή οι παράμετροι κανονικοποιήσης θα έπρεπε να γίνουν πίνακεες)
    ### αυτή η κλση έχει τον κίνδυνο να είναι αρκετά πιο αργή για μεγάλους πίνακες από τους έτοιμους scalers του sklearn
    def __init__(self, scaling_method, scaling_power):
        self.scaling_method = scaling_method
        self.mean = np.NaN
        self.std = np.NaN
        self.min = np.NaN
        self.max = np.NaN
        self.decimal_scale = scaling_power
        self.Q1 = np.NaN # 25%
        self.Q2 = np.NaN # this is the median
        self.Q3 = np.NaN # 75%

    def inspect_scale_params(self):
        """This fucntion prints all the scaling parameters"""
        print("Scaling_method: ", self.scaling_method)
        print("Mean is: ", self.mean)
        print("Standard deviation is: ", self.std)
        print("Minimum is: ", self.min)
        print("Maximum is: ", self.max)
        print("First (25%) quantile is: ", self.Q1)
        print("Median (50% quantile) is: ", self.Q2)
        print("Third (75%) quantile is: ", self.Q3)

    def fit1d(self, signal:np.ndarray):
        """This function extracts the normalization parameters of a 1d time-series.
        This function is made for 1d signal, but in can used for a multi-dimensional matrix as well, extracting the normalization 
        parameters for the whole set of samples altogether """
        if self.scaling_method == 'min_max': self.min = signal.min(); self.max = signal.max()
        if self.scaling_method == 'max_abs': self.max = signal.max()
        if self.scaling_method == 'z_normalization': self.mean = signal.mean(); self.std = signal.std()
        if self.scaling_method == 'robust_scaling': self.Q1 = np.quantile(signal, 0.25); self.Q2 = np.median(signal); self.Q3 = np.quantile(signal, 0.75)
        if self.scaling_method == 'fit_all_params':
            self.mean = signal.mean()
            self.std = signal.std()
            self.min = signal.min()
            self.max = signal.max()
            self.Q1 = np.quantile(signal, 0.25)
            self.Q2 = np.median(signal)
            self.Q3 = np.quantile(signal, 0.75)

    def change_scaling_method(self, scaling_method):
        self.scaling_method = scaling_method

    def normalize1d (self, signal:np.ndarray, epsilon = 10 ** (-30)):
        """"This function normalizes a one dimensional time-series with the fited normalization parameters
        This function is made for 1d signal, but in can used for a multi-dimensional matrix as well, ableit with the same normalization parameters for the whole data and not
        individualized for every line i.e. every signal. Epsilon is used in order to prevent errors due to zero denominator """
        # epsilon = 10 ** (-30) ## to prevent errors due to zero denominator 
        if self.scaling_method == 'min_max': signal_norm = (signal - self.min)/((self.max - self.min) + epsilon)
        if self.scaling_method == 'max_abs': signal_norm = signal/(np.abs(self.max) + epsilon)
        if self.scaling_method == 'z_normalization': signal_norm = (signal - self.mean)/(self.std + epsilon)
        if self.scaling_method == 'robust_scaling': signal_norm = (signal - self.Q2)/ ((self.Q3 - self.Q1) + epsilon)
        if self.scaling_method == 'decimal_scaling': signal_norm = signal * np.array(10**self.decimal_scale, dtype=signal.dtype)
        if self.scaling_method == 'log_normalization': 
            if signal.min()<0: 
                print ("signal has negative values, it can't be logarithmized. The initial signal will be returnd")
                signal_norm  = signal
            else: signal_norm = np.array(np.log10(signal), dtype=signal.dtype)
        if self.scaling_method == 'None': signal_norm = signal
        return signal_norm

    def inverse1d(self, signal:np.ndarray):
        """This function inverses the normilization of a signal. Be careful to use the same method as the one you used to normalize the signal"""
        if self.scaling_method == 'min_max': signal_denorm = signal*(self.max - self.min) + self.min
        if self.scaling_method == 'max_abs': signal_denorm = signal*np.abs(self.max)
        if self.scaling_method == 'z_normalization': signal_denorm = signal*self.std + self.mean
        if self.scaling_method == 'robust_scaling': signal_denorm = signal*(self.Q3 - self.Q1) + self.Q2
        if self.scaling_method == 'decimal_scaling': signal_denorm = signal * np.array(10**(-self.decimal_scale), dtype=signal.dtype)
        if self.scaling_method == 'log_normalization': signal_denorm = np.array(10**(signal), dtype=signal.dtype)
        if self.scaling_method == 'None': signal_denorm = signal
        return signal_denorm
    
    def fit_transform1d(self, signal:np.ndarray):
        """This function fits and scales a one dimensional time-series at once"""
        self.fit1d(signal)
        # if self.std == 0: print('!!! std is zero'); print(signal)
        # if self.std == np.NaN: print('!!! std is NaN')
        signal_norm = self.normalize1d(signal)
        return signal_norm

    def normalize2d(self, data:np.ndarray):
        """This function fits and scales a two dimensional time-series sample by sample (i.e. row by row)"""
        for idx in np.arange(data.shape[0]):
            data[idx,:] = self.fit_transform1d(data[idx,:])
        # the normalization parameters would remain in the last sample, so they are initialized to NaN
        self.mean, self.std, self.min, self.max, self.Q1, self.Q2, self.Q3 = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        return data
     
    def fit2d (self, data:np.ndarray):
        """This function extract common parameters for a set of time-series. The assumption behind the process is that the means of individual parameters of all the time-series
        might be better reflecting the whole sample of time series, so they might be more suitable for normilizing individual time-series of the sample"""
        idx=0
        norm_params = np.array([data[idx,:].mean(), data[idx,:].std(), data[idx,:].min(), data[idx,:].max(), np.quantile(data[idx,:], 0.25), 
                    np.median(data[idx,:]), np.quantile(data[idx,:], 0.75)])
        for idx in np.arange(1,data.shape[0]):
            i_params = np.array([data[idx,:].mean(), data[idx,:].std(), data[idx,:].min(), data[idx,:].max(), np.quantile(data[idx,:], 0.25), 
                    np.median(data[idx,:]), np.quantile(data[idx,:], 0.75)])
            norm_params = np.vstack((norm_params, i_params))
        mean_norm_params = np.mean(norm_params, axis=0)
        self.mean, self.std, self.min, self.max, self.Q1, self.Q2, self.Q3 = tuple(mean_norm_params)

    def normalize3d(self, data:np.ndarray):
        """This function normalizes data only to the last dimension (like the nn.LayerNorm does but only for 3 dimansional arrays). The reason for this is because LSTM
        inputs are in 3 dimension, so are the batches"""
        for idx in np.arange(data.shape[0]):
            data[idx,:,:] = self.normalize2d(data[idx,:,:])
        return data
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


if  __name__ == "__main__":
    pass
    downsample_scale = 10
    test_series = combine(lists_of_names('test'), downsample_scale) # create test data
    np.save('D:/Files/peirama_dipl/project_files/test_series_ds'+ str(downsample_scale)  + '.npy', test_series)


    # tag = 'WT1'
    # downsample_scale = 1000
    # extract_data(tag, downsample_scale, 'D:/Files/peirama_dipl/project_files/LSTM_fc_data_' + tag + '_ds'+ str(downsample_scale)  + '.npy') # create training data
    
    # downsampling = 1
    # time_ser = combine (lists_of_names('WT1'), downsampling)
    # check_signal_chars(time_ser, 'None', downsampling, check_basic_stats=1, check_time_chars=1, check_stat_diagrams=1, check_normality=1)

    # name = 'WT2'
    # downsampling = 10
    # visualise_lfp_signal(name, downsampling, save_name = 'None', file_or_whole='whole')
    # visualize_events(name, downsampling)



    # name = 'WT1_1in6'
    # downsampling = 1000
    # save_path = 'D:/Files/peirama_dipl/project_files/' + 'WT1_EA' +'_ds' + str(downsampling)
    # ts = time_series (name, downsampling)
    # #ts = combine (lists_of_names('WT1'), downsampling)
    # np.save(save_path, ts)
    
    # name = 'WT1_1in6'
    # downsampling = 'None'
    # for name in lists_of_names('All'):
    #     plot_PSD(name, downsampling, plot = 'with scipy')

    # check_signal_stats('All_EA_WT_0Mg', 10)
    # check_signal_time_chars('All_EA_WT_0Mg', 10)






