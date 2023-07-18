# -*- coding: utf-8 -*-
# This script in order to get raw data from selected dataset path
import os
from scipy.io import savemat
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import normalize

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')

def getDataPath(path):
    """
    Return the raw data absoluate path    
    """
    # data path of the recording data during experiment
    all_lis = [os.path.join(path, subject_path, day_path, gesture_path)
                for subject_path in os.listdir(path)
                for day_path in os.listdir(os.path.join(path, subject_path))
                for gesture_path in os.listdir(os.path.join(path, subject_path, day_path))]
    
    # # data path of the relax data 
    # relax_list = [path for path in all_lis if path.split('_')[-1].endswith('relax.xlsx')]

    # data path of the stimu data
    stimu_lis = [path for path in all_lis if not path.split('_')[-1].endswith('relax.xlsx')]

    return stimu_lis

def select_periods(x, sample_rate=1000):
    """
    select the stimu period
    """
    return pd.concat([x[0:int(3*sample_rate)], x[int(5*sample_rate):int(8*sample_rate)]], axis=0, ignore_index=None)

def getStimData(stimuPath, noBaseLine=True, normalization=True, windowSize=500, windowStride=250):
    """
    get the stimu data
    """
    relaxDataPath = os.path.join(os.path.dirname(stimuPath), 'gesture_relax.xlsx')
    baseData = pd.read_excel(relaxDataPath, header=None).iloc[:, 2:].mean(axis=0)
    sti_data = pd.read_excel(stimuPath, header=None).iloc[:, 2:]
    if noBaseLine:
        sti_data = sti_data - baseData
    sti_data.columns = list(range(6))

    select_data = sti_data.apply(select_periods)
    select_data.index = list(range(select_data.shape[0]))
    select_data = np.array(select_data) # (6000, 6)

    if normalization:
        select_data[:, :3] = (select_data[:, :3] - select_data[:, :3].min()) / (select_data[:, :3].max() - select_data[:, :3].min())  # FMG normalization
        select_data[:, 3:] = (select_data[:, 3:] - select_data[:, 3:].min()) / (select_data[:, 3:].max() - select_data[:, 3:].min())  # EMG normalization
    # print(select_data)

    event1 = select_data[:3000, :]  # 3000 is duration of the event.
    event2 = select_data[3000:, :]

    length = int((event1.shape[0] - windowSize) / windowStride) + 1  # prepare for window work.
    for i in range(length):
        if windowStride*i+windowSize > event1.shape[0]:
            data1 = event1[event1.shape[0]-windowSize:]
            data2 = event2[event1.shape[0]-windowSize:]
        else:
            data1 = event1[windowStride*i:windowStride*i+windowSize]
            data2 = event2[windowStride*i:windowStride*i+windowSize]
        subject_name = stimuPath.split('/')[-3] # replace / to others if u do not run this code in linux.
        day_name = stimuPath.split('/')[-2]
        save_name1 = stimuPath.split('/')[-1].split('.')[0] + '_' + str(i) + '.mat'
        save_name2 = stimuPath.split('/')[-1].split('.')[0] + '_' + str(i+length) + '.mat'
        info_name = 'data_' + str(windowSize) + '_' + str(windowStride)
        save_path =  os.path.join(os.path.dirname(__file__), '..', 'data', info_name, subject_name, day_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path1 = os.path.join(save_path, save_name1)
        save_path2 = os.path.join(save_path, save_name2)

        # print(save_path)
        savemat(
            save_path1,
            {'data': data1}
        )    
        savemat(
            save_path2,
            {'data': data2}
        )
        

if __name__ == '__main__':
    sti = getDataPath(DATASET_PATH)
    with Pool(32) as p:
        p.map(getStimData, sti) 