# -*- coding: utf-8 -*-
import numpy as np
import os
from scipy import signal
import pandas as pd
from scipy.io import loadmat, savemat
import torch
import random
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class NinaProDataset(Dataset):

    def __init__(self, DB_id, windowSize=[20], windowStride=1, transforms=None, mode='train', subjectID='S1') -> None:
        """
            param DB_id: (Dataset id)
            param windowSize: sample rate: 100Hz -> 1dot: 10ms  20dot: 200ms   (200ms -> 20dot, 150ms -> 15dot, 100ms -> 10dot, 50ms -> 5dot)
            param windowStride: overlap: 10ms -> 1dot
        """
        super().__init__()
        self.DB_id = DB_id
        self.dataset_path = os.path.join(BASE_DIR, "..", "..", "dataset", "PublicDataset",
                                         "NinaProDataset", "DB" + self.DB_id)
        self.transform = transforms
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.allDataPath = self.dataPath(mode)
        if subjectID is not None:
            self.allDataPath = [i for i in self.allDataPath if i.split('_')[0] == subjectID]

    def __getitem__(self, index):
        fileName = self.allDataPath[index]
        dataPath = os.path.join(self.path, fileName)
        data = loadmat(dataPath)
        emg = data['data']
        label = data['label'][0][0] - 1

        if self.transform is not None:
            emg = self.transform(emg)

        if emg.ndim == 3:
            emg = emg.transpose(0, 2, 1)
            label = np.stack((label, label), 0)
        else:
            emg = emg.transpose(1, 0)

        return emg, label

    def __len__(self):
        return len(self.allDataPath)

    def dataPath(self, mode='train'):
        if mode == 'train':
            self.path = os.path.join('/home/peiji/Desktop/NinaDataset', "DB" + self.DB_id + '_train')
        elif mode == 'test':
            self.path = os.path.join('/home/peiji/Desktop/NinaDataset', "DB" + self.DB_id + '_test')
        else:
            raise AttributeError('No such mode')
        allPath = os.listdir(self.path)
        return allPath

    def getData(self, mode='test'):
        if mode == 'train':
            path = os.path.join(self.dataset_path, 'train.csv')
            saved_path = os.path.join('/home/peiji/Desktop/NinaDataset', "DB" + self.DB_id + '_train')
        elif mode == 'test':
            path = os.path.join(self.dataset_path, 'test.csv')
            saved_path = os.path.join('/home/peiji/Desktop/NinaDataset', "DB" + self.DB_id + '_test')
        else:
            raise AttributeError('No such mode')

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        data = pd.read_csv(path)
        data = list(data.groupby("Path"))

        for windowSize in self.windowSize:
            for groupData in data:
                currentPath = groupData[0]
                indexData = groupData[1]
                name = currentPath.split('/')[-1].split('.')[0]
                currentData = loadmat(currentPath)['emg']
                # perform 1-order 1Hz low-pass filter
                order = 1
                fs = 100  # sample rate: 100Hz
                cutoff = 0.1  # cutoff frequency
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq

                b, a = signal.butter(order, normal_cutoff, btype='lowpass')
                for _col in range(currentData.shape[1]):
                    currentData[:, _col] = signal.filtfilt(b, a, currentData[:, _col])

                for j in range(indexData.shape[0]):
                    currentStartPoint = list(indexData['startPoint'])[j]
                    currentEndPoint = list(indexData['endPoint'])[j]
                    currentRerepetition = list(indexData['rerepetition'])[j]
                    currentLabel = list(indexData['restimulus'])[j]
                    length_dots = currentEndPoint - currentStartPoint
                    for idx in range(currentStartPoint, currentStartPoint + length_dots - windowSize + 1, self.windowStride):
                        saved_data = currentData[idx:idx+windowSize, :]
                        Saved_path = os.path.join(saved_path, name + '_' + str(currentRerepetition) + '_' + str(idx) + '_' + str(idx + windowSize) + '_' +
                                                  str(currentLabel) + '.mat')
                        savemat(Saved_path, {
                            'data': saved_data,
                            'label': currentLabel
                        })

    def getDataPath(self):
        # DB1 sampling rate 100Hz
        # DB1: repetition 2, 5, 7 are test data and the remaining repetitions are train data.
        subjectPathList = [os.path.join(self.dataset_path, i) for i in os.listdir(self.dataset_path) if
                           i.startswith('s')]
        sessionPathList = [os.path.join(i, j) for i in subjectPathList
                           for j in os.listdir(i)]
        return sessionPathList

    def mat2csv(self, mode='test'):
        sessionPathList = self.getDataPath()
        dataframe = pd.DataFrame(
            columns=['Path', 'subjectID', 'exerciseID', 'restimulus', 'rerepetition', 'startPoint', 'endPoint'])
        for path in sessionPathList:
            if path.split('_')[-1] == 'E1.mat':
                data = loadmat(path)
                subject_id = str(data['subject'][0][0])
                exercise_id = str(data['exercise'][0][0])
                restimulus = data['restimulus']
                rerepetition = data['rerepetition']
                restimu_length = restimulus.shape[0]
                rerepet_length = rerepetition.shape[0]
                assert restimu_length == rerepet_length
                for i in range(restimu_length):
                    if mode == 'train':
                        if (0 < i < restimu_length - 1 and restimulus[i - 1, 0] != restimulus[i, 0] and restimulus[i - 1] <
                            restimulus[i, 0]) and (rerepetition[i, 0] == 1 or \
                                                 rerepetition[i, 0] == 3 or rerepetition[i, 0] == 4 or rerepetition[
                                                     i, 0] == 6 or \
                                                 rerepetition[i, 0] == 8 or rerepetition[i, 0] == 9 or rerepetition[
                                                     i, 0] == 10):

                            start_point = i

                        elif (0 < i < restimu_length - 1 and restimulus[i + 1, 0] != restimulus[i, 0] and restimulus[i + 1] <
                              restimulus[i, 0]) and (rerepetition[i, 0] == 1 or \
                                                   rerepetition[i, 0] == 3 or rerepetition[i, 0] == 4 or rerepetition[
                                                       i, 0] == 6 or \
                                                   rerepetition[i, 0] == 8 or rerepetition[i, 0] == 9 or rerepetition[
                                                       i, 0] == 10):

                            end_point = i
                            assert np.sum(restimulus[start_point:end_point + 1, :]) / \
                                   restimulus[start_point:end_point + 1, :].shape[0] == restimulus[start_point,
                                                                                      :] == restimulus[end_point, :]
                            restimulus_id = restimulus[start_point, :]
                            rerepetition_id = rerepetition[start_point, :]
                            duration = (start_point, end_point + 1)
                            dataframe = dataframe.append(
                                pd.DataFrame({'Path': [path], 'subjectID': [subject_id], 'exerciseID': [exercise_id],
                                              'restimulus': [restimulus_id][0], 'rerepetition': [rerepetition_id][0],
                                              'startPoint': [duration[0]], 'endPoint': [duration[1]]}),
                                ignore_index=True)


                    elif mode == 'test':
                        if (0 < i < restimu_length - 1 and restimulus[i - 1, 0] != restimulus[i, 0] and restimulus[i - 1] <
                            restimulus[i, 0]) and (rerepetition[i, 0] == 2 or \
                                                 rerepetition[i, 0] == 5 or rerepetition[i, 0] == 7):

                            start_point = i

                        elif (0 < i < restimu_length - 1 and restimulus[i + 1, 0] != restimulus[i, 0] and restimulus[i + 1] <
                              restimulus[i, 0]) and (rerepetition[i, 0] == 2 or \
                                                   rerepetition[i, 0] == 5 or rerepetition[i, 0] == 7):

                            end_point = i
                            assert np.sum(restimulus[start_point:end_point + 1, :]) / \
                                   restimulus[start_point:end_point + 1, :].shape[0] == restimulus[start_point,
                                                                                      :] == restimulus[end_point, :]
                            restimulus_id = restimulus[start_point, :]
                            rerepetition_id = rerepetition[start_point, :]
                            duration = (start_point, end_point + 1)
                            dataframe = dataframe.append(
                                pd.DataFrame({'Path': [path], 'subjectID': [subject_id], 'exerciseID': [exercise_id],
                                              'restimulus': [restimulus_id][0], 'rerepetition': [rerepetition_id][0],
                                              'startPoint': [duration[0]], 'endPoint': [duration[1]]}),
                                ignore_index=True)

            elif path.split('_')[-1] == 'E2.mat':
                data = loadmat(path)
                subject_id = str(data['subject'][0][0])
                exercise_id = str(data['exercise'][0][0])
                restimulus = data['restimulus']
                rerepetition = data['rerepetition']
                restimu_length = restimulus.shape[0]
                rerepet_length = rerepetition.shape[0]
                assert restimu_length == rerepet_length
                for i in range(restimu_length):
                    if mode == 'train':
                        if (0 < i < restimu_length - 1 and restimulus[i - 1, 0] != restimulus[i, 0] and restimulus[
                            i - 1] <
                            restimulus[i, 0]) and (rerepetition[i, 0] == 1 or rerepetition[i, 0] == 3 or rerepetition[i, 0] == 4 or rerepetition[
                                                       i, 0] == 6 or rerepetition[i, 0] == 8 or rerepetition[i, 0] == 9 or rerepetition[
                                                       i, 0] == 10):

                            start_point = i

                        elif (0 < i < restimu_length - 1 and restimulus[i + 1, 0] != restimulus[i, 0] and restimulus[
                            i + 1] <
                              restimulus[i, 0]) and (rerepetition[i, 0] == 1 or rerepetition[i, 0] == 3 or rerepetition[i, 0] == 4 or rerepetition[
                                                         i, 0] == 6 or rerepetition[i, 0] == 8 or rerepetition[i, 0] == 9 or rerepetition[
                                                         i, 0] == 10):

                            end_point = i
                            assert np.sum(restimulus[start_point:end_point + 1, :]) / \
                                   restimulus[start_point:end_point + 1, :].shape[0] == restimulus[start_point,
                                                                                        :] == restimulus[end_point, :]
                            restimulus_id = restimulus[start_point, :] + 12
                            rerepetition_id = rerepetition[start_point, :]
                            duration = (start_point, end_point + 1)
                            dataframe = dataframe.append(
                                pd.DataFrame({'Path': [path], 'subjectID': [subject_id], 'exerciseID': [exercise_id],
                                              'restimulus': [restimulus_id][0], 'rerepetition': [rerepetition_id][0],
                                              'startPoint': [duration[0]], 'endPoint': [duration[1]]}),
                                ignore_index=True)

                    elif mode == 'test':
                        if (0 < i < restimu_length - 1 and restimulus[i - 1, 0] != restimulus[i, 0] and restimulus[
                            i - 1] <
                            restimulus[i, 0]) and (rerepetition[i, 0] == 2 or rerepetition[i, 0] == 5 or rerepetition[i, 0] == 7):

                            start_point = i

                        elif (0 < i < restimu_length - 1 and restimulus[i + 1, 0] != restimulus[i, 0] and restimulus[
                            i + 1] <
                              restimulus[i, 0]) and (rerepetition[i, 0] == 2 or rerepetition[i, 0] == 5 or rerepetition[i, 0] == 7):

                            end_point = i
                            assert np.sum(restimulus[start_point:end_point + 1, :]) / \
                                   restimulus[start_point:end_point + 1, :].shape[0] == restimulus[start_point,
                                                                                        :] == restimulus[end_point, :]
                            restimulus_id = restimulus[start_point, :] + 12
                            rerepetition_id = rerepetition[start_point, :]
                            duration = (start_point, end_point + 1)
                            dataframe = dataframe.append(
                                pd.DataFrame({'Path': [path], 'subjectID': [subject_id], 'exerciseID': [exercise_id],
                                              'restimulus': [restimulus_id][0], 'rerepetition': [rerepetition_id][0],
                                              'startPoint': [duration[0]], 'endPoint': [duration[1]]}),
                                ignore_index=True)

            elif path.split('_')[-1] == 'E3.mat':
                data = loadmat(path)
                subject_id = str(data['subject'][0][0])
                exercise_id = str(data['exercise'][0][0])
                restimulus = data['restimulus']
                rerepetition = data['rerepetition']
                restimu_length = restimulus.shape[0]
                rerepet_length = rerepetition.shape[0]
                assert restimu_length == rerepet_length
                for i in range(restimu_length):
                    if mode == 'train':
                        if (0 < i < restimu_length - 1 and restimulus[i - 1, 0] != restimulus[i, 0] and restimulus[
                            i - 1] <
                            restimulus[i, 0]) and (rerepetition[i, 0] == 1 or \
                                                   rerepetition[i, 0] == 3 or rerepetition[i, 0] == 4 or rerepetition[
                                                       i, 0] == 6 or \
                                                   rerepetition[i, 0] == 8 or rerepetition[i, 0] == 9 or rerepetition[
                                                       i, 0] == 10):

                            start_point = i

                        elif (0 < i < restimu_length - 1 and restimulus[i + 1, 0] != restimulus[i, 0] and restimulus[
                            i + 1] <
                              restimulus[i, 0]) and (rerepetition[i, 0] == 1 or \
                                                     rerepetition[i, 0] == 3 or rerepetition[i, 0] == 4 or rerepetition[
                                                         i, 0] == 6 or \
                                                     rerepetition[i, 0] == 8 or rerepetition[i, 0] == 9 or rerepetition[
                                                         i, 0] == 10):

                            end_point = i
                            assert np.sum(restimulus[start_point:end_point + 1, :]) / \
                                   restimulus[start_point:end_point + 1, :].shape[0] == restimulus[start_point,
                                                                                        :] == restimulus[end_point, :]
                            restimulus_id = restimulus[start_point, :] + 29
                            rerepetition_id = rerepetition[start_point, :]
                            duration = (start_point, end_point + 1)
                            dataframe = dataframe.append(
                                pd.DataFrame({'Path': [path], 'subjectID': [subject_id], 'exerciseID': [exercise_id],
                                              'restimulus': [restimulus_id][0], 'rerepetition': [rerepetition_id][0],
                                              'startPoint': [duration[0]], 'endPoint': [duration[1]]}),
                                ignore_index=True)


                    elif mode == 'test':
                        if (0 < i < restimu_length - 1 and restimulus[i - 1, 0] != restimulus[i, 0] and restimulus[
                            i - 1] <
                            restimulus[i, 0]) and (rerepetition[i, 0] == 2 or \
                                                   rerepetition[i, 0] == 5 or rerepetition[i, 0] == 7):

                            start_point = i

                        elif (0 < i < restimu_length - 1 and restimulus[i + 1, 0] != restimulus[i, 0] and restimulus[
                            i + 1] <
                              restimulus[i, 0]) and (rerepetition[i, 0] == 2 or \
                                                     rerepetition[i, 0] == 5 or rerepetition[i, 0] == 7):

                            end_point = i
                            assert np.sum(restimulus[start_point:end_point + 1, :]) / \
                                   restimulus[start_point:end_point + 1, :].shape[0] == restimulus[start_point,
                                                                                        :] == restimulus[end_point, :]
                            restimulus_id = restimulus[start_point, :] + 29
                            rerepetition_id = rerepetition[start_point, :]
                            duration = (start_point, end_point + 1)
                            dataframe = dataframe.append(
                                pd.DataFrame({'Path': [path], 'subjectID': [subject_id], 'exerciseID': [exercise_id],
                                              'restimulus': [restimulus_id][0], 'rerepetition': [rerepetition_id][0],
                                              'startPoint': [duration[0]], 'endPoint': [duration[1]]}),
                                ignore_index=True)

        dataframe.to_csv(os.path.join(self.dataset_path, mode + '.csv'))

    def plotData(self, threshold=0.5):
        sessionPathList = self.getDataPath()
        sessionPathList = sorted(sessionPathList)
        data = loadmat(sessionPathList[0])
        subject_id = str(data['subject'][0][0])
        exercise_id = str(data['exercise'][0][0])

        currentData = data['emg']
        # perform 1-order 1Hz low-pass filter
        order = 1
        fs = 100  # sample rate: 100Hz
        cutoff = 1  # cutoff frequency
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, 'lowpass')
        for _col in range(currentData.shape[1]):
            currentData[:, _col] = signal.filtfilt(b, a, currentData[:, _col])

        emg = currentData[8845:9038]
        emg_length = emg.shape[0]
        # restimulus = data['stimulus'][:int(emg_length * threshold)][8845:9038]
        # rerepetition = data['repetition'][:int(emg_length * threshold)][8845:9038]

        restimulus = data['stimulus'][8845:9038]
        rerepetition = data['repetition'][8845:9038]

        # plot emg
        plt.figure(figsize=(6, 3))
        # x = emg[:int(emg_length * threshold)].shape[0]
        x = emg_length
        for i in range(10):
            plt.plot(range(x), emg[:, i])
            # print(min(emg[:, i]))
        # plt.plot(range(x), stimulus)
        #plt.plot(range(x), restimulus)
        # plt.plot(range(x), repetition)
        #plt.plot(range(x), rerepetition)
        plt.show()

    def compute(self):
        Paths = self.allDataPath
        total_length = len(Paths)
        total_sum = np.array([0.0] * 10)
        total_std = np.array([0.0] * 10)
        for i in tqdm(Paths):
            current_path = os.path.join(self.path, i)
            data = loadmat(current_path)
            emg = data['data']
            minmax_scale = MinMaxScaler()
            emg = minmax_scale.fit_transform(emg)

            average = np.mean(emg, axis=0)
            std = np.std(emg, axis=0)
            total_sum += average
            total_std += std

        average_value = total_sum / total_length
        std_value = total_std / total_length
        return average_value, std_value


if __name__ == '__main__':
    ninapro = NinaProDataset("1")
    ninapro.getData(mode='test')