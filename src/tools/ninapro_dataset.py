class OurDataset(Dataset):
    
    def __init__(self, dataset_path, transforms=None, windowSize=500, windowStride=250, noBaseline=True, subject='all', day='all'):
        """
        :param dataset_path (str) : path of the sEMG&FMG dataset
        :param transform    (bool): method of the data augmentation for sEMG and FMG
        :param noBaseline   (bool): if need relax data of the subject. default True  
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.noBaseline = noBaseline
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.subject = subject
        self.day = day
        self.data_path = self.getDataPath(self.dataset_path, self.windowSize, self.windowStride, self.subject, self.day)

    def __len__(self):
        return len(self.data_path) 

    def __getitem__(self, index):
        pass
    
    @staticmethod
    def getDataPath(path, wSize, wStride, subject, day):
        """
        Return the absoluate path of every input data    
        """
        if subject == 'all' and day == 'all':
            dataPathList = [os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject_path, day_path, gesture_path)
                            for subject_path in os.listdir(os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride)))
                            for day_path in os.listdir(os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject_path))
                            for gesture_path in os.listdir(os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject_path, day_path))]    

        elif subject != 'all' and day == 'all':
            dataPathList = [os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject, day_path, gesture_path)
                            for day_path in os.listdir(os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject))
                            for gesture_path in os.listdir(os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject, day_path))]   

        elif subject != 'all' and day != 'all':
            dataPathList = [os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject, day, gesture_path)
                            for gesture_path in os.listdir(os.path.join(path, 'data_'+str(wSize)+'_'+str(wStride), subject, day))]   

        return dataPathList

    def compute_mean_std(self):
        num = len(self.data_path)
        Mean = 0.0
        Std = 0.0
        for i in self.data_path:
            data = loadmat(i)['data']
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            Mean += mean
            Std += std
        Mean = Mean / num
        Std = Std / num
        print(Mean)
        print(Std)