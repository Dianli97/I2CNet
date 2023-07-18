from math import perm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Gaussion noise
flag = 0
if flag:
    random_signal = np.random.normal(loc=0, scale=5, size=1000)
    x_axis = range(1000)

    noise = np.random.normal(loc=0, scale=0.5, size=1000)

    plt.subplots_adjust(hspace=1)

    plt.subplot(4, 1, 1)
    plt.title('raw data')
    plt.plot(x_axis, random_signal, 'r')

    plt.subplot(4, 1, 2)
    plt.title('noise')
    plt.plot(x_axis, noise, 'g')

    plt.subplot(4, 1, 3)
    plt.title('raw data + noise')
    plt.plot(x_axis, random_signal+noise, 'b')

    plt.subplot(4, 1, 4)
    plt.plot(x_axis, random_signal, 'r')
    plt.plot(x_axis, noise, 'g')
    plt.plot(x_axis, random_signal+noise, 'b')

    plt.show()

# Scaling
flag = 0
if flag:
    a = np.random.normal(loc=0.0, scale=0.5, size=(500, 2))
    
    scalingFactor = np.random.normal(loc=10.0, scale=0.2, size=(1, 2))
    noise = np.matmul(np.ones((500, 1)), scalingFactor)
    b = a * noise

    plt.subplot(2, 2, 1)
    plt.title('raw data channel 1')
    plt.plot(range(500), np.squeeze(a[:, 0]), 'r')

    plt.subplot(2, 2, 2)
    plt.title('raw data channel 2')
    plt.plot(range(500), np.squeeze(a[:, 1]), 'r')

    plt.subplot(2, 2, 3)
    plt.title('raw data * noise, channel 1')
    plt.plot(range(500), np.squeeze(b[:, 0]), 'b')

    plt.subplot(2, 2, 4)
    plt.title('raw data * noise, channel 2')
    plt.plot(range(500), np.squeeze(b[:, 1]), 'b')

    plt.show()

# Roatation
# flag = 0
# if flag:
#     a = np.random.normal(loc=0.0, scale=2, size=(500, 2))
    
#     a_axis = np.random.uniform(low=-1, high=1, size=3)
#     a_angle = np.random.uniform(low=-np.pi, high=np.pi)
#     b = np.matmul(a, axangle2mat(a_axis, a_angle))

#     plt.subplot(2, 2, 1)
#     plt.title('raw data channel 1')
#     plt.plot(range(500), np.squeeze(a[:, 0]), 'r')

#     plt.subplot(2, 2, 2)
#     plt.title('raw data channel 2')
#     plt.plot(range(500), np.squeeze(a[:, 1]), 'r')

#     plt.subplot(2, 2, 3)
#     plt.title('rotation data, channel 1')
#     plt.plot(range(500), np.squeeze(b[:, 0]), 'b')

#     plt.subplot(2, 2, 4)
#     plt.title('rotation data, channel 2')
#     plt.plot(range(500), np.squeeze(b[:, 1]), 'b')

#     plt.show()


class Permutation(object):

    def __init__(self, nPerm=4, minSegLength=10, p=0.5, wSize=500):
        self.p = p
        self.wSize = wSize
        self.nPerm = nPerm
        self.minSegLength = minSegLength

    def __call__(self, signal):
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.zeros(signal.shape)
            idx = np.random.permutation(self.nPerm)
            flag = True
            while flag == True:
                segs = np.zeros(self.nPerm+1, dtype=int)
                segs[1:-1] = np.sort(np.random.randint(self.minSegLength, self.wSize-self.minSegLength, self.nPerm-1))
                segs[-1] = self.wSize
                if np.min(segs[1:]-segs[0:-1]) > self.minSegLength:
                    flag = False
            
            pp = 0
            for i in range(self.nPerm):
                signal_temp = signal[segs[idx[i]]:segs[idx[i]+1],:]
                signal_[pp:pp+len(signal_temp),:] = signal_temp
                pp += len(signal_temp)
            return signal_
        return signal
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(p={})'.format(self.p)

flag = 0
if flag:
    a = np.random.normal(loc=0.0, scale=1, size=(500, 2))
    
    permu = Permutation(p=1.0)
    b = permu(a)

    plt.subplot(2, 2, 1)
    plt.title('raw data channel 1')
    plt.plot(range(500), np.squeeze(a[:, 0]), 'r')

    plt.subplot(2, 2, 2)
    plt.title('raw data channel 2')
    plt.plot(range(500), np.squeeze(a[:, 1]), 'r')

    plt.subplot(2, 2, 3)
    plt.title('raw data * noise, channel 1')
    plt.plot(range(500), np.squeeze(b[:, 0]), 'b')

    plt.subplot(2, 2, 4)
    plt.title('raw data * noise, channel 2')
    plt.plot(range(500), np.squeeze(b[:, 1]), 'b')

    plt.show()

class MagnitudeWarping(object):

    def __init__(self, sigma=0.2, knot=4, p=0.5, wSize=500, channels=1):
        self.p = p
        self.x = (np.ones((channels, 1)) * (np.arange(0, wSize, (wSize-1)/(knot+1)))).transpose()
        self.y = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, channels))
        self.x_range = np.arange(wSize)
        self.randomCurves = np.squeeze(np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range) for i in range(channels)]).transpose())
       
        
    def __call__(self, signal):
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            return signal_ * self.randomCurves, self.randomCurves
        return signal

flag = 1
if flag:
    a = np.random.normal(loc=0.0, scale=0.5, size=(500, ))
    print(a.mean(axis=0))
    
    permu = MagnitudeWarping(p=1.0, sigma=0.2, knot=4)
    b, c = permu(a)
    print(b.mean(axis=0))

    plt.subplot(2, 2, 1)
    plt.title('raw data channel 1')
    plt.plot(range(500), a, 'r')

    plt.subplot(2, 2, 2)
    plt.title('MW')
    plt.plot(range(500), c, 'y')

    plt.subplot(2, 2, 3)
    plt.title('raw data * noise, channel 1')
    plt.plot(range(500), b, 'b')

    # plt.subplot(2, 2, 4)
    # plt.title('raw data * noise, channel 2')
    # plt.plot(range(500), np.squeeze(b[:, 0]), 'b')

    plt.show()


class TimeWarping(object):

    def __init__(self, sigma=0.2, knot=4, p=0.5, wSize=500):
        self.p = p
        self.wSize = wSize
        self.x = (np.ones((2, 1)) * (np.arange(0, wSize, (wSize-1)/(knot+1)))).transpose()
        self.y = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, 2))
        self.x_range = np.arange(wSize)
        self.tt = np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range) for i in range(2)]).transpose()
        self.tt_cum = np.cumsum(self.tt, axis=0)
        # set the shape
        self.t_scale = [(wSize-1) / self.tt_cum[-1, i] for i in range(2)]
        for i in range(2): self.tt_cum[:, i] = self.tt_cum[:, i] * self.t_scale[i]

    def __call__(self, signal):
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            signal_new = np.zeros((self.wSize, 2))
            x_range = np.arange(self.wSize)
            for i in range(2): signal_new[:,i] = np.interp(x_range, self.tt_cum[:,0], signal_[:,i])
            return signal_new
        return signal

flag = 0
if flag:
    a = np.random.normal(loc=0.0, scale=1, size=(500, 2))
    print(a.mean(axis=0))
    
    permu = TimeWarping(p=1.0)
    b = permu(a)
    print(b.mean(axis=0))

    plt.subplot(2, 2, 1)
    plt.title('raw data channel 1')
    plt.plot(range(500), np.squeeze(a[:, 0]), 'r')

    plt.subplot(2, 2, 2)
    plt.title('raw data channel 2')
    plt.plot(range(500), np.squeeze(a[:, 1]), 'r')

    plt.subplot(2, 2, 3)
    plt.title('raw data * noise, channel 1')
    plt.plot(range(500), np.squeeze(b[:, 0]), 'b')

    plt.subplot(2, 2, 4)
    plt.title('raw data * noise, channel 2')
    plt.plot(range(500), np.squeeze(b[:, 1]), 'b')

    plt.show()

class RandomSampling(object):
    """Perform the RandomSampling to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the TimeWarping. Default value is 0.5.
        sigma    (float) : sd of the scale value.
        knot     (int)   :                      .                     
        wSize    (int)   : Length of the input data.
        channels (int)   : Number of the input channels.
    """

    def __init__(self, p=0.5, nSample=300, wSize=500, channels=6):
        self.p = p
        self.wSize = wSize
        self.channels = channels
        self.tt = np.zeros((nSample, channels), dtype=int)
        for i in range(channels): 
            self.tt[1:-1, i] = np.sort(np.random.randint(1, wSize-1,nSample-2))
        self.tt[-1, :] = wSize-1

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be performed the RandomSampling.

        Returns:
            Signal or Tensor: Randomly RandomSampling signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            new_signal = np.zeros((self.wSize, self.channels))
            for i in range(self.channels): 
                new_signal[:, i] = np.interp(np.arange(self.wSize), self.tt[:, i], signal_[self.tt[:, i], i])
            return new_signal
        return signal

flag = 0
if flag:
    a = np.random.normal(loc=0.0, scale=1, size=(500, 2))
    print(a.mean(axis=0))
    permu = RandomSampling(p=1.0, channels=2)
    b = permu(a)
    print(b.mean(axis=0))

    plt.subplot(2, 2, 1)
    plt.title('raw data channel 1')
    plt.plot(range(500), np.squeeze(a[:, 0]), 'r')

    plt.subplot(2, 2, 2)
    plt.title('raw data channel 2')
    plt.plot(range(500), np.squeeze(a[:, 1]), 'r')

    plt.subplot(2, 2, 3)
    plt.title('raw data * noise, channel 1')
    plt.plot(range(500), np.squeeze(b[:, 0]), 'b')

    plt.subplot(2, 2, 4)
    plt.title('raw data * noise, channel 2')
    plt.plot(range(500), np.squeeze(b[:, 1]), 'b')

    plt.show()

# 7. Random Cutout.
class RandomCutout(object):
    """Random cutout selected area of the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the random cutout. Default value is 0.5.
        area     (float) : size of fixed area.
        num      (int)   : number of the area.                     
        wSize    (int)   : Length of the input data.
        channels (int)   : Number of the input channels.
        default  (float) : replace value of the cutout area 
    """
    def __init__(self, p=0.5, area=10, num=8, wSize=500, channels=6, default=0.0):
        self.p = p
        self.area = area
        self.num = num
        self.wSize = wSize
        self.channels = channels
        self.default = default

    def __call__(self, signal):
        """
        Args:
            Signal (EMG and FMG or Tensor): Signal to be performed the RandomSampling.

        Returns:
            Signal or Tensor: Randomly RandomSampling signal.
        """
        if np.random.uniform(0, 1) < self.p:
            signal_ = np.array(signal).copy()
            mask = np.ones(self.wSize, np.float32)
            for _ in range(self.num):
                x = np.random.randint(self.wSize)

                x1 = np.clip(x - self.area // 2, 0, self.wSize)
                x2 = np.clip(x + self.area // 2, 0, self.wSize)
                print(x1, x2)

                mask[x1:x2] = 0
            
            new_mask = np.zeros((self.wSize, self.channels))
            for i in range(self.channels):
                new_mask[:, i] = mask
            print(new_mask)
            
            mask_b = 1- new_mask
            print(mask_b)
            signal_ = signal_ * new_mask + mask_b * self.default
            return signal_
        return signal

flag = 0
if flag:
    a = np.random.normal(loc=0.0, scale=1, size=(500, 2))
    print(a.mean(axis=0))
    permu = RandomCutout(p=1.0, channels=2, default=0)
    b = permu(a)
    print(b.mean(axis=0))

    plt.subplot(2, 2, 1)
    plt.title('raw data channel 1')
    plt.plot(range(500), np.squeeze(a[:, 0]), 'r')

    plt.subplot(2, 2, 2)
    plt.title('raw data channel 2')
    plt.plot(range(500), np.squeeze(a[:, 1]), 'r')

    plt.subplot(2, 2, 3)
    plt.title('raw data * noise, channel 1')
    plt.plot(range(500), np.squeeze(b[:, 0]), 'b')

    plt.subplot(2, 2, 4)
    plt.title('raw data * noise, channel 2')
    plt.plot(range(500), np.squeeze(b[:, 1]), 'b')

    plt.show()