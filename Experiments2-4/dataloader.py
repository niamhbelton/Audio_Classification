import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import cv2
import sys
import imageio



import librosa
import numpy as np

MAX_LEN = 1320
MINIMUM = -87.05343 #this is the minimum value of Mel-Spectrograms in the training data
MAXIMUM = 112.92507 #after substrating the minimum value from all values in the Mel Spectrogram, this is the maximum value in the Mel Spectrograms in the training data

class Dataset(data.Dataset):
    def __init__(self, target, split, create, transform=None):
        super().__init__()
        
        self.split = split
        self.create = create
        self.transform = transform
        
        #Retrieve the metadata depending on the split
        if self.split == 'train':
            target = target.loc[target['split']=='train'].reset_index(drop=True)
        elif self.split == 'valid':
            target = target.loc[target['split']=='valid'].reset_index(drop=True) 
        elif self.split == 'test':
            target = target.loc[target['split']=='test'].reset_index(drop=True) 
            target = target.drop(target.loc[(target['path'] =='missing') ].index.values, axis =0).reset_index(drop=True)

        #variables paths and labels store the path to the audio and its label
        self.paths = target['path'].tolist()
        self.labels = target['label_index'].tolist()
        
       
    def __len__(self): #returns the amount of data in the dataloader
        return len(self.paths)


    def __getitem__(self, index):
        #if creating the images, load in the audio data, otherwise read in the images
        if self.create == 1:
            
            y, sr = librosa.load(self.paths[index],22050)
            #if there is a transform, augment the data
            if self.transform:
                 y = self.transform(y)
    
            S = librosa.stft(y, n_fft=1024, hop_length=512, win_length=1024) #short-term fourier transfrom
            melspec = librosa.feature.melspectrogram(S=S, n_mels = 256) #create mel-spectrogram
            magnitude, phase = librosa.magphase(melspec) #extract the magniutde component from the data
            img = librosa.amplitude_to_db(magnitude) #convert amplitude to decibels
            img = (img - MINIMUM )/ MAXIMUM #normalise the images
            img = torch.FloatTensor(np.stack((img,)*3)) #stack the images so that dimensions are (3, 256, 1320)

            #pad with zeros if the last dimension is not equal to 1320
            if img.shape[2] != MAX_LEN:
                new_img = np.zeros((img.shape[0], img.shape[1],MAX_LEN))
                new_img[:, :, 0:img.shape[2]] = img
                img=torch.FloatTensor(new_img)
        else:
                #read in image
                img = imageio.imread(self.paths[index])
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) #convert from BGRA to BGR
                img = torch.FloatTensor(img / 255).permute(2,0,1) #normalise the image
                
        #get label
        label = self.labels[index]
        return img, label



