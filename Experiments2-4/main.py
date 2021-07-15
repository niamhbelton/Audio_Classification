import os
import pandas as pd
import config
import numpy as np
from torchvision import transforms
import argparse
import torch
from dataloader import Dataset
from train import train
from torchvision import transforms
from evaluate import test
import librosa


#The code in the functions Noise, Shift, change_speed, stretch, hpss, change_pitch are based on code from
#*    Title: Music Genre Classification Pytorch
#*    Author: SeungH eon Doh
#*    Date: 2019
#*    Availability: https://github.com/Dohppak/Music_Genre_Classification_Pytorch

class Noise(object):
    #adding noise to the data
    def __init__(self, margin=0.005):
        self.margin = margin

    def __call__(self, data):
        rand = np.random.randint(low= 0, high = 4)
        if rand == 0:
            noise = np.random.randn(len(data))
            data_noise = data + self.margin * noise
            return data_noise
        else:
            return data

class Shift(object):
    #shifting the audio
    def __init__(self):
        print('')

    def __call__(self, data):
        rand = np.random.randint(low= 0, high = 4)
        if rand == 0:
            return  np.roll(data, 1600)
        else:
            return data


class change_speed(object):
    #Changing the speed of the audio
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, data):
        rand = np.random.randint(low= 0, high = 4)
        if rand == 0:
            y_speed = data.copy()
            speed_change = np.random.uniform(low=self.low, high=self.high)
            tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
            minlen = min(y_speed.shape[0], tmp.shape[0])
            y_speed *= 0
            y_speed[0:minlen] = tmp[0:minlen]
            return y_speed
        else:
            return data


class hpss(object):
    #This function returns only the percussion component of the audio
    def __init__(self):
        print('')

    def __call__(self, data):
        rand = np.random.randint(low= 0, high = 4)
        if rand == 0:
            y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'))
            return y_percussive
        else:
            return data

class change_pitch(object):
    #This function changes the pitch of the data
    def __init__(self, sr):
        self.sr = sr

    def __call__(self, data):
        rand = np.random.randint(low= 0, high = 4)
        if rand == 0:
            y_pitch = data.copy()
            bins_per_octave = 12
            pitch_pm = 2
            pitch_change = pitch_pm * 2 * (np.random.uniform())
            y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), self.sr, n_steps=pitch_change,
                                                  bins_per_octave=bins_per_octave)
            return y_pitch
        else:
            return data



def create_dataset(create_images):
    '''
    This function creates a dataframe 'target' with columns;
    'path' that stores the path to the image
    'split' that shows whether the data is from train, test or validation
    'label' that states what music genre the case is from
    'label_index' that states the index of the label. i.e. blues genre is equal to zero
    '''
    if create_images == 1:
        path= './genres_original/'
        target = pd.DataFrame(0, index = list(range(0, config.dataset_length)), columns = ['path', 'split', 'label', 'label_index'])
    else:
        path = './images_original/'
        #there an image missing, therefore, one is subtracted from config.dataset_length
        target = pd.DataFrame(0, index = list(range(0, config.dataset_length -1)), columns = ['path', 'split', 'label', 'label_index'])



    images_directories = sorted(os.listdir(path))
    row_index = 0
    #loop through directories in path
    for ind, im_dir in enumerate(images_directories):
        images = sorted(os.listdir(path+ im_dir + '/'))
        #loop through each data point
        for im in images:
            #check if this is equal to the missing/corrupted data point. Otherwise,get the path and the label for the audio
            if (row_index ==config.missing_image) | (row_index ==config.corrupted) :
                target.loc[target.index[row_index], 'path'] = 'missing'
                row_index+=1
            else:
                target.loc[target.index[row_index], 'path'] = path+ im_dir + '/' + im
                target.loc[target.index[row_index], 'label'] = im_dir
                target.loc[target.index[row_index], 'label_index'] = ind
                row_index+=1

    #create column 'split', set all values initially to train
    target['split'] = 'train'
    #shuffle the indexes of the data with random seed 0
    indexes = target.index.values.copy()
    np.random.seed(0)
    np.random.shuffle(indexes)
    #get number of data points for the test set
    proportion = np.floor(len(indexes) * 0.2).astype('int')
    #set test indexes to the last 20% of the data
    test_inds = indexes[-proportion:]
    target = target.reset_index() #reset the index to create a column named 'index'
    target.loc[target['index'].isin(test_inds), 'split'] = 'test' #set the value of split to equal test where the column index is in test indexes
    valid_inds = indexes[-(2*proportion):-proportion] #get validation indexes (second last 20% segment of the data)
    target.loc[target['index'].isin(valid_inds), 'split'] = 'valid' #set the value of split to equal valid where the column index is in valid indexes
    return target



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, choices=['train', 'test'], required=False)
    parser.add_argument('--prefix', type=str, default = None, required=False)
    parser.add_argument('--epochs', type=int, default = 50, required=False)
    parser.add_argument('-c','--create_images', type = int, default = 1, choices = [0,1], help = 'One is to generate images and zero is to use ready-made ones', required=False)
    parser.add_argument('-arc', '--architecture', type=str, default = 'ResNet18', choices  = ['ResNet18','AlexNet', 'VGG16', 'ResNet18_not_pretrained'], required=False)
    parser.add_argument('--lr', type=float, default = 0.00001, required=False)
    parser.add_argument('-aug','--augmentation', type=int, default= 0, choices = [0,1], help = 'One is for augmentation and zero means no data augmentation', required=False)
    parser.add_argument('--model_name', type=str, default = None, required=False, help ='Name and path to model for testing')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_arguments()
    split = args.task
    prefix = args.prefix
    epochs = args.epochs
    create = args.create_images
    arc = args.architecture
    aug = args.augmentation
    lr=args.lr
    model_name = args.model_name

    #Ensure that the name of the model to test is specified if testing the model
    if (split == 'test') & (model_name == None):
        print('Please specify the "model_name" parameter')
        sys.exit()

    #Ensure a prefix for the model is specified if training the model
    if (split == 'train') & (prefix == None):
        print('Please specify the "prefix" parameter')
        sys.exit()

    #split data into train, validation and test. Information is stored in target
    target = create_dataset(create)

    #Create augmentor if creating the images and implementing augmentation.
    if (create ==1) & (aug ==1) :
        augmentor = transforms.Compose([
        Noise(),
        change_speed(),
        Shift(),
        change_pitch(22050),
        hpss()
        ] )
    else:
        augmentor = None

    #Initialise the model
    import model
    if arc == 'ResNet18':
        model = model.ResNet18()
    elif arc == 'AlexNet':
         model = model.AlexNet()
    elif arc == 'VGG16':
         model = model.VGG16()
    elif arc == 'ResNet18_not_pretrained':
         model = model.ResNet18_not_pre()

    #If training, initialise train and validation loaders, otherwise initialise test loader
    if split == 'train':
        train_dataset = Dataset(target, split, create, augmentor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
        valid_dataset = Dataset(target, 'valid', create)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=False)
        #begin training
        train(model, epochs, lr, train_loader, valid_loader, prefix)
    else:
        test_dataset = Dataset(target, split, create)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        #begin testing
        test(args.model_name, test_loader)
