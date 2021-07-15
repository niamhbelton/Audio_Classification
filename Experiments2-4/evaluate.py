
import model
import config
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys
import pandas as pd 
import numpy as np
import config

def test(model_name, generator):
    """Test

    Args:
      model_name: string.
      generator: object.


    """
    
    #load model
    model = torch.load(model_name)
    _ = model.eval() #set to evaluation
    if torch.cuda.is_available(): #if there is a GPU available, load the model to GPU
        model.cuda()
    y_trues=[]
    y_preds=[]
    # iterate through test data
    for i,data in enumerate(generator):
        batch_x =data[0]
        batch_y=data[1]
        
        #forward
        if torch.cuda.is_available():
            output = model(batch_x.squeeze(0).cuda())
        else:
            output = model(batch_x.squeeze(0))
            
        #get class with with highest output
        target = torch.max(output, dim =1)[1]
        y_trues = y_trues + (batch_y.detach().cpu().numpy().tolist())
        y_preds = y_preds + (target.detach().cpu().numpy().tolist())
  

    #print out the classification rate for each class
    df = pd.concat([pd.DataFrame(y_trues), pd.DataFrame(y_preds)], axis =1)
    df.columns= ['actual', 'pred']
    data = ['Class','Fraction', 'Classification Rate']
    print("{:25} {:25} {:25} ".format(*data))
    for i in range(0, config.num_labels):
 
        class_rate =str(len(df.loc[(df['actual'] == i) & (df['pred'] == i )] )) + '/' + str(len(df.loc[(df['actual'] == i) ] ))
        class_perc = str(np.round((len(df.loc[(df['actual'] == i) & (df['pred'] == i )] ) / len(df.loc[(df['actual'] == i) ] )) * 100, 2)) + '%'
        data = [config.labels[i],class_rate, class_perc]
        print("{:25} {:25} {:25} ".format(*data))
    print('Accuracy on test set is {}%'.format(np.round(accuracy_score(y_trues, y_preds) * 100, 2)))

