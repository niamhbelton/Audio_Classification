
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


def evaluate(model, generator):
    """Evaluate

    Args:
      model: object.
      generator: object.

    Returns:
      accuracy: float
    """
    y_trues=[]
    y_preds=[]
    loss_sum=0
    # Get data in batches
    for i,data in enumerate(generator):
        batch_x =data[0]
        batch_y=data[1]

        model.eval()
        # Forward
        if torch.cuda.is_available():
            output = model(batch_x.squeeze(0).cuda())
        else:
            output = model(batch_x.squeeze(0))
        
        #calculate loss
        if torch.cuda.is_available():
            loss = F.nll_loss(output, batch_y.cuda())
        else:
            loss = F.nll_loss(output, batch_y)
                
        loss_sum+= loss.item()
        target = torch.max(output, dim =1)[1]
        y_trues = y_trues + (batch_y.detach().cpu().numpy().tolist())
        y_preds = y_preds + (target.detach().cpu().numpy().tolist())
 
    return accuracy_score(y_trues, y_preds), loss_sum





def train(model, epochs, lr, train_loader, valid_loader, prefix):
    """Train

    Args:
      model: object.
      epochs: int.
      lr: float.
      train_loader: object.
      tval_loader: object.
      prefix: string

    """
    
    #set model to train mode
    model.train()
    
    #put on GPU if there is one available
    if torch.cuda.is_available():
        model.cuda()
   
    #initialise optimiser and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=.1, threshold=1e-4, verbose=True)


    #Create directories 'models' and 'logs'
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    if not os.path.exists('./logs/'):
        os.makedirs('./logs/')
        

    
    early_trigger = 8 #this variable will help stop the training process if there has been no increase in validation accuracy in 8 rounds
    early_iterations =0 #this variable counts the number of iterations with no increase in validation loss
    best_acc = 0
    loss_epoch = []
    va_losses=[]
    start_time = time.time()
    #for loop for each epoch
    for epoch in range(0, epochs):
        y_preds=[]
        y_trues=[]
        loss_sum=0
        #loop through batches of data
        for i,data in enumerate(train_loader): 
            batch_x =data[0]
            batch_y=data[1]
         
            #forward
            if torch.cuda.is_available():
                output = model(batch_x.squeeze(0).cuda())
            else:
                output = model(batch_x.squeeze(0))
            
            #get loss
            if torch.cuda.is_available():
                loss = F.nll_loss(output, batch_y.cuda())
            else:
                loss = F.nll_loss(output, batch_y)
            
            loss_sum+=loss.item()
          
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      
            target = torch.max(output, dim =1)[1] #get class with highest output
            y_trues = y_trues + (batch_y.detach().cpu().numpy().tolist())
            y_preds = y_preds + (target.detach().cpu().numpy().tolist())

            
        loss_epoch.append(loss_sum) 
        train_acc = accuracy_score(y_trues, y_preds) #get train accuracy
        print('Finished epoch {}. Train_accuracy is {}%'.format(epoch, np.round(train_acc*100, 2)))
        
        va_acc, va_loss = evaluate(model, valid_loader) #get accuracy on validation set 
        va_losses.append(va_loss)
        scheduler.step(va_loss)
        print('Validation_accuracy is {}%'.format( np.round(va_acc*100, 2)))
        print('Training time is {} minutes'.format(np.round((time.time() - start_time) / 60), 2))
        
        #if this is the best accuracy so far, save the model and remove earlier saved models that have the same prefix
        if va_acc > best_acc:
            best_acc = va_acc
            model_name = 'model_' + prefix + '_epoch_' + str(epoch) + '_train_acc_' + str(np.round(train_acc*100, 2)) + '%_valid_acc_' + str(np.round(va_acc*100,2)) +'%'
            for f in os.listdir('./models/'):
                to_remove = '_' + prefix + '_'
                if (to_remove in f)  :
                    os.remove(f'./models/{f}')
            torch.save(model, './models/'+model_name)
            early_iterations = 0
        else:
            #if there is no increase in validation accuracy, increment the early_iterations variable and check if the value is greater than early_trigger. If so, stop the trainnig and write out the train and validation loss
            early_iterations+=1
            if early_iterations > early_trigger:
                print('Early stopping after ' + str(epoch) + ' epochs.')
                pd.DataFrame(loss_epoch).to_csv('./logs/losses_train_model_{}_epoch_{}.csv'.format(prefix, epoch))
                pd.DataFrame(va_losses).to_csv('./logs/losses_valid_model_{}_epoch_{}.csv'.format(prefix, epoch))
                print('Finished training.')
                sys.exit()
        
        #Every five iterations write out the losses
        if (epoch % 5 == 0) | (epoch == epochs -1):
            pd.DataFrame(loss_epoch).to_csv('./logs/losses_train_model_{}_epoch_{}.csv'.format(prefix, epoch))
            pd.DataFrame(va_losses).to_csv('./logs/losses_valid_model_{}_epoch_{}.csv'.format(prefix, epoch))
            
        
    print('Finished training.')
        
        




