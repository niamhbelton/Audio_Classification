# Audio_Classification


## 1.	Data
The GTZan dataset is available for download at https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification. It consists of 1000 data points belonging to ten genres.

## 2.	Experiment 1
The code for experiment 1 can be found in mlp.ipynb.

## 3.	Experiments 2-4
The code in the Exp2-3 folder consists of 'main.py', 'train.py', 'dataloader.py', 'model.py', 'config.py' and 'evaluate.py'. train.py is responsible for training the model and evaluate.py is responsible for testing the model. These scripts are called from the main.py script. The requirements for this code are available in the .txt file. 

Arguments for ‘main.py’ are shown below.

 
•	‘task’ – This is a required argument that specifies whether the model is being trained or tested.
•	‘prefix’ – This is the name of the model. This is not needed when testing the model. Therefore, it is not a required argument. However, if the model is being trained and the prefix is not specified, a string will be printed out that states that the prefix argument is required and the training will be aborted. 
•	‘epochs’ – The number of epochs.
•	‘create_images’ – This is a binary value where one generate images of Mel-Spectrograms and zero uses the ready-made images provided in the dataset.
•	‘architecture’ – This specifies the architecture of the model.
•	‘lr’ – This specifies the learning rate of the model.
•	‘augmentation’ – This is a binary value where one augments the images and zero does not. This is only used when we are generating the images (rather than using the ready-made ones)
•	‘model_name’ – This is not needed when training the model but it is a required argument for testing the model as this specifies the path and name of the model to test. if the model is being tested and the model_name is not specified, a string will be printed out that states that the model_name argument is required and the testing will be aborted. 

Additional Information
•	Training the model will create two directories ‘logs’ and ‘models’.
•	It records the losses every five iterations in the log directory.
•	It saves the model with best validation accuracy in the models directory and it removes any other saved models in this directory with the same prefix/name. 
•	This script will run on the GPU if there is one available.
•	Decrease batch size to 8 if training a model with VGG16 architecture to avoid out of memory errors.




Running the code
•	Running this code assumes that all scripts are in your current working directory, along with the unzipped data. For example, you current working directory would have the folder ‘genres_original’, ‘images_original’, the csv ‘features_30_sec.csv’ and all scripts.

Experiment 2 Commands:
python main.py -t 'train' --epochs 150 --prefix exp2_res -c 0 -arc 'ResNet18'
python main.py -t 'train' --epochs 150 --prefix exp2_alex -c 0 -arc 'AlexNet'

Experiment 3 Commands:
python main.py -t 'train' --epochs 150 --prefix exp3_res -arc 'ResNet18'
python main.py -t 'train' --epochs 150 --prefix exp3_alex -arc 'AlexNet'

Experiment 4 Commands:
python main.py -t 'train' --epochs 150 --prefix exp4 -c 1 -arc 'ResNet18' --lr 0.0001 -aug 1

Testing command:
python main.py -t 'test' --model_name './models/model_exp4_epoch_22_train_acc_94.67%_valid_acc_87.5%'

Note: ‘model_exp4_epoch_22_train_acc_94.67%_valid_acc_87.5%’ is was selected as the final model.


