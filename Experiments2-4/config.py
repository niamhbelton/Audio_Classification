import pandas as pd
import os 
data = pd.read_csv('features_30_sec.csv')
dataset_length = len(data)

labels = sorted(os.listdir('./genres_original/'))

missing_image = 544

num_labels = 10

corrupted = 554
