import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import os

data_folder_path = 'data_folder'
figures_folder_path = 'Figures'
if not os.path.exists(figures_folder_path):
    os.makedirs(figures_folder_path)
print('--------------Reading Dataset--------------')
df = pd.read_csv(f'{data_folder_path}/train.csv')
print(f'The Boston Dataset have: {len(df.iloc[0])} parameters.')

plt.figure(figsize=(10, 6))
plt.hist(df['SalePrice'], bins=100, color='blue', alpha=0.7)
plt.title('Distribution of House Prices', fontsize=16)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f'{figures_folder_path}/Price Distribution.png')

i = df.info()
d = df.describe()
print(f"--------------Dataset Info--------------\n{df.info()}")
print(f"--------------Dataset Description--------------\n{df.describe()}")

print('--------------Cleaning Data--------------')
# TODO: Roni, eliminate string features.
#  Use one-hot if necessary but try to quantify where it make sense (but don't do it objectively)
#  Use the data_description.txt file.
print('--------------Normalizing The Data--------------')
# TODO: Yakir, after Roni is done normalize the features that should be normalized (everything that isn't bound in 0-1, dont touch one-hot features)

# TODO: Shahaf, after Yakir is done get histograms of some variables
#  Run TSNE, PCA, and other visualizations, ged std, mean, median vectors.

print('--------------Saving Data--------------')
df.to_csv(f'{data_folder_path}/Clean_prepared_dataset')
print('--------------DONE--------------')
