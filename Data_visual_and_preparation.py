import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data_folder_path = 'data_folder'
df = pd.read_csv(f'{data_folder_path}/train.csv')
print(f'The Boston Dataset have: {len(df.iloc[0])} parameters.')

plt.figure(figsize=(10, 6))
plt.hist(df['SalePrice'], bins=100, color='blue', alpha=0.7)
plt.title('Distribution of House Prices', fontsize=16)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

i = df.info()
d = df.describe()
print(f"Dataset Info:\n{df.info()}")
print(f"Dataset Description:\n{df.describe()}")
