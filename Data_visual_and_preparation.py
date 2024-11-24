import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import os
import kagglehub
import json


# Download latest version
data_path = kagglehub.dataset_download("debarshichanda/goemotions")
print("Path to dataset files:", data_path)
with open('config.json') as conf:
    data = json.load(conf)
data_path = data.get('Original_Dataset', None)

dataset_folder_path = 'data_folder'
figures_folder_path = 'Figures'
if not os.path.exists(dataset_folder_path):
    os.makedirs(dataset_folder_path)
if not os.path.exists(figures_folder_path):
    os.makedirs(figures_folder_path)

print('\n--------------Reading Dataset--------------')
df = pd.read_csv(f'{data_path}/data/train.tsv', sep='\t', names=['sample', 'label', 'id']).drop(columns='id')
df_val = pd.read_csv(f'{data_path}/data/dev.tsv', sep='\t', names=['sample', 'label', 'id']).drop(columns='id')
df_test = pd.read_csv(f'{data_path}/data/test.tsv', sep='\t', names=['sample', 'label', 'id']).drop(columns='id')

# Function to convert a string of comma-separated numbers into a list of integers
def convert_to_int_list(label):
    if isinstance(label, str):
        # Split the string by commas and convert each element to an integer
        return list(map(int, label.split(',')))
    else:
        # If it's already a list of integers, return it as is
        return label
# Apply the conversion function to the 'label' column
df['label'] = df['label'].apply(convert_to_int_list)
df_val['label'] = df_val['label'].apply(convert_to_int_list)
df_test['label'] = df_test['label'].apply(convert_to_int_list)

print(f"\n\n--------------Dataset Info--------------\n\n{df.info()}")
print(f"\n\n--------------Dataset Description--------------\n\n{df.describe()}")

print(f"\n\n--------------Dataset Figures--------------\n\n")

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]
num_emotions = len(emotion_labels)

train_emotions = []
val_emotions = []
test_emotions = []

# Calculate percentages for each emotion in train, validation, and test datasets
for e in range(num_emotions):
    # Count the occurrences of emotion 'e' in each dataset0
    train_count = sum([e in labels for labels in df['label']])
    val_count = sum([e in labels for labels in df_val['label']])
    test_count = sum([e in labels for labels in df_test['label']])
    train_emotions += [100 * train_count / len(df)]
    val_emotions += [100 * val_count / len(df_val)]
    test_emotions += [100 * test_count / len(df_test)]

# Create a histogram with three columns
x = np.arange(num_emotions)  # Emotion indices
width = 0.25  # Width of each bar

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
datasets = {
    "Train": train_emotions,
    "Validation": val_emotions,
    "Test": test_emotions
}

colors = ['blue', 'green', 'orange']


for ax, (dataset_name, emotions), color in zip(axes, datasets.items(), colors):
    ax.bar(np.arange(num_emotions), emotions, color=color, alpha=0.7)
    ax.set_title(f"{dataset_name} Emotion Distribution")
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.xticks(np.arange(num_emotions), emotion_labels, rotation=45)
plt.xlabel("Emotion")
plt.tight_layout()
plt.savefig(f'{figures_folder_path}/Emotions Distribution.png')
# plt.show()
plt.clf()


train_len = df['sample'].apply(len)
val_len = df_val['sample'].apply(len)
test_len = df_test['sample'].apply(len)

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Create histograms for each dataset
axes[0].hist(train_len, bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Sample Length Distribution - Train Dataset', fontsize=14)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

axes[1].hist(val_len, bins=30, color='green', edgecolor='black')
axes[1].set_title('Sample Length Distribution - Validation Dataset', fontsize=14)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

axes[2].hist(test_len, bins=30, color='orange', edgecolor='black')
axes[2].set_title('Sample Length Distribution - Test Dataset', fontsize=14)
axes[2].set_xlabel('Length of Sample', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].grid(axis='y', linestyle='--', alpha=0.6)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save and show the plot
plt.savefig(f'{figures_folder_path}/Sample Length Distribution.png')
# plt.show()
plt.clf()

def count_words(sample):
    return len(sample.split())

train_num_words = df['sample'].apply(count_words)
val_len_num_words = df_val['sample'].apply(count_words)
test_len_num_words = df_test['sample'].apply(count_words)

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Create histograms for each dataset
axes[0].hist(train_num_words, bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Sample Word Number Distribution - Train Dataset', fontsize=14)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

axes[1].hist(val_len_num_words, bins=30, color='green', edgecolor='black')
axes[1].set_title('Sample Word Number Distribution - Validation Dataset', fontsize=14)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

axes[2].hist(test_len_num_words, bins=30, color='orange', edgecolor='black')
axes[2].set_title('Sample Word Number Distribution - Test Dataset', fontsize=14)
axes[2].set_xlabel('Length of Sample', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].grid(axis='y', linestyle='--', alpha=0.6)

# Adjust layout to avoid overlap
plt.tight_layout()


print(f"\n\n--------------Cleaning Special Chars--------------\n\n")
# TODO: remove special chars


# Save and show the plot
plt.savefig(f'{figures_folder_path}/Sample Word Number Distribution.png')
# plt.show()
plt.clf()

print('--------------Saving Data--------------')
# TODO: save
# df.to_csv(f'{data_folder_path}/Clean_prepared_dataset')

print('--------------Encoding--------------')
# TODO: encode

print('--------------Saving Encoded Datasets--------------')
# TODO: save

print('--------------DONE--------------')
