import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import fetch_california_housing
import os
import kagglehub
import json
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

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

# Cleans the string by converting letters to lowercase, removing unwanted characters,
# and replacing newline characters with spaces
df['sample'] = df['sample'].apply(cleaning_strings)
df_val['sample'] = df_val['sample'].apply(cleaning_strings)
df_test['sample'] = df_test['sample'].apply(cleaning_strings)

# Function to convert a string of comma-separated numbers into a list of integers
def convert_to_int_list(label):
    if isinstance(label, str):
        # Split the string by commas and convert each element to an integer
        return list(map(int, label.split(',')))
    else:
        # If it's already a list of integers, return it as is
        return label
# Apply the conversion function to the 'label' column
df['label'] = df['label'].apply(convert_to_int_list)  # =>>>>>>>>
df_val['label'] = df_val['label'].apply(convert_to_int_list)
df_test['label'] = df_test['label'].apply(convert_to_int_list)

print(f"\n\n--------------Dataset Info--------------\n\n{df.info()}")
print(f"\n\n--------------Dataset Description--------------\n\n{df.describe()}")

print(f"\n\n--------------Cleaning Special Chars--------------\n\n")
def cleaning_strings(s: str):
    # Replace newlines with spaces
    s = s.replace('\n', ' ')
    # Using list comprehension to filter out non-letter, non-number, non-space characters,
    # and convert letters to lowercase
    return ''.join(
        [char.lower() if char.isalpha() else char for char in s if char.isalpha() or char.isdigit() or char == ' '])

df['sample'] = df['sample'].apply(cleaning_strings)
df_val['sample'] = df_val['sample'].apply(cleaning_strings)
df_test['sample'] = df_test['sample'].apply(cleaning_strings)


print(f"\n\n--------------Dataset Figures--------------\n\n")

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]
num_emotions = len(emotion_labels)
mlb = MultiLabelBinarizer(classes=range(num_emotions))


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

# Save and show the plot
plt.savefig(f'{figures_folder_path}/Sample Word Number Distribution.png')
# plt.show()
plt.clf()

print('--------------Saving Data--------------')
df.to_csv(f'{dataset_folder_path}/train_prep_with_labels.csv', index=False)
df_val.to_csv(f'{dataset_folder_path}/val_prep_with_labels.csv', index=False)
df_test.to_csv(f'{dataset_folder_path}/test_prep_with_labels.csv', index=False)

compute_embed = True
def tokenize_sentence(sentence):
    return tokenizer.tokenize(sentence)
def get_word_embeddings(tokens):
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0].numpy()

if compute_embed:
    print('--------------Encoding--------------')
    train_sentences = df['sample'].tolist()
    val_sentences = df_val['sample'].tolist()
    test_sentences = df_test['sample'].tolist()

    train_tokens = [tokenize_sentence(sentence) for sentence in train_sentences]
    val_tokens = [tokenize_sentence(sentence) for sentence in val_sentences]
    test_tokens = [tokenize_sentence(sentence) for sentence in test_sentences]

    train_word_embeddings = [get_word_embeddings(tokens) for tokens in train_tokens]
    val_word_embeddings = [get_word_embeddings(tokens) for tokens in val_tokens]
    test_word_embeddings = [get_word_embeddings(tokens) for tokens in test_tokens]

    train_embeddings = model.encode_sentences(train_sentences, combine_strategy="mean")
    print(f'Train Done\nShaped: {train_embeddings.shape}')
    val_embeddings = model.encode_sentences(val_sentences, combine_strategy="mean")
    print('Val Done')
    test_embeddings = model.encode_sentences(test_sentences, combine_strategy="mean")
    print('Test Done')

    mlb = MultiLabelBinarizer(classes=range(len(emotion_labels)))
    train_one_hot_labels = mlb.fit_transform(df['label'].tolist()[:100])
    val_one_hot_labels = mlb.fit_transform(df['label'].tolist()[:50])
    test_one_hot_labels = mlb.fit_transform(df['label'].tolist()[:50])

    train_data = np.hstack((train_embeddings, train_one_hot_labels))
    val_data = np.hstack((val_embeddings, val_one_hot_labels))
    test_data = np.hstack((test_embeddings, test_one_hot_labels))

    print("Train Embeddings Shape:", train_embeddings.shape)
    print("Validation Embeddings Shape:", val_embeddings.shape)
    print("Test Embeddings Shape:", test_embeddings.shape)

    print('--------------Saving Encoded Datasets--------------')
    np.save(f'{dataset_folder_path}/train_prep_with_labels.npy', train_data)
    np.save(f'{dataset_folder_path}/val_prep_with_labels.npy', val_data)
    np.save(f'{dataset_folder_path}/test_prep_with_labels.npy', test_data)
else:
    train_data = np.load(f'{dataset_folder_path}/train_prep_with_labels.npy')
    val_data = np.load(f'{dataset_folder_path}/val_prep_with_labels.npy')
    test_data = np.load(f'{dataset_folder_path}/test_prep_with_labels.npy')
print('--------------DONE--------------')




