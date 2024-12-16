import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import kagglehub
import json
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from keras_preprocessing.sequence import pad_sequences


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Download the latest version
data_path = kagglehub.dataset_download("debarshichanda/goemotions")
print("Path to dataset files:", data_path)
with open('config.json') as conf:
    c = json.load(conf)
try:
    data_path = c.get('Original_Dataset', None)
except Exception as e:
    print(f'Update the Original_Dataset field in config.json to be:\n{data_path}\nThen run this script again.')
    exit(1)
dataset_folder_path = 'data_folder'
figures_folder_path = 'Figures'
if not os.path.exists(dataset_folder_path):
    os.makedirs(dataset_folder_path)
if not os.path.exists(f'{dataset_folder_path}/TFidf'):
    os.makedirs(f'{dataset_folder_path}/TFidf')
if not os.path.exists(f'{dataset_folder_path}/NN'):
    os.makedirs(f'{dataset_folder_path}/NN')
if not os.path.exists(f'{dataset_folder_path}/Transformer'):
    os.makedirs(f'{dataset_folder_path}/Transformer')


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

print(f"\n\n--------------Train Info--------------\n\n{df.info()}")
print(f"\n\n--------------Train Description--------------\n\n{df.describe()}")
print(f"\n\n--------------Dev Info--------------\n\n{df_val.info()}")
print(f"\n\n--------------Dev Description--------------\n\n{df_val.describe()}")
print(f"\n\n--------------Test Info--------------\n\n{df_test.info()}")
print(f"\n\n--------------Test Description--------------\n\n{df_test.describe()}")

print(f'The full dataset contains {len(df)+len(df_val)+len(df_test)} samples.')


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
df.to_csv(f'{dataset_folder_path}/TFidf/train_prep_with_labels.csv', index=False)
df_val.to_csv(f'{dataset_folder_path}/TFidf/val_prep_with_labels.csv', index=False)
df_test.to_csv(f'{dataset_folder_path}/TFidf/test_prep_with_labels.csv', index=False)

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
    # TODO: run on the full dataset- remove the [:5/15] filters
    train_sentences = df['sample'].tolist()[:15]
    val_sentences = df_val['sample'].tolist()[:5]
    test_sentences = df_test['sample'].tolist()[:5]

    train_tokens = [tokenize_sentence(sentence) for sentence in train_sentences]
    val_tokens = [tokenize_sentence(sentence) for sentence in val_sentences]
    test_tokens = [tokenize_sentence(sentence) for sentence in test_sentences]


    train_embeddings = [get_word_embeddings(tokens) for tokens in train_tokens]
    val_embeddings = [get_word_embeddings(tokens) for tokens in val_tokens]
    test_embeddings = [get_word_embeddings(tokens) for tokens in test_tokens]

    # TODO: flat_and_pad make sure the dims fit
    # def flat_and_pad(list_embeds, target_len=max_word_count, single_embd_len=len(train_embeddings[0][0])):
    #     ans = np.zeros((len(list_embeds), target_len, single_embd_len), dtype=np.float32)
    #     arr = np.array(list_embeds)
    #     ans[:min(target_len, len(arr)), :] = arr[:target_len, :]
    #     return ans.flatten()

    max_num_tokens = max([max(len(item) for item in train_embeddings), max(len(item) for item in val_embeddings), max(len(item) for item in test_embeddings)])


    def flat_and_pad(list_embeds, target_len=max_num_tokens, single_embd_len=len(train_embeddings[0][0])):
        # target_len: Desired number of rows per element
        # single_embd_len: Desired embedding size (columns)

        flattened_list = []

        for embed in list_embeds:
            # Initialize a zero-padded array
            padded_embed = np.zeros((target_len, single_embd_len), dtype=np.float32)

            # Get the actual shape of the current embedding
            rows_to_copy, cols_to_copy = embed.shape

            # Copy the values to the padded array
            padded_embed[:rows_to_copy, :cols_to_copy] = embed

            # Flatten the padded array and append to the result list
            flattened_list.append(padded_embed.flatten())

        return np.array(flattened_list)


    train_embeddings_padded = flat_and_pad(train_embeddings)
    val_embeddings_padded = flat_and_pad(val_embeddings)
    test_embeddings_padded = flat_and_pad(test_embeddings)


    mlb = MultiLabelBinarizer(classes=range(len(emotion_labels)))
    train_one_hot_labels = mlb.fit_transform(df['label'].tolist()[:15])
    val_one_hot_labels = mlb.fit_transform(df['label'].tolist()[:5])
    test_one_hot_labels = mlb.fit_transform(df['label'].tolist()[:5])

    train_data_padded = (train_embeddings_padded, train_one_hot_labels)
    val_data_padded = (val_embeddings_padded, val_one_hot_labels)
    test_data_padded = (test_embeddings_padded, test_one_hot_labels)

    train_data = (train_embeddings, train_one_hot_labels)
    val_data = (val_embeddings, val_one_hot_labels)
    test_data = (test_embeddings, test_one_hot_labels)

    print('--------------Saving Encoded Datasets--------------')

    with open(f'{dataset_folder_path}/NN/train_prep_ped_with_labels.pkl', 'wb') as f:
        pickle.dump(train_data_padded, f)
    with open(f'{dataset_folder_path}/NN/val_prep_ped_with_labels.pkl', 'wb') as f:
        pickle.dump(val_data_padded, f)
    with open(f'{dataset_folder_path}/NN/test_prep_ped_with_labels.pkl', 'wb') as f:
        pickle.dump(test_data_padded, f)

    with open(f'{dataset_folder_path}/Transformer/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{dataset_folder_path}/Transformer/val_data_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(f'{dataset_folder_path}/Transformer/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
print('--------------DONE--------------')
