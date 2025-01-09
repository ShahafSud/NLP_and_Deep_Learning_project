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

import nltk

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

global conversion_dictionary
conversion_dictionary = {
    # Removing non-English words:
    '22meirl4meirl42meirl4meirl': '',
    'sノ': '',
    'ノ': '',
    '카니발': '',

    # Consolidating numbers into a NUM token:
    '4k': 'NUM',
    '16k': 'NUM',
    '3': 'NUM',
    '4': 'NUM',
    '9k': 'NUM',
    '41': 'NUM',
    '390': 'NUM',
    '100': 'NUM',
    '15': 'NUM',
    '8': 'NUM',
    '30': 'NUM',
    '000': 'NUM',
    '5': 'NUM',
    '09': 'NUM',
    '1': 'NUM',
    '23rd': 'NUM',
    '0': 'NUM',
    '1122': 'NUM',
    '7k': 'NUM',
    '250': 'NUM',
    '2002': 'NUM',
    '11': 'NUM',
    '10': 'NUM',
    '7': 'NUM',
    '2019': 'NUM',
    '70': 'NUM',
    '1900': 'NUM',
    '10k': 'NUM',
    '9': 'NUM',
    '1970': 'NUM',
    '2': 'NUM',
    '6': 'NUM',
    '2014': 'NUM',
    '32': 'NUM',
    '1111': 'NUM',
    '100k': 'NUM',
    '59': 'NUM',
    '13': 'NUM',
    '003': 'NUM',
    '280': 'NUM',
    '2005': 'NUM',
    '50': 'NUM',
    '2018': 'NUM',
    '799': 'NUM',
    '190': 'NUM',
    '140': 'NUM',
    '12': 'NUM',
    '25': 'NUM',
    '20': 'NUM',
    '2013': 'NUM',
    '2016': 'NUM',
    '22': 'NUM',
    '18': 'NUM',
    '16': 'NUM',
    '40': 'NUM',
    '60': 'NUM',
    '28': 'NUM',
    '61': 'NUM',
    '27': 'NUM',
    '34': 'NUM',
    '2015': 'NUM',
    '00': 'NUM',
    '26': 'NUM',
    '80': 'NUM',
    '1940': 'NUM',
    '23': 'NUM',
    '99': 'NUM',
    '1954': 'NUM',
    '66': 'NUM',
    '88': 'NUM',
    '49': 'NUM',
    '29': 'NUM',
    '17': 'NUM',
    '2028': 'NUM',
    '696': 'NUM',
    '5933': 'NUM',
    '43': 'NUM',
    '125': 'NUM',
    '30000': 'NUM',
    '2007': 'NUM',
    '2020': 'NUM',
    '615': 'NUM',
    '741': 'NUM',
    '3491': 'NUM',
    '87': 'NUM',
    '06': 'NUM',
    '500': 'NUM',

    # Removing repeating letters:
    # 'haha'
    'hahahaha': 'haha',
    'hahhaha': 'haha',
    'hahahahahaha': 'haha',
    'hahahahshshajsha': 'haha',
    'hahahaahhhahahahahahaaaaaa': 'haha',
    'hahahahahahahahaha': 'haha',
    'hahahahahahaha': 'haha',
    'hhahah': 'haha',
    'hahahahah': 'haha',
    'hahahahgaha': 'haha',
    'hahahahahahahahahahahahaha': 'haha',
    'ahahahhahahahahahaha': 'haha',
    'hahahahahahahahahahahahahahahahahahahahahahahahahahhahahahhahahaa': 'haha',
    'hahahah': 'haha',
    'ahahhahahaha': 'haha',
    'ahaha': 'haha',
    'hahaha': 'haha',
    'hahah': 'haha',

    # 'bahaha'
    'bahahaha': 'bahaha',
    'bahahahahahaha': 'bahaha',
    'bahahahahaha': 'bahaha',
    'bhahaha': 'bahaha',
    'paahahaha': 'bahaha',
    'bhahahahahahahahah': 'bahaha',

    # 'fuck'
    'fuuuuck': 'fuck',
    'fuuuck': 'fuck',
    'fuckkkkk': 'fuck',
    'fuuuuuuuuuck': 'fuck',
    'fuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuck': 'fuck',
    'fuuuuuck': 'fuck',

    # 'no'
    'noooo': 'no',
    'nooooooooooo': 'no',
    'nooooo': 'no',
    'nooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo': 'no',
    'nooo': 'no',

    # 'me'
    'meeee': 'me',
    'mmeeeee': 'me',
    'meeeee': 'me',

    # 'shh'
    'shhh': 'shh',
    'shhhh': 'shh',

    # 'ah'
    'ahhh': 'ah',
    'aaaaaaaaaaaaaahhh': 'ah',
    'ahhhh': 'ah',
    'aaahhh': 'ah',
    'ahahahahahaha': 'ah',

    # 'what'
    'whaaaaaaat': 'what',
    'whhhattttt': 'what',

    # 'uh'
    'uhhhhhhhhh': 'uh',
    'uuuuh': 'uh',
    'uhhhhhhh': 'uh',

    # 'ayyy'
    'ayyyyyyyyy': 'ayyy',
    'ayyy': 'ayyy',

    # 'and'
    'aaaaand': 'and',
    'aaaaaaaaaaaaand': 'and',
    'aaaaaand': 'and',
    'aaaand': 'and',

    # 'boo'
    'boooooooo': 'boo',
    'booo': 'boo',
    'boooo': 'boo',

    # 'so'
    'soooooo': 'so',
    'soooo': 'so',
    'sooooo': 'so',
    'soooooooooooooooooooooooooooooooooooooooooooooooooo': 'so',
    'sooo': 'so',

    # 'hmm'
    'hmmmhmmm': 'hmm',
    'hmmmm': 'hmm',
    'hmmmmm': 'hmm',
    'hmmm': 'hmm',

    # 'go'
    'goooo': 'go',
    'gooooo': 'go',

    # 'good'
    'gooooood': 'good',
    'gooooooooood': 'good',

    'bitchhhhh': 'bitch',
    'byyiiitttcchhh': 'bitch',

    # 'mmm'
    'mmmm': 'mmm',
    'mmmmm': 'mmm',
    'mmm': 'mmm',
    'mmmmmm': 'mmm',

    # 'oh'
    'ooohh': 'oh',
    'ohhhh': 'oh',
    'ohhhhh': 'oh',
    'ooooh': 'oh',
    'ohhh': 'oh',
    'ooooooh': 'oh',
    'oooooooh': 'oh',

    # 'yes'
    'yeeesss': 'yes',
    'yaaaaaaas': 'yes',
    'yessss': 'yes',
    'yesssss': 'yes',

    # 'oof'
    'ooooof': 'oof',
    'ooof': 'oof',

    # 'hey'
    'heyyy': 'hey',
    'haaaaay': 'hey',

    # 'lmao'
    'lmaoooo': 'lmao',
    'lmaoooooooooooooooooooo': 'lmao',

    # 'whooshed'
    'woooshed': 'whooshed',
    'wooooshed': 'whooshed',

    # 'wow'
    'wooow': 'wow',
    'woooooooahahhohohohohho': 'wow',

    # 'you'
    'youuuuu': 'you',
    'yoooouuuuu': 'you',
    'yooooooou': 'you',

    # 'whoosh'
    'woooosh': 'whoosh',
    'wooosh': 'whoosh',

    # 'all'
    'allllll': 'all',
    'aalll': 'all',
    'allll': 'all',
    'alll': 'all',

    # 'wha'
    'whaaa': 'wha',
    'whhhaaa': 'wha',

    # 'love'
    'loooooove': 'love',
    'looooooove': 'love',

    # 'ooo'
    'oooooooo': 'ooo',
    'oooo': 'ooo',
    'ooooo': 'ooo',

    # 'down'
    'downnnn': 'down',
    'dooooooown': 'down',

    # 'yas'
    'yaaaaaas': 'yas',
    'yaaassss': 'yas',

    'wahooooo': 'wahoo',
    'loooooool': 'lol',
    'jeeem': 'jeem',
    'iittssss': 'its',
    'highhhhh': 'high',
    'noooooooonnn': 'non',
    'ooooonnn': 'on',
    'rooosaa': 'rosa',
    'yuuuuuuuuuup': 'yup',
    'hoooome': 'home',
    'cherrrryyyyy': 'cherry',
    'foooooooor': 'for',
    'ooook': 'ok',
    'theeeeeese': 'these',
    'boiii': 'boi',
    'yeaaahhh': 'yeah',
    'wheeere': 'where',
    'laaaaaand': 'land',
    'shuuuuut': 'shut',
    'goddddd': 'god',
    'gaaahhh': 'gah',
    'thankkks': 'thanks',
    'hhhhhh': 'hhhh',
    'yaaaay': 'yay',
    'riiiicchhh': 'rich',
    'cuteee': 'cute',
    'sstaaaaahhhhpppp': 'stop',
    'sliiiiick': 'slick',
    'n00b': 'noob',
    'saaaffffffeeeee': 'safe',
    'bruhhh': 'bruh',
    'neeeeeews': 'news',
    'sendddinggg': 'sending',
    'yoooooooooooooooooooooooooooooooooooooooooo': 'yo',
    'wooooshiety': 'wooshiety',
    'himmmmmmmmmmmmmm': 'him',
    'hellooo': 'hello',
    'draaaaaaged': 'dragged',
    'byyy': 'by',
    'stoooooone': 'stone',
    'leaveeee': 'leave',
    'wooooo': 'woo',
    'shockeeeeeeeeeeeer': 'shocker',
    'whyyyyy': 'why',
    'ughhh': 'ugh',
    'soooooooooooooooooooooooooooooooooooooooooon': 'soon',
    'oooops': 'oops',
    'ohiooooooooo': 'ohio',
    'thissss': 'this',
    'muhahahah': 'muhaha',
    'oooooooceans': 'oceans',
    'waaaaaaaaves': 'waves',
    'whoooo': 'whoo',
    'smoooootthhhhhhh': 'smooth',
    'oldddd': 'old',
    'hhhhhaaarrryyy': 'harry',
    'hiiiiiighway': 'highway',
    'hotsssssssssssss': 'hots',
    'veeeery': 'very',
    'ginnno': 'gino',
    'timessss': 'times',
    'shhhhhhh': 'shhhh',
    'wellllll': 'well',
    'alllllllllmooooooooost': 'almost',
    'ittttttttt': 'it',
    'tooooo': 'to',
    'iiii': 'i',
    'waaaaaaaaaaaaah': 'wah',
    'thooooose': 'those',
    'blessss': 'bless',
    'eeggsss': 'eggs',
    'huhhhh': 'huh',
    'oooh': 'ooh',
    'hisss': 'hiss',
    'waaaaay': 'way',
    'riiiiight': 'right',
    'shitttt': 'shit',

    # Synonyms and correcting spelling errors:
    # '7grams': 'seven grams',
    '20s': 'twenties',
    '60s': 'sixties',
    # '3pt': '3 point',
    # '2weeks': 'two weeks',
    '2nd': 'second',
    '4eva': 'forever',
    '30s': 'thirties',
    '4th': 'fourth',
    'zer0': 'zero',
    # 'oooooooohhhhnnnoooooo': 'oh no',
    # '12hours': 'NUM hours',
    # 'yesyeesyesyesnooooo': 'yes yes yes yes no',
    'booooooyz': 'boys',
    '1st': 'first',
}
global model_dictionary
model_dictionary = {}
global word_Encoding
word_Encoding = 1

print(f"""That’s the most badass joke I’ve heard => {"That’s the most badass joke I’ve heard".replace("’", '')}""")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
bert_embed_len = 768
num_words_to_filter = 8
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

df = df[df['sample'].apply(lambda x: len(x.split(' '))) < num_words_to_filter]
df_val = df_val[df_val['sample'].apply(lambda x: len(x.split(' '))) < num_words_to_filter]
df_test = df_test[df_test['sample'].apply(lambda x: len(x.split(' '))) < num_words_to_filter]

df = df[df['sample'].apply(lambda x: len(x.split(' '))) > 0]
df_val = df_val[df_val['sample'].apply(lambda x: len(x.split(' '))) > 0]
df_test = df_test[df_test['sample'].apply(lambda x: len(x.split(' '))) > 0]

df = df.reset_index().drop(columns=['index'])
df_val = df_val.reset_index().drop(columns=['index'])
df_test = df_test.reset_index().drop(columns=['index'])


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

print(f'The full dataset contains {len(df) + len(df_val) + len(df_test)} samples.')

print(f"\n\n--------------Cleaning And Coding of Samples--------------\n\n")


def cleaning_sample(sample: str):
    global model_dictionary
    global conversion_dictionary
    global word_Encoding
    model_dictionary_size = 8383

    # Replace newlines with spaces
    sample = sample.replace('\'', '')
    sample = sample.replace("^", '')

    sample = sample.replace('\n', ' ')
    sample = sample.lower()
    sample = sample.replace('[name]', 'NAME')
    sample = sample.replace('[religion]', 'RELIGION')
    for w in sample.split(' '):
        if 'u/' in w:
            sample = sample.replace(w, 'USERNAME')
        if w.isdigit():
            sample = sample.replace(w, 'NUM')

    sample = ''.join([char if char.isalpha() or char.isdigit() or char == ' ' else ' ' for char in sample])
    sample = ''.join([char if char.isalpha() else char for char in sample if char.isalpha() or char.isdigit() or char == ' '])

    tensor_sample = None
    if len(word_tokenize(sample)) < 9:
        tensor_sample = []
        tensor_list = []
        for w in word_tokenize(sample):
            vector = [0.0] * model_dictionary_size
            if w in conversion_dictionary.keys():
                w = conversion_dictionary[w]
            if w not in model_dictionary.keys():
                model_dictionary[w] = word_Encoding
                word_Encoding += 1
            # tensor_sample += [[model_dictionary[w]]]
    #     print(len(tensor_sample))
    # return tensor_sample
                vector[model_dictionary[w] - 1] = 1.0
            tensor_list.append(vector)
        vector = [0.0] * model_dictionary_size
        ped = 8 - len(tensor_list)
        for i in range(ped):
            tensor_list.append(vector)
        tensor_sample = torch.tensor(tensor_list)
    return tensor_sample

    # Using list comprehension to filter out non-letter, non-number, non-space characters,
    # and convert letters to lowercase


df['sample'] = df['sample'].apply(cleaning_sample)
df_val['sample'] = df_val['sample'].apply(cleaning_sample)
df_test['sample'] = df_test['sample'].apply(cleaning_sample)

# df = df[df['sample'].apply(lambda x: len(x.split(' '))) < num_words_to_filter]
df = df[df['sample'].apply(lambda x: x is not None)]
df_val = df_val[df_val['sample'].apply(lambda x: x is not None)]
df_test = df_test[df_test['sample'].apply(lambda x: x is not None)]

print(f"\n\n--------------Train Info--------------\n\n{df.info()}")
print(f"\n\n--------------Train Description--------------\n\n{df.describe()}")
print(f"\n\n--------------Dev Info--------------\n\n{df_val.info()}")
print(f"\n\n--------------Dev Description--------------\n\n{df_val.describe()}")
print(f"\n\n--------------Test Info--------------\n\n{df_test.info()}")
print(f"\n\n--------------Test Description--------------\n\n{df_test.describe()}")


print(f"\n\n--------------Dataset Figures--------------\n\n")
print(f"model_dictionary size is {len(model_dictionary)}")
# print(model_dictionary)

# ===========================================================================================================
"""
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

compute_embed = True # Make sure you have enough swap-memory\virtual-memory


def tokenize_sentence(sentence):
    return tokenizer.tokenize(sentence)


def get_word_embeddings(tokens):
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0].numpy()


if compute_embed:
    print('--------------Encoding--------------')

    # Clean data
    df = df[df['sample'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    df_val = df_val[df_val['sample'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    df_test = df_test[df_test['sample'].apply(lambda x: isinstance(x, str) and x.strip() != "")]

    # TODO: run on the full dataset- remove the [:5/15] filters
    train_sentences = df['sample'].tolist()#[:15]
    val_sentences = df_val['sample'].tolist()#[:5]
    test_sentences = df_test['sample'].tolist()#[:5]

    train_tokens = [tokenize_sentence(sentence) for sentence in train_sentences]
    train_tokens = [tokens for tokens in train_tokens if tokens]

    val_tokens = [tokenize_sentence(sentence) for sentence in val_sentences]
    val_tokens = [tokens for tokens in val_tokens if tokens]

    test_tokens = [tokenize_sentence(sentence) for sentence in test_sentences]
    test_tokens = [tokens for tokens in test_tokens if tokens]


    train_embeddings = [get_word_embeddings(tokens) for tokens in train_tokens]
    val_embeddings = [get_word_embeddings(tokens) for tokens in val_tokens]
    test_embeddings = [get_word_embeddings(tokens) for tokens in test_tokens]

    max_num_tokens = max(
        max(len(item) for item in train_embeddings),
        max(len(item) for item in val_embeddings),
        max(len(item) for item in test_embeddings)
    )

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
    train_one_hot_labels = mlb.fit_transform(df['label'].tolist())#[:15])
    val_one_hot_labels = mlb.fit_transform(df_val['label'].tolist())#[:5])
    test_one_hot_labels = mlb.fit_transform(df_test['label'].tolist())#[:5])

    train_data_padded = (train_embeddings_padded, train_one_hot_labels)
    val_data_padded = (val_embeddings_padded, val_one_hot_labels)
    test_data_padded = (test_embeddings_padded, test_one_hot_labels)

    print(f'train_data_padded: {train_data_padded}')

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
    with open(f'{dataset_folder_path}/Transformer/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(f'{dataset_folder_path}/Transformer/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
print('--------------DONE--------------')
"""
