"""
Prepare the dataset for phoneme-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map phonemes to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
from tqdm import tqdm
import requests
import numpy as np
import pronouncing
import random

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input_cleaned.txt')

with open(input_file_path, 'r') as f:
    text = f.read()
print(f"length of dataset in characters: {len(text):,}")


# tokenize the dataset
data = []
phones_cache = {}
for verse in tqdm(text.split("\n\n")):
    data.append("\n\n")
    for line in verse.split("\n"):
        data.append("\n")
        for word in line.split():
            word = word.lower().strip('.,!?-)(]["}{')
            if word in phones_cache:
                phones = phones_cache[word]
            else:
                phones = pronouncing.phones_for_word(word)
                phones_cache[word] = phones
            if phones:
                tokens = random.choice(phones).split()
                data += tokens
            else:
                data.append("?")

print(f"length of dataset in phonemes: {len(data):,}")

# get the unique phonemes
phonemes = set(data)
vocab_size = len(phonemes)
print("all the unique phonemes:", ' '.join(phonemes))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ph:i for i,ph in enumerate(phonemes) }
itos = { i:ph for i,ph in enumerate(phonemes) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
