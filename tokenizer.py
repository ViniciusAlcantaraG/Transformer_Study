from transformers import AutoTokenizer
from collections import defaultdict, Counter


corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
tokenizer = AutoTokenizer.from_pretrained('gpt2')

word_freqs = defaultdict(int)

# Computing frequencies
for text in corpus:
    words_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offset]
    for word in new_words:
        word_freqs[word]+=1

# Compute base vocabulary
alphabet = []
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

vocab = ['<|endoftext|>'] + alphabet.copy()

# Split words into individual characters
splits = {word: [c for c in word] for word in word_freqs.keys()}

def compute_pair_freq(splits):
    pair_freq = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split)==1:
            continue
        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair_freq[pair]+=freq
    return pair_freq

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split)==1:
            continue

        i = 0
        while i < len(split)-1:
            if (split[i] == a and split[i+1] == b):
                split = split[:i] + [a+b] + split[i+2:]
            else:
                i+=1
        splits[word] = split
    return splits


            



