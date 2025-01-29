from collections import Counter
import torch
from torch.utils.data import Dataset
from task1 import WordPieceTokenizer

# Reference: https://towardsdatascience.com/a-word2vec-implementation-using-numpy-and-python-d256cf0e5f28
# https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html

class Word2VecDataset(Dataset):
    def __init__(self, corpus, tokenizer, context_window = 2):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.data = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.preprocess(corpus)

    def preprocess(self, corpus):
        tokenized_corpus = []
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            tokenized_corpus.extend(tokens)

        # word_counts = Counter(tokenized_corpus)
        # print(word_counts)
        # vocab = list(set(tokenized_corpus))
        # print(vocab)
        # self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        for i in range(len(tokenized_corpus)):
            if tokenized_corpus[i] not in self.word_to_idx:
                self.word_to_idx[tokenized_corpus[i]] = len(self.word_to_idx)
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        print(self.word_to_idx)
        print(self.idx_to_word)
        print(f"Vocabulary size: {self.vocab_size}")
        # Create CBOW training pairs
        for i in range(self.context_window, len(tokenized_corpus) - self.context_window):
            context = (tokenized_corpus[i-self.context_window:i] + 
                      tokenized_corpus[i+1:i+self.context_window+1])
            target = tokenized_corpus[i]
            # print(context, target)
            # Convert words to indices
            context_idx = [self.word_to_idx[w] for w in context]
            target_idx = self.word_to_idx[target]
            # print(context_idx, target_idx)
            self.data.append((context_idx, target_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_idx, target_idx = self.data[idx]
        return (torch.tensor(context_idx, dtype=torch.long),
                torch.tensor(target_idx, dtype=torch.long))
    
# with open("corpus.txt", "r") as file:
#     corpus = file.readlines()

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization."
    ]

tokenizer = WordPieceTokenizer(corpus, vocab_size=1000)
tokenizer.construct_vocabulary()
# for text in corpus:
#     print(tokenizer.tokenize(text))
dataset = Word2VecDataset(corpus, tokenizer=tokenizer, context_window=2)
print(dataset.data)