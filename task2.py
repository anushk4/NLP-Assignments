import torch
from torch.utils.data import Dataset
from task1 import WordPieceTokenizer
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.utils.data as DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

# Reference: https://towardsdatascience.com/a-word2vec-implementation-using-numpy-and-python-d256cf0e5f28
# https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html
# https://jaketae.github.io/study/word2vec/

class Word2VecDataset(Dataset):
    def __init__(self, corpus, tokenizer, context_window = 2):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.data = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        # self.one_hot = {}
        # self.one_hot_inverse = {}
        self.vocab_size = 0
        self.preprocess(corpus)

    def preprocess(self, corpus):
        tokenized_corpus = []
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            tokenized_corpus.extend(tokens)
        for i in range(len(tokenized_corpus)):
            if tokenized_corpus[i] not in self.word_to_idx:
                self.word_to_idx[tokenized_corpus[i]] = len(self.word_to_idx)
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        # print(self.word_to_idx)
        # print(self.idx_to_word)
        # print(f"Vocabulary size: {self.vocab_size}")
        # for i in tokenized_corpus:
        #     self.one_hot[i] = np.eye(self.vocab_size)[self.word_to_idx[i]].tolist()
        # print(self.one_hot)
        # self.one_hot_inverse = {idx: word for word, idx in self.one_hot.items()}
        # Create CBOW training pairs
        for i in range(self.context_window, len(tokenized_corpus) - self.context_window):
            context = (tokenized_corpus[i-self.context_window:i] + 
                      tokenized_corpus[i+1:i+self.context_window+1])
            target = tokenized_corpus[i]
            # print(context, target)
            # context_idx = np.zeros(self.vocab_size)
            # for word in context:
            #     context_idx += np.array(self.one_hot[word])
            # context_idx = np.clip(context_idx, 0, 1)
            context_idx = [self.word_to_idx[word] for word in context]
            target_idx = self.word_to_idx[target]
            # print(context_idx, target_idx)
            self.data.append((context_idx, target_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_words, target_word = self.data[idx]

        # Convert words to integer indices
        context_idx = [word for word in context_words]  
        target_idx = target_word

        return (torch.tensor(context_idx, dtype=torch.long),  
                torch.tensor(target_idx, dtype=torch.long))


        
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embed = self.embeddings(context)  # (batch_size, context_size, embedding_dim)
        embed = embed.mean(dim=1)  # Average over context words
        output = self.linear(embed)  # Predict word
        return output

        
        
def train(dataset, embedding_dim=100, epochs=10, batch_size=32, lr=0.001):
    data_loader = DataLoader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Word2VecModel(dataset.vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for context, target in data_loader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(data_loader))
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), "word2vec_cbow.pth")
    return model, losses

# Plot Training Loss
def plot_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epochs")
    plt.show()

# Compute cosine similarity and find triplets
def find_similar_words(model, dataset):
    embeddings = model.embeddings.weight.detach().numpy()
    word_pairs = [
        ("unfit", "buff", "yoga"),  # Replace with actual words
        ("enjoying", "laugh", "scoffing")
    ]
    for w1, w2, w3 in word_pairs:
        vec1 = embeddings[dataset.word_to_idx[w1]]
        vec2 = embeddings[dataset.word_to_idx[w2]]
        vec3 = embeddings[dataset.word_to_idx[w3]]
        sim1 = cosine_similarity([vec1], [vec2])[0, 0]
        sim2 = cosine_similarity([vec1], [vec3])[0, 0]
        print(f"Similarity({w1}, {w2}): {sim1:.4f}, Similarity({w1}, {w3}): {sim2:.4f}")

with open("corpus.txt", "r") as file:
    corpus = file.readlines()

# corpus = [
#     "This is the Hugging Face Course.",
#     "This chapter is about tokenization."
#     ]

tokenizer = WordPieceTokenizer(corpus, vocab_size=1000)
tokenizer.construct_vocabulary()
dataset = Word2VecDataset(corpus, tokenizer=tokenizer, context_window=2)
# print(dataset.word_to_idx)
model, losses = train(dataset)
plot_loss(losses)
find_similar_words(model, dataset)