import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import json
from tqdm import tqdm
import sys
import os

# Adding the parent directory to the system path so that the task1 module can be imported
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from task1.task1 import WordPieceTokenizer

'''
References:
1. https://towardsdatascience.com/a-word2vec-implementation-using-numpy-and-python-d256cf0e5f28
2. https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html
3. https://jaketae.github.io/study/word2vec/
4. https://medium.com/@vishwasbhanawat/the-architecture-of-word2vec-78659ceb6638
5. https://www.geeksforgeeks.org/negaitve-sampling-using-word2vec/
6. https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py
'''

class Word2VecDataset(Dataset):
    '''
    Custom Dataset class for Word2Vec model.
    '''
    def __init__(self, corpus, tokenizer, context_window = 2):
        '''
        Initializes the dataset with the given corpus and tokenizer.
        
        Parameters:
        corpus (list of str): The corpus of text data.
        tokenizer (Tokenizer): The tokenizer to use for tokenizing the text.
        context_window (int): The size of the context window.
        '''
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.data = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.preprocess(corpus)

    def preprocess(self, corpus):
        '''
        Preprocesses the corpus to create context and target pairs
        according to the index of the token from the tokenized corpus.
        The tokenizer is the WordPieceTokenizer from task1.
        
        Parameters:
        corpus (list of str): The corpus of text data.
        '''
        tokenized_corpus = []
        # Tokenizing the corpus using WordPieceTokenizer
        for text in corpus:
            tokens = self.tokenizer.tokenize(text)
            tokenized_corpus.extend(tokens)
        # Creating the vocabulary and mapping of words to indices and indices to words
        for i in range(len(tokenized_corpus)):
            if tokenized_corpus[i] not in self.word_to_idx:
                self.word_to_idx[tokenized_corpus[i]] = len(self.word_to_idx)
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        # Creating the context-target pairs using CBOW architecture
        for i in range(self.context_window, len(tokenized_corpus) - self.context_window):
            context = (tokenized_corpus[i-self.context_window:i] + 
                      tokenized_corpus[i+1:i+self.context_window+1])
            target = tokenized_corpus[i]
            context_idx = [self.word_to_idx[word] for word in context]
            target_idx = self.word_to_idx[target]
            self.data.append((context_idx, target_idx))

    def __len__(self):
        '''
        Returns the number of context-target pairs in the dataset.
        
        Returns:
        int: The number of context-target pairs.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns the context and target word indices for a given index.
        
        Parameters:
        idx (int): The index of the context-target pair.
        
        Returns:
        tuple: A tuple containing the context word indices and the target word index.
        '''
        context_words, target_word = self.data[idx]
        context_idx = [word for word in context_words]  
        target_idx = target_word
        return (torch.tensor(context_idx, dtype=torch.long),  
                torch.tensor(target_idx, dtype=torch.long))
    
class Word2VecModel(nn.Module):
    '''
    Custom Word2Vec model class.
    '''
    def __init__(self, vocab_size, embedding_dim):
        '''
        Initializes the Word2Vec model with the given vocabulary size and embedding dimension.
        
        Parameters:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding vectors.
        '''
        super(Word2VecModel, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, context, target, negative_samples):
        '''
        Forward pass for the Word2Vec model.
        
        Parameters:
        context (torch.Tensor): The context word indices.
        target (torch.Tensor): The target word indices.
        negative_samples (torch.Tensor): The negative sample word indices.
        
        Returns:
        torch.Tensor: The loss value for the batch.
        '''
        target_embedding = self.embeddings(target)
        context_embedding = self.context_embeddings(context)
        context_embedding = context_embedding.mean(dim=1)
        negative_embeddings = self.context_embeddings(negative_samples)
        positive_score = self.log_sigmoid(torch.sum(context_embedding * target_embedding, dim=1))
        negative_score = self.log_sigmoid(-torch.bmm(negative_embeddings, context_embedding.unsqueeze(2)).squeeze(2)).sum(1)
        loss = - (positive_score + negative_score).mean()
        return loss
    
def get_negative_samples(target, num_negative_samples, vocab_size):
    '''
    Generate a list of negative samples for a given target.

    Parameters:
    target (int): The target integer that should not be included in the negative samples.
    num_negative_samples (int): The number of negative samples to generate.
    vocab_size (int): The size of the vocabulary from which to sample.

    Returns:
    list: A list of negative samples (integers) that are different from the target.
    '''
    neg_samples = []
    while len(neg_samples) < num_negative_samples:
        neg_sample = np.random.randint(0, vocab_size)
        if neg_sample != target:
            neg_samples.append(neg_sample)
    return neg_samples    
    
def train(dataset, embedding_dim=100, epochs=20, batch_size=32, lr=0.001, val_split=0.2, num_negative_samples = 5):
    '''
    Trains the Word2Vec model on the given dataset.
    
    Parameters:
    dataset (Word2VecDataset): The dataset to train on.
    embedding_dim (int): The dimension of the embedding vectors.
    epochs (int): The number of training epochs.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for the optimizer.
    val_split (float): The proportion of the dataset to use for validation.
    num_negative_samples (int): The number of negative samples to use.
    
    Returns:
    Word2VecModel: The trained Word2Vec model and the losses.
    '''
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = Word2VecModel(dataset.vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        model.train()
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for context, target in train_progress:
            negative_samples = torch.LongTensor([get_negative_samples(t.item(), num_negative_samples, dataset.vocab_size) for t in target])
            optimizer.zero_grad()
            loss = model(context, target, negative_samples)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for context, target in val_progress:
                negative_samples = torch.LongTensor([get_negative_samples(t.item(), num_negative_samples, dataset.vocab_size) for t in target])
                loss = model(context, target, negative_samples)
                total_val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), "word2vec_cbow.pth")
    return model, train_losses, val_losses


def plot_loss(train_losses, val_losses):
    '''
    Plots the training and validation loss over epochs.
    
    Parameters:
    train_losses (list of float): List of training loss values for each epoch.
    val_losses (list of float): List of validation loss values for each epoch.
        
    Returns:
    None
    '''
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

def cosine_similarity(vec1, vec2):
    '''
    Computes the cosine similarity between two vectors.
    
    Parameters:
    vec1 (np.array): The first vector.
    vec2 (np.array): The second vector.
    
    Returns:
    float: The cosine similarity between the two vectors.
    '''
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def get_triplet_combinations(model, dataset, words):
    '''
    Makes all combinaitons of the given words and finds the most similar and least similar pairs.
    
    Parameters:
    model (Word2VecModel): The trained Word2Vec model.
    dataset (Word2VecDataset): The dataset used for training the model.
    words (list of str): The list of words to make combinations from.
    
    Returns:
    list: A list containing the most similar and least similar pairs of words.
    '''
    embeddings = model.embeddings.weight.detach().numpy()
    combinations = []
    for i in range(len(words)):
        for j in range(len(words)):
            for k in range(len(words)):
                if i != j and i != k and j != k:
                    combinations.append((words[i], words[j], words[k]))
    similarities = []
    # Getting cosine similarities for all the possible combinations
    for w1, w2, w3 in combinations:
        vec1 = embeddings[dataset.word_to_idx[w1]]
        vec2 = embeddings[dataset.word_to_idx[w2]]
        vec3 = embeddings[dataset.word_to_idx[w3]]
        sim1 = cosine_similarity(vec1, vec2)
        sim2 = cosine_similarity(vec1, vec3)
        sim3 = cosine_similarity(vec2, vec3)
        similarities.extend([[sim1, w1, w2], [sim2, w1, w3], [sim3, w2, w3]])
    similarities.sort(reverse=True)
    return similarities[0] + similarities[-1]

def find_similar_words(model, dataset):
    '''
    Finds the most similar and least similar words for a given set of word triplets.
    
    Parameters:
    model (Word2VecModel): The trained Word2Vec model.
    dataset (Word2VecDataset): The dataset used for training the model.
    
    Returns:
    None
    '''
    word_pairs = [
        ("kingdom", "folks", "attacking"),
        ("subject", "study", "preschool")
    ]
    for word_pair in word_pairs:
        top_similarities = get_triplet_combinations(model, dataset, word_pair)
        print("Word triplets: ", word_pair)
        print("Top similarity: {:.4f} between {} and {}".format(top_similarities[0], top_similarities[1], top_similarities[2]))
        print("Lowest similarity: {:.4f} between {} and {}".format(top_similarities[3], top_similarities[4], top_similarities[5]))
        print()
        
# Load the trained model
# def load_model(model, filepath):
#     model.load_state_dict(torch.load(filepath))
#     model.eval()
#     return model

# Main function
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 100
    EPOCHS = 25
    BATCH_SIZE = 32
    LR = 0.001
    VAL_SPLIT = 0.2
    NUM_NEGATIVE_SAMPLES = 5
    # Reading the corpus
    with open("../corpus.txt", "r") as file:
        corpus = file.readlines()
    # Initializing the WordPieceTokenizer
    tokenizer = WordPieceTokenizer(corpus, vocab_size=VOCAB_SIZE)
    # Constructing vocabulary
    '''
    If the vocabulary file exists (for vocab size 10k suppose), load the vocabulary from the file.
    Otherwise, construct the vocabulary and save it to a file.
    '''
    if os.path.exists("vocabulary_50.txt"):
        with open("vocabulary_50.txt", "r") as file:
            tokenizer.vocab = file.readlines()
        tokenizer.vocab = [word.strip('\n') for word in tokenizer.vocab]
    else:
        tokenizer.construct_vocabulary()
    # Instantiating the Word2VecDataset
    dataset = Word2VecDataset(corpus, tokenizer=tokenizer, context_window=2)
    # Saving the word_to_idx and idx_to_word mappings
    with open("word_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(dataset.word_to_idx, f, ensure_ascii=False, indent=2)
    idx_to_word_str = {str(k): v for k, v in dataset.idx_to_word.items()}
    with open("idx_to_word.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_word_str, f, ensure_ascii=False, indent=2)
    # Training the Word2Vec model
    model, train_loss, val_loss = train(dataset, embedding_dim=EMBEDDING_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, val_split=VAL_SPLIT, num_negative_samples=NUM_NEGATIVE_SAMPLES)
    plot_loss(train_loss, val_loss)
    # model = Word2VecModel(dataset.vocab_size, 100)
    # model = load_model(model, "word2vec_cbow.pth")
    find_similar_words(model, dataset)