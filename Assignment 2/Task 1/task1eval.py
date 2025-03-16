import json
import re
import string
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from conlleval import evaluate

import json
import string

def tokenize_with_spans(sentence):
    """
    Tokenizes the sentence using Python's split() after removing punctuation.
    
    First, punctuation is removed from the sentence using str.translate().
    Then, the cleaned sentence is split using split() to generate tokens.
    To compute spans, we search for each token in the original sentence,
    starting from a moving index so that repeated tokens are located in order.
    
    Returns:
        tokens: a list of tokens (words)
        spans: a list of (start, end) tuples for each token in the original sentence.
    """
    translator = str.maketrans('', '', string.punctuation)
    cleaned_sentence = sentence.translate(translator)
    tokens = cleaned_sentence.split()
    
    # Compute spans in the original sentence
    spans = []
    current_index = 0
    for token in tokens:
        # Find the token in the original sentence, starting from current_index
        start = sentence.find(token, current_index)
        if start == -1:
            # Fallback: if not found, use current_index
            start = current_index
        end = start + len(token)
        spans.append((start, end))
        current_index = end  # Move pointer forward to avoid matching the same occurrence again
    return tokens, spans

def assign_bio_labels(tokens, spans, aspect_terms):
    """
    Assigns BIO labels to tokens based on the aspect term offsets.
    
    The aspect term offsets ('from' and 'to') refer to the original sentence.
    For each aspect term, tokens whose original character spans fall completely 
    within [from, to] are labeled. The first token gets 'B', and subsequent tokens get 'I'.
    """
    labels = ['O'] * len(tokens)
    for aspect in aspect_terms:
        # Convert the provided offsets from strings to integers
        start_offset = int(aspect['from'])
        end_offset = int(aspect['to'])
        is_first = True
        for i, (token_start, token_end) in enumerate(spans):
            # Check if the token's span is completely within the aspect term's offsets.
            if token_start >= start_offset and token_end <= end_offset:
                labels[i] = 'B' if is_first else 'I'
                is_first = False
    return labels

def preprocess_data(input_file, output_file):
    """
    Loads the JSON data, tokenizes each sentence while tracking character spans,
    assigns BIO labels using the provided 'from' and 'to' offsets, and writes the output.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    preprocessed_examples = []
    for example in data:
        sentence = example['sentence']
        tokens, spans = tokenize_with_spans(sentence)
        labels = assign_bio_labels(tokens, spans, example.get('aspect_terms', []))
        extracted_terms = [aspect['term'] for aspect in example.get('aspect_terms', [])]
        
        preprocessed_examples.append({
            "sentence": sentence,
            "tokens": tokens,
            "labels": labels,
            "aspectterms": extracted_terms
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_examples, f, indent=4)

# 1. Build Vocabulary (from training preprocessed file)

def build_vocab(preprocessed_file, min_freq=1):
    """
    Build a vocabulary from the preprocessed JSON file.
    """
    with open(preprocessed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    from collections import Counter
    counter = Counter()
    for ex in data:
        counter.update([token.lower() for token in ex['tokens']])
    
    # Reserve indices for special tokens.
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)
    
    return word2idx

# Assume train_task_1.json exists from your preprocessing
word2idx = build_vocab('train_task_1.json')
print("Vocabulary size:", len(word2idx))


# 2. Define Tag Mapping and Other Hyperparameters

tag2idx = {'O': 0, 'B': 1, 'I': 2}
vocab_size = len(word2idx)
tagset_size = len(tag2idx)
hidden_dim = 128

# FastText parameters
embedding_dim_fasttext = 300

# GloVe parameters
embedding_dim_glove = 300

# Load fastText embeddings
def load_fasttext_embeddings(fasttext_file_path, embedding_dim):
    embeddings = {}
    with open(fasttext_file_path, 'r', encoding='utf8') as f:
        header = f.readline()  # skip header
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings

fasttext_file = "wiki-news-300d-1M.vec"
fasttext_embeddings = load_fasttext_embeddings(fasttext_file, embedding_dim_fasttext)

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path, embedding_dim):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings

glove_file = "glove.6B.100d.txt"
glove_embeddings = load_glove_embeddings(glove_file, embedding_dim_glove)

def create_embedding_matrix(word2idx, embeddings, embedding_dim):
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
    return embedding_matrix

embedding_matrix_fasttext = create_embedding_matrix(word2idx, fasttext_embeddings, embedding_dim_fasttext)
embedding_matrix_glove = create_embedding_matrix(word2idx, glove_embeddings, embedding_dim_glove)


# 3. Define the Dataset for Test Data

class AspectDataset(Dataset):
    def __init__(self, preprocessed_file, word2idx, tag2idx, max_len=100):
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        tokens = ex['tokens']
        labels = ex['labels']
        token_ids = [self.word2idx.get(token.lower(), self.word2idx["<UNK>"]) for token in tokens]
        label_ids = [self.tag2idx[label] for label in labels]
        pad_length = self.max_len - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [self.word2idx["<PAD>"]] * pad_length
            label_ids = label_ids + [-100] * pad_length
        else:
            token_ids = token_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]
        return torch.tensor(token_ids), torch.tensor(label_ids), tokens

def custom_collate(batch):
    token_ids = torch.stack([item[0] for item in batch])
    label_ids = torch.stack([item[1] for item in batch])
    tokens = [item[2] for item in batch]
    return token_ids, label_ids, tokens

preprocess_data('val.json', 'val_task_1.json')

test_dataset = AspectDataset('val_task_1.json', word2idx, tag2idx, max_len=100)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=custom_collate)


# 4. Define the Model (RNNTagger)

class RNNTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, embeddings=None, rnn_type='GRU'):
        super(RNNTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
            self.embedding.weight.requires_grad = False
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")
        self.fc = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, _ = self.rnn(embeds)
        logits = self.fc(rnn_out)
        return logits


# 5. Define Evaluation Function

def evaluate_model_on_dataset(model, dataset, device):
    """
    Runs the model on the dataset and generates evaluation lines in the format:
      token true_tag predicted_tag
    Each sentence is separated by an empty line.
    The numeric labels are converted as follows:
      0 -> "O"
      1 -> "B-ASPECT"
      2 -> "I-ASPECT"
    Padding tokens (true label == -100) are ignored.
    """
    model.eval()
    idx2tag = {0: "O", 1: "B-ASPECT", 2: "I-ASPECT"}
    eval_lines = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            token_ids, label_ids, tokens = dataset[i]
            token_ids = token_ids.unsqueeze(0).to(device)
            label_ids = label_ids.unsqueeze(0).to(device)
            logits = model(token_ids)
            predictions = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            true_labels = label_ids.squeeze(0).cpu().numpy()
            
            for j in range(len(tokens)):
                if true_labels[j] == -100:
                    continue
                true_tag = idx2tag[true_labels[j]]
                pred_tag = idx2tag[predictions[j]]
                eval_lines.append(f"{tokens[j]} {true_tag} {pred_tag}")
            eval_lines.append("")
    return eval_lines


# 6. Define Function to Load Saved Model and Evaluate Test File

def evaluate_test_file(test_json_file, model_path, device, embedding_dim, rnn_type='GRU', embeddings=None):
    """
    Loads the test.json file, creates a test dataset, loads the trained model from model_path,
    runs the model on the test dataset, and returns (precision, recall, F1) scores using conlleval's evaluate.
    
    Parameters:
      test_json_file: Path to test.json.
      model_path: Path to the saved model state dictionary.
      device: 'cpu' or 'cuda'
      embedding_dim: Dimension of embeddings used by the model.
      rnn_type: 'RNN' or 'GRU'
      embeddings: Pre-trained embedding matrix to initialize the model.
      
    Returns:
      A tuple (precision, recall, F1).
    """
    test_dataset = AspectDataset(test_json_file, word2idx, tag2idx, max_len=100)
    model = RNNTagger(vocab_size, embedding_dim, hidden_dim, tagset_size, embeddings=embeddings, rnn_type=rnn_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    eval_lines = evaluate_model_on_dataset(model, test_dataset, device)
    scores = evaluate(eval_lines)
    return scores


# 7. Evaluate the Saved Models on the Test File

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate GRU with fastText embeddings
print("Evaluating GRU with FastText embeddings on test set...")
test_scores = evaluate_test_file("val_task_1.json", "best_model_gru_fasttext.pt", device, embedding_dim_fasttext, rnn_type='GRU', embeddings=embedding_matrix_fasttext)
print("GRU with FastText - Evaluation Result (Precision, Recall, F1):", test_scores)

# Evaluate GRU with GloVe embeddings
print("\nEvaluating GRU with GloVe embeddings on test set...")
test_scores = evaluate_test_file("val_task_1.json", "best_model_gru_glove.pt", device, embedding_dim_glove, rnn_type='GRU', embeddings=embedding_matrix_glove)
print("GRU with GloVe - Evaluation Result (Precision, Recall, F1):", test_scores)

# Evaluate RNN with fastText embeddings
print("\nEvaluating RNN with FastText embeddings on test set...")
test_scores = evaluate_test_file("val_task_1.json", "best_model_rnn_fasttext.pt", device, embedding_dim_fasttext, rnn_type='RNN', embeddings=embedding_matrix_fasttext)
print("RNN with FastText - Evaluation Result (Precision, Recall, F1):", test_scores)

# Evaluate RNN with GloVe embeddings
print("\nEvaluating RNN with GloVe embeddings on test set...")
test_scores = evaluate_test_file("val_task_1.json", "best_model_rnn_glove.pt", device, embedding_dim_glove, rnn_type='RNN', embeddings=embedding_matrix_glove)
print("RNN with GloVe - Evaluation Result (Precision, Recall, F1):", test_scores)
