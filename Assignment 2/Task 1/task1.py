import json
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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

preprocess_data('train.json', 'train_task_1.json')
preprocess_data('val.json', 'val_task_1.json')


# 1. Build Vocabulary

def build_vocab(preprocessed_file, min_freq=1):
    """
    Build a vocabulary from the preprocessed JSON file.
    """
    with open(preprocessed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    counter = Counter()
    for ex in data:
        counter.update([token.lower() for token in ex['tokens']])
    
    # Reserve indices for special tokens
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)
    
    return word2idx

word2idx = build_vocab('train_task_1.json')
print("Vocabulary size:", len(word2idx))


# 2. Load Pre-trained Embeddings

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

embedding_dim_glove = 300
glove_file = "glove.6B.300d.txt"
glove_embeddings = load_glove_embeddings(glove_file, embedding_dim_glove)
print("Loaded", len(glove_embeddings), "GloVe embeddings.")

def load_fasttext_embeddings(fasttext_file_path, embedding_dim):
    embeddings = {}
    with open(fasttext_file_path, 'r', encoding='utf8') as f:
        header = f.readline()
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings

embedding_dim_fasttext = 300
fasttext_file = "wiki-news-300d-1M.vec"
fasttext_embeddings = load_fasttext_embeddings(fasttext_file, embedding_dim_fasttext)
print("Loaded", len(fasttext_embeddings), "fastText embeddings.")

def create_embedding_matrix(word2idx, embeddings, embedding_dim):
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
    return embedding_matrix

embedding_matrix_glove = create_embedding_matrix(word2idx, glove_embeddings, embedding_dim_glove)
embedding_matrix_fasttext = create_embedding_matrix(word2idx, fasttext_embeddings, embedding_dim_fasttext)


# 3. Define the Dataset

# Fixed mapping for BIO tags.
tag2idx = {'O': 0, 'B': 1, 'I': 2}

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
        # Convert tokens to indices (using lower-case and <UNK> for missing words)
        token_ids = [self.word2idx.get(token.lower(), self.word2idx["<UNK>"]) for token in tokens]
        label_ids = [self.tag2idx[label] for label in labels]
        # Pad the sequences up to max_len
        pad_length = self.max_len - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [self.word2idx["<PAD>"]] * pad_length
            label_ids = label_ids + [-100] * pad_length  # -100 to ignore in loss computation
        else:
            token_ids = token_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]
        # Return token_ids, label_ids, and the original tokens (for evaluation)
        return torch.tensor(token_ids), torch.tensor(label_ids), tokens


# 4. Custom Collate Function

def custom_collate(batch):
    """
    Custom collate function to handle variable-length token lists.
    It stacks token_ids and label_ids, but leaves tokens as a list.
    """
    token_ids = torch.stack([item[0] for item in batch])
    label_ids = torch.stack([item[1] for item in batch])
    tokens = [item[2] for item in batch]
    return token_ids, label_ids, tokens


train_dataset = AspectDataset('train_task_1.json', word2idx, tag2idx, max_len=100)
val_dataset = AspectDataset('val_task_1.json', word2idx, tag2idx, max_len=100)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=custom_collate)


# 5. Define the Model

class RNNTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, embeddings=None, rnn_type='RNN'):
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


# 6. Training Code

def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.0001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # to ignore padded tokens
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for token_ids, label_ids, _ in train_loader:
            token_ids, label_ids = token_ids.to(device), label_ids.to(device)
            optimizer.zero_grad()
            logits = model(token_ids)  # shape: (batch, seq_len, tagset_size)
            loss = criterion(logits.view(-1, logits.shape[-1]), label_ids.view(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for token_ids, label_ids, _ in val_loader:
                token_ids, label_ids = token_ids.to(device), label_ids.to(device)
                logits = model(token_ids)
                loss = criterion(logits.view(-1, logits.shape[-1]), label_ids.view(-1))
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model, train_losses, val_losses

def train_and_save(model, model_name, train_loader, val_loader, device, num_epochs=30, lr=0.0001):
    """
    Trains the given model, saves the best model weights, and returns the loss curves.
    Also saves one loss curve image (training and validation) per model.
    """
    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device)
    torch.save(trained_model.state_dict(), f"best_model_{model_name}.pt")
    print(f"Saved best model for {model_name} as best_model_{model_name}.pt")
    

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.legend()
    plt.savefig(f"{model_name}_loss_curve.png")
    plt.close()
    
    return train_losses, val_losses


# 7. Train Four Model Configurations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 128
vocab_size = len(word2idx)
tagset_size = len(tag2idx)

# 1. RNN with GloVe Embeddings
model_rnn_glove = RNNTagger(vocab_size, embedding_dim_glove, hidden_dim, tagset_size,
                              embeddings=embedding_matrix_glove, rnn_type='RNN').to(device)
print("Training RNN with GloVe embeddings...")
train_losses_rnn_glove, val_losses_rnn_glove = train_and_save(model_rnn_glove, "rnn_glove", train_loader, val_loader, device)

# 2. RNN with fastText Embeddings
model_rnn_fasttext = RNNTagger(vocab_size, embedding_dim_fasttext, hidden_dim, tagset_size,
                               embeddings=embedding_matrix_fasttext, rnn_type='RNN').to(device)
print("Training RNN with fastText embeddings...")
train_losses_rnn_fasttext, val_losses_rnn_fasttext = train_and_save(model_rnn_fasttext, "rnn_fasttext", train_loader, val_loader, device)

# 3. GRU with GloVe Embeddings
model_gru_glove = RNNTagger(vocab_size, embedding_dim_glove, hidden_dim, tagset_size,
                            embeddings=embedding_matrix_glove, rnn_type='GRU').to(device)
print("Training GRU with GloVe embeddings...")
train_losses_gru_glove, val_losses_gru_glove = train_and_save(model_gru_glove, "gru_glove", train_loader, val_loader, device)

# 4. GRU with fastText Embeddings
model_gru_fasttext = RNNTagger(vocab_size, embedding_dim_fasttext, hidden_dim, tagset_size,
                               embeddings=embedding_matrix_fasttext, rnn_type='GRU').to(device)
print("Training GRU with fastText embeddings...")
train_losses_gru_fasttext, val_losses_gru_fasttext = train_and_save(model_gru_fasttext, "gru_fasttext", train_loader, val_loader, device)


# 8. Evaluation Code Using conlleval


from conlleval import evaluate  # conlleval's evaluate function

def evaluate_model_on_dataset(model, dataset, device):
    """
    Runs the model on the dataset and generates evaluation lines in the format:
      token true_tag predicted_tag
    Each sentence is separated by an empty line.
    The numeric labels are converted by mapping:
      0 -> "O"
      1 -> "B-ASPECT"
      2 -> "I-ASPECT"
    Padding tokens are ignored (where true label == -100).
    """
    model.eval()
    idx2tag = {0: "O", 1: "B-ASPECT", 2: "I-ASPECT"}
    eval_lines = []
    
    with torch.no_grad():
        # one example at a time to preserve sentence boundaries
        for i in range(len(dataset)):
            token_ids, label_ids, tokens = dataset[i]  # tokens is the original token list
            token_ids = token_ids.unsqueeze(0).to(device)  # add batch dim
            label_ids = label_ids.unsqueeze(0).to(device)
            
            logits = model(token_ids)  # shape: (1, seq_len, tagset_size)
            predictions = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # shape: (seq_len,)
            true_labels = label_ids.squeeze(0).cpu().numpy()
            
            # For each token, ignore padded positions (where true label == -100)
            for j in range(len(tokens)):
                if true_labels[j] == -100:
                    continue
                true_tag = idx2tag[true_labels[j]]
                pred_tag = idx2tag[predictions[j]]
                eval_lines.append(f"{tokens[j]} {true_tag} {pred_tag}")
            eval_lines.append("")  # blank line to separate sentences
    
    return eval_lines

print("Evaluating RNN with GloVe embeddings on validation set...")
eval_lines_rnn_glove = evaluate_model_on_dataset(model_rnn_glove, val_dataset, device)
result_rnn_glove = evaluate(eval_lines_rnn_glove)
print("RNN with GloVe - Evaluation Result (Precision, Recall, F1):", result_rnn_glove)

print("\nEvaluating RNN with fastText embeddings on validation set...")
eval_lines_rnn_fasttext = evaluate_model_on_dataset(model_rnn_fasttext, val_dataset, device)
result_rnn_fasttext = evaluate(eval_lines_rnn_fasttext)
print("RNN with fastText - Evaluation Result (Precision, Recall, F1):", result_rnn_fasttext)

print("\nEvaluating GRU with GloVe embeddings on validation set...")
eval_lines_gru_glove = evaluate_model_on_dataset(model_gru_glove, val_dataset, device)
result_gru_glove = evaluate(eval_lines_gru_glove)
print("GRU with GloVe - Evaluation Result (Precision, Recall, F1):", result_gru_glove)

print("\nEvaluating GRU with fastText embeddings on validation set...")
eval_lines_gru_fasttext = evaluate_model_on_dataset(model_gru_fasttext, val_dataset, device)
result_gru_fasttext = evaluate(eval_lines_gru_fasttext)
print("GRU with fastText - Evaluation Result (Precision, Recall, F1):", result_gru_fasttext)

