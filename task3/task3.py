import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from task1.task1 import WordPieceTokenizer
from task2.task2 import Word2VecModel


def load_pretrained_word2vec(checkpoint_path, vocab_size, embedding_dim):
    """Instantiate Word2VecModel and load saved weights from Task 2."""
    model = Word2VecModel(vocab_size, embedding_dim)
    model.load_state_dict(torch.load(checkpoint_path))
    return model


class NeuralLMDataset(Dataset):
    """
    Next-word prediction dataset using:
      - PreTrainedWordPieceTokenizer for subword tokenization
      - The dictionaries from Task 2 (word_to_idx.json, idx_to_word.json)
    """
    def __init__(self, corpus, tokenizer, word2idx, idx2word, context_size=2):
        """
        Args:
            corpus (list[str]): Raw text data
            tokenizer : WordPieceTokenizer from task 1
            word2idx (dict): token -> int
            idx2word (dict): int -> token
            context_size (int): How many tokens of context for prediction
        """
        self.tokenizer = tokenizer
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(self.word2idx)
        self.context_size = context_size
        
        # Build training samples
        self.samples = []
        self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        for sentence in corpus:
            tokens = self.tokenizer.tokenize(sentence)
            indices = [self.word2idx[t] for t in tokens if t in self.word2idx]
            
            # For each position i, we gather context_size tokens -> next token
            for i in range(len(indices) - self.context_size):
                context = indices[i : i + self.context_size]
                target = indices[i + self.context_size]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Fetch the (context, target) pair at the given index in the dataset."""
        context, target = self.samples[idx]
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return context_tensor, target_tensor


class NeuralLM1(nn.Module):
    
    """A simple CBOW language model with:
      - An embedding layer
      - Mean pooling over context embeddings
      - A single linear layer to output vocab scores
    """
    def __init__(self, vocab_size, embedding_dim, context_size,
                 pretrained_embeddings=None, freeze_emb=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_emb:
                self.embedding.weight.requires_grad = False
        
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.context_size = context_size

    def forward(self, x):
        # x: [batch_size, context_size]
        embedded = self.embedding(x)            # [batch_size, context_size, embedding_dim]
        cbow_vec = embedded.mean(dim=1)         # [batch_size, embedding_dim]
        out = self.fc(cbow_vec)                 # [batch_size, vocab_size]
        return out


class NeuralLM2(nn.Module):
    
    """
    A slightly deeper language model with:
      - Embedding layer
      - Mean pooling
      - Single hidden layer (tanh activation)
      - Output linear layer
    """
    def __init__(self, vocab_size, embedding_dim, context_size,
                 hidden_dim=128, pretrained_embeddings=None, freeze_emb=False):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_emb:
                self.embedding.weight.requires_grad = False
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.context_size = context_size

    def forward(self, x):
        """
        Forward pass:
          1) Embedding lookup
          2) Mean pooling for context
          3) Hidden layer with tanh
          4) Output layer
        """
        embedded = self.embedding(x)
        cbow_vec = embedded.mean(dim=1)
        h = torch.tanh(self.fc1(cbow_vec))
        out = self.fc2(h)
        return out


class NeuralLM3(nn.Module):
    
    """
    A model that concatenates context embeddings instead of averaging them.
    Then passes through a hidden layer with batch normalization and dropout,
    followed by a final output layer.
    """
    def __init__(self, vocab_size, embedding_dim, context_size,
                 hidden_dim=256, dropout=0.5,
                 pretrained_embeddings=None, freeze_emb=False):
        
        super(NeuralLM3, self).__init__()
        self.context_size = context_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        # Instead of averaging, we will concatenate the embeddings.
        # Therefore, the input dimension becomes: context_size * embedding_dim.
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        

    def forward(self, x):
        """
        Forward pass:
          1) Embedding lookup
          2) Concatenate context embeddings
          3) Hidden layer (ReLU) + batch norm + dropout
          4) Output layer
        """
        # [batch_size, context_size, embedding_dim]
        embedded = self.embedding(x)
        
        # Concatenate embeddings along the context dimension:
        flat = embedded.view(embedded.size(0), -1)  # [batch_size, context_size * embedding_dim]
        
        h = self.fc1(flat)                         # [batch_size, hidden_dim]
        h = F.relu(h)
        h = self.bn1(h)                            
        h = self.dropout(h)                        
        
        out = self.fc2(h)                          # [batch_size, vocab_size]
        return out


def train_model(model, train_loader, valid_loader, model_id, num_epochs=5, lr=1e-3, device='cpu'):
    
    """
    Train the specified language model, computing losses, accuracy, and perplexity
    on both the training and validation sets.

    Args:
        model (nn.Module): The language model to train.
        train_loader (DataLoader): DataLoader for training samples.
        valid_loader (DataLoader): DataLoader for validation samples.
        model_id (int): ID to save model checkpoints (e.g., "NeuralLM1.pth")
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        device (str): 'cpu' or 'cuda' for GPU training.

    Returns:
        (train_loss_history, valid_loss_history,
         train_acc_history, valid_acc_history,
         train_ppl_history, valid_ppl_history)
    """
    
    model.to(device)
        # Used Adam with weight decay for L2 regularization
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []
    train_ppl_history = []
    valid_ppl_history = []
    
    for epoch in range(num_epochs):
        # ---------------------------
        # Training
        # ---------------------------
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * context.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == target).sum().item()
            total_samples += target.size(0)
        
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_correct / total_samples
        epoch_train_ppl = compute_perplexity(epoch_train_loss)

        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        train_ppl_history.append(epoch_train_ppl)

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for context, target in valid_loader:
                context = context.to(device)
                target = target.to(device)
                
                logits = model(context)
                loss = criterion(logits, target)
                
                val_loss += loss.item() * context.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_ppl = compute_perplexity(epoch_val_loss)

        valid_loss_history.append(epoch_val_loss)
        valid_acc_history.append(epoch_val_acc)
        valid_ppl_history.append(epoch_val_ppl)

        print(f"Epoch [{epoch+1}/{num_epochs}]: "
              f"Train Loss={epoch_train_loss:.4f}, Acc={epoch_train_acc:.4f}, PPL={epoch_train_ppl:.2f} | "
              f"Val Loss={epoch_val_loss:.4f}, Acc={epoch_val_acc:.4f}, PPL={epoch_val_ppl:.2f}")
    torch.save(model.state_dict(), f"NeuralLM{model_id}.pth")
    return (
        train_loss_history,
        valid_loss_history,
        train_acc_history,
        valid_acc_history,
        train_ppl_history,
        valid_ppl_history
    )

def compute_perplexity(loss):
    """Perplexity = exp(loss)"""
    return np.exp(loss)

def generate_next_tokens(model, tokenizer, token2idx, idx2token,
                         initial_text, num_tokens=3, device='cpu'):
    """
    Predict the next `num_tokens` tokens from an initial text using the trained model.
    
    Args:
        model (nn.Module): The trained language model.
        tokenizer (WordPieceTokenizer): Tokenizer to handle text.
        token2idx (dict): token -> int mapping.
        idx2token (dict): int -> token mapping.
        initial_text (str): The starting sentence/phrase.
        num_tokens (int): How many tokens to predict.
        device (str): 'cpu' or 'cuda'.

    Returns:
        generated_tokens (list[str]): A list of the predicted next tokens.
    """
    model.eval()

    # 1) Tokenize input text
    tokens = tokenizer.tokenize(initial_text)
    # 2) Convert recognized tokens -> indices. Could assign [UNK] if missing.
    indices = [token2idx[t] for t in tokens if t in token2idx]

    context_size = model.context_size
    generated_tokens = []

    for _ in range(num_tokens):
        # If we don't have enough tokens to form a context window, break
        if len(indices) < context_size:
            break
        
        # Building the context
        context_indices = indices[-context_size:]
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(context_tensor)
            next_token_id = torch.argmax(logits, dim=1).item()
        
        # Lookup in idx2token
        # If idx2token has int keys, do idx2token[next_token_id].
        predicted_token = idx2token[next_token_id]  
        generated_tokens.append(predicted_token)

        # Add this predicted token's index to the context for next prediction
        indices.append(next_token_id)

    return generated_tokens


if __name__ == "__main__":
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    CONTEXT_SIZE = 2
    DEVICE = "cpu"  # or "cuda"
    VOCAB_SIZE = 10000

    with open("../corpus.txt", "r") as file:
        corpus = file.readlines()
    
    tokenizer = WordPieceTokenizer(corpus, vocab_size=VOCAB_SIZE)
    if os.path.exists("vocabulary_50.txt"):
        with open("vocabulary_50.txt", "r") as file:
            tokenizer.vocab = file.readlines()
        tokenizer.vocab = [word.strip('\n') for word in tokenizer.vocab]
    else:
        tokenizer.construct_vocabulary()

    with open("../task2/word_to_idx.json", "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)

    with open("../task2/idx_to_word.json", "r", encoding="utf-8") as f:
        idx_to_word_str = json.load(f)
    # Convert string keys -> int
    idx_to_word = {int(k): v for k, v in idx_to_word_str.items()}

    vocab_size = len(word_to_idx)
    print(f"Loaded dictionaries: {vocab_size} tokens in word_to_idx.")

    # 4) Load Pretrained Word2Vec Model
    word2vec_model = load_pretrained_word2vec(
        "../task2/word2vec_cbow.pth",
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM
    )

    # 5) Create NeuralLMDataset
    dataset = NeuralLMDataset(
        corpus=corpus,
        tokenizer=tokenizer,
        word2idx=word_to_idx,
        idx2word=idx_to_word,
        context_size=CONTEXT_SIZE
    )

    # Train/Valid Split
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6) Instantiate 3 NeuralLM Models
    # Use the actual .weight tensor for pretrained embeddings
    pretrained_weights = word2vec_model.embeddings.weight

    model1 = NeuralLM1(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        pretrained_embeddings=pretrained_weights,
        freeze_emb=True
    )
    model2 = NeuralLM2(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        hidden_dim=HIDDEN_DIM,
        pretrained_embeddings=pretrained_weights,
        freeze_emb=False
    )
    model3 = NeuralLM3(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        hidden_dim=HIDDEN_DIM,
        pretrained_embeddings=pretrained_weights,
        freeze_emb=False,
        dropout=0.3
    )

    # 7) Train Each Model
    results = {}

    print("\n=== Training NeuralLM1 ===")
    res1 = train_model(model1, train_loader, valid_loader, 1, 
                       num_epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    results["model1"] = res1
    

    print("\n=== Training NeuralLM2 ===")
    res2 = train_model(model2, train_loader, valid_loader, 2,
                       num_epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    results["model2"] = res2

    print("\n=== Training NeuralLM3 ===")
    res3 = train_model(model3, train_loader, valid_loader, 3, 
                       num_epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)
    results["model3"] = res3

    # 8) Plot Train & Val Loss
    epochs_range = range(1, EPOCHS+1)
    plt.figure(figsize=(10, 6))
    for model_name in ["model1", "model2", "model3"]:
        train_losses = results[model_name][0]
        valid_losses = results[model_name][1]
        plt.plot(epochs_range, train_losses, label=f"{model_name} Train Loss")
        plt.plot(epochs_range, valid_losses, label=f"{model_name} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss (All 3 Models)")
    plt.legend()
    plt.savefig("task3_loss_plot.png")
    plt.show()
    model_accuracy = []
    # 9) Print final performance
    for model_name in ["model1", "model2", "model3"]:
        (train_losses, valid_losses,
         train_accs, valid_accs,
         train_ppls, valid_ppls) = results[model_name]
        
        print(f"\nFinal Performance for {model_name}:")
        print(f"  Train Loss: {train_losses[-1]:.4f} | Val Loss: {valid_losses[-1]:.4f}")
        print(f"  Train Acc:  {train_accs[-1]:.4f} | Val Acc:  {valid_accs[-1]:.4f}")
        print(f"  Train PPL:  {train_ppls[-1]:.2f}  | Val PPL:  {valid_ppls[-1]:.2f}")
        model_accuracy.append([valid_accs[-1], model_name])
    model_accuracy.sort(reverse=True)
    print(f"\nBest Model: {model_accuracy[0][1]} with Validation Accuracy: {model_accuracy[0][0]:.4f}")
    best_model = None
    if model_accuracy[0][1] == "model1":
        best_model = model1
    elif model_accuracy[0][1] == "model2":
        best_model = model2
    else:
        best_model = model3
    # 10) Predict Next Tokens
    test_file = "sample_test.txt"
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            test_sentences = [line.strip() for line in f if line.strip()]
        
        print("\n=== Next Token Predictions for test.txt ===")
        for sentence in test_sentences:
            preds = generate_next_tokens(
                model=best_model,
                tokenizer=tokenizer,
                token2idx=word_to_idx,
                idx2token=idx_to_word,
                initial_text=sentence,
                num_tokens=3,
                device=DEVICE
            )
            print(f"Input: {sentence}")
            print(f"Predicted next 3 tokens: {preds}\n")
            print()
    else:
        print("test.txt not found. Skipping next-token predictions.")
