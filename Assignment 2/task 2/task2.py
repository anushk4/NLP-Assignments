import json
import re
import os
from torch.utils.data import Dataset

class ABSADataset(Dataset):
    def __init__(self, file_path, saved_file_path = None):
        """
        Initialize the dataset by loading and preprocessing the data.
        """
        with open(file_path, 'r') as f:
            raw_data = json.load(f)  # Load raw JSON data
        if saved_file_path is not None and os.path.exists(saved_file_path):
            with open(saved_file_path, 'r') as f:
                self.data = json.load(f)  # Load preprocessed data
            print("Preprocessed data loaded from", saved_file_path)
        else:
            self.data = self.preprocess_data(raw_data)  # Preprocess and store in self.data
            if saved_file_path is not None:
                with open(saved_file_path, 'w') as f:
                    json.dump(self.data, f, indent=4)  # Save preprocessed data
                print("Preprocessed data saved to", saved_file_path)    
        
    """
    Preprocessing rule:
    - Extract the index of each token in the original sentence by taking multiple spaces into account
    - Remove leading and trailing punctuation like ,.!()-{}[]"";:
    - Remove single quotes
    - Skip words that become empty (which means they were only punctuation)
    - Allow an error range of +/- 2 tokens when matching the aspect term
    - Check if the token's prefix matches the start of the aspect term
    - Return -1 if no match is found
    """

    def preprocess_data(self, data):
        """
        Tokenizes and processes data into required format.
        """
        preprocessed_data = []
        for d in data:
            tokens = self.tokenize_data(d["sentence"])
            for term in d['aspect_terms']:
                new_data = {
                    "tokens": [t[0] for t in tokens],  # Extract token words
                    "polarity": term["polarity"],
                    "aspect_term": [term["term"]],
                    "index": self.find_aspect_index(tokens, int(term["from"]), term["term"])
                }
                preprocessed_data.append(new_data)
        return preprocessed_data

    def tokenize_data(self, sentence):
        """
        Tokenizes sentence while preserving exact indices.
        """
        tokens_idx = []
        i = 0
        word = ''
        start_idx = -1

        while i < len(sentence):
            while i < len(sentence) and sentence[i] != ' ':
                if not word:
                    start_idx = i
                word += sentence[i]
                i += 1
            if word:
                tokens_idx.append((word, start_idx))
            word = ''
            i += 1

        return self.clean_tokens(tokens_idx)

    def clean_tokens(self, tokens_idx):
        """
        Cleans tokens by removing punctuation and handling contractions.
        """
        cleaned_tokens = []
        for word, idx in tokens_idx:
            stripped_word = re.sub(r'^[.!(),\[\]{}\'":;-]+|[.!(),\[\]{}\'":;-]+$', '', word)
            stripped_word = stripped_word.replace("'", "")

            if stripped_word:  # Skip empty tokens
                cleaned_tokens.append((stripped_word, idx))
        return cleaned_tokens

    def find_aspect_index(self, tokens, aspect_from, aspect_term):
        """
        Finds the token index corresponding to the aspect term with an error range of ±2.
        """
        aspect_term = self.tokenize_data(aspect_term)
        for i in range(len(tokens)):
            if aspect_from - 2 <= tokens[i][1] <= aspect_from + 2:
                if tokens[i][0].startswith(aspect_term[0][0]):  # Check prefix match
                    return i
        return -1

    def __len__(self):
        """
        Returns the total number of data samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single preprocessed data sample.
        """
        return self.data[idx]

# Usage Example
train_dataset = ABSADataset("train.json", "train_task_2.json")
val_dataset = ABSADataset("val.json", "val_task_2.json")

print(len(train_dataset))
print(len(val_dataset)) 
print(train_dataset[0])
print(val_dataset[0])  
