import json
import os
import re

# Preprocess dataset

def preprocess_tokens(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens

def preprocess_data(data):
    preprocessed_data = []
    for d in data:
        for term in d['aspect_terms']:
            new_data = {}
            tokens = preprocess_tokens(d["sentence"])
            new_data["tokens"] = tokens
            new_data["polarity"] = term["polarity"]
            new_data["aspect_term"] = [term["term"]]
            new_data["index"] = tokens.index(preprocess_tokens(term["term"])[0])
            preprocessed_data.append(new_data)
    return preprocessed_data

if not os.path.exists('train_task_2.json'):
    with open('train.json') as f:
        train_data = json.load(f)
    preprocessed_data = preprocess_data(train_data)
    with open('train_task_2.json', 'w') as f:
        json.dump(preprocessed_data, f, indent=4)
        
if not os.path.exists('val_task_2.json'):
    with open('val.json') as f:
        val_data = json.load(f)
    preprocessed_data = preprocess_data(val_data)
    with open('val_task_2.json', 'w') as f:
        json.dump(preprocessed_data, f, indent=4)
        
train_data = json.load(open('train_task_2.json'))
val_data = json.load(open('val_task_2.json'))
print(len(train_data))
print(len(val_data))
print(train_data[0])
print(val_data[0])