import re
from collections import defaultdict
import json

''' 
Reference :  https://huggingface.co/learn/nlp-course/en/chapter6/6
'''

class WordPieceTokenizer:
    def __init__(self, corpus, vocab_size):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocab = ["[PAD]", "[UNK]"]
        self.splits = {}
        self.word_freqs = defaultdict(int)
    
    def preprocess_data(self):
        """ Preprocessing the data """
        cleaned_corpus = []
        for text in self.corpus:
            text = re.sub(r'([^\w\s])', r' \1 ', text)
            cleaned_corpus.append(text)
        
        return cleaned_corpus

    def construct_vocabulary(self):
        """ Constructing vocabulary based on the WordPiece algorithm """
    
        cleaned_corpus = self.preprocess_data()

        for text in cleaned_corpus:
            words = text.split()  # Spliting into words based on spaces
            for word in words:
                self.word_freqs[word] += 1

        # Constructing the vocabulary from the initial characters and special tokens
        alphabet = []
        for word in self.word_freqs.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")
        # alphabet.sort()
        self.vocab = ["[PAD]", "[UNK]"] + alphabet.copy()

         # Initializing splits dictionary where each word is split into characters prefixed with "##" except the first
        self.splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.word_freqs.keys()
        }

        # WordPiece algorithm to build the vocabulary 
        while len(self.vocab) < self.vocab_size:
            pair_scores = self.compute_pair_scores()
            if not pair_scores:
                break
            best_pair, max_score = self.get_best_pair(pair_scores)
            if best_pair is None:
                break
            self.merge_pair(best_pair)
            new_token = self.create_new_token(best_pair)
            self.vocab.append(new_token)

        # Saving the final vocabulary to a file
        with open(f"vocabulary_50.txt", "w") as vocab_file:
            for token in self.vocab:
                vocab_file.write(token + "\n")

    def compute_pair_scores(self):
        """ Compute the pair scores for WordPiece merges """
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)

        # Count frequencies of individual letters and pairs
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        # pair scores
        pair_scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }

        return pair_scores

    def get_best_pair(self, pair_scores):
        """ Get the best pair (with highest score) """
        best_pair = None
        max_score = None
        for pair, score in pair_scores.items():
            if max_score is None or score > max_score:
                best_pair = pair
                max_score = score
        return best_pair, max_score

    def merge_pair(self, pair):
        """ Merging the best pair in the splits dictionary """
        a, b = pair
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split

    def create_new_token(self, pair):
        """ Creating a new token by merging the pair """
        a, b = pair
        return a + b[2:] if b.startswith("##") else a + b


    def tokenize(self, sentence):
        """ Tokenize a sentence using the WordPiece algorithm """
        # Preprocessing the sentence to separate punctuation
        sentence = re.sub(r'([^\w\s])', r' \1 ', sentence)  # Adding spaces around punctuation
        sentence = re.sub(r'\s+', ' ', sentence)  # Removing extra spaces

        words = sentence.split()

        tokenized_sentence = []
        for word in words:
            tokens = self.encode_word(word)
            tokenized_sentence.extend(tokens)

        return tokenized_sentence

    def encode_word(self, word):
        """ Encoding a word into tokens using the WordPiece vocabulary """
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens


if __name__ == "__main__":

    with open("corpus.txt", "r") as file:
        corpus = file.readlines()

    # Instantiating WordPieceTokenizer
    tokenizer = WordPieceTokenizer(corpus, vocab_size=10000)

    # Constructing vocabulary
    tokenizer.construct_vocabulary()

    with open("sample_test.json", "r") as test_file:
        test_samples = json.load(test_file)

    # Tokenizing sentences and save results
    tokenized_results = []
    for sample in test_samples:
        tokenized_sentence = tokenizer.tokenize(sample["sentence"])
        tokenized_results.append({"id": sample["id"], "tokens": tokenized_sentence})

    # Saving tokenized output to JSON
    with open("tokenized_50.json", "w") as output_file:
        json.dump(tokenized_results, output_file, indent=4)

    print("Tokenization completed and saved to tokenized_50.json!")
