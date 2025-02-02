# Assignment 1

## Task 1: Implement WordPieceTokenizer
Done by Aarzoo (2022008)

### Tasks Covered
- Implemented WordPiece Tokenization, constructing a vocabulary by iteratively merging frequent token pairs from a corpus.
- Preprocessed text by cleaning, splitting into words, and encoding them into subword tokens based on the learned vocabulary.
- Tokenized sentences using the constructed vocabulary, replacing unknown words with [UNK].

## Task 2: Implement Word2Vec
Done by Anushka Srivastava (2022086)

### Tasks Covered
- Created a Word2VecDataset custom class compatible with Pytorch’s Dataloader. The dataset preprocessed the dataset such that the corpus is first tokenized with the help of WordPieceTokenizer from Task 1. Then, the vector embeddings are created with context size 2 and CBOW architecture by using the token indices as per the tokenized corpus. 
- Created and trained a Word2VecModel, which generates word embeddings by learning to predict a word from its surrounding context, which helps to capture the relationships between words so that words that appear in similar contexts end up with similar vector representations. Negative Sampling loss function is used.
- Calculated cosine similarities between two triplets and extracted pairs with maximum similarity and minimum similarity.

## Task 3: Train a Neural LM
Done by Anish Jain (2022077)

### Tasks Covered
- Loaded pretrained embeddings to initialize word representations.
- Created a dataset using WordPiece tokenization and trained three different neural models.
- Tracked loss, accuracy, and perplexity, selecting the best-performing model.
- Used the trained model to generate predicted tokens for given input sentences.

## Running the code
To run each `.py` file individually, navigate to each directory and run the file using the following command:
```
cd task
python task.py
```

To run all the python files together, navigate to the root of the repository and run the `.bat` file to run the script to run all 3 files together. <b>This will only work in Windows-based systems.</b>
```
run_tasks.bat
```

Delete the `vocabulary_50.txt` file if you wish to change the `vocab_size` as this file is only applicable for `vocab_size` of 10000.

## Project Repository Structure

### task1
- `sample_test.json:` This contains the sample sentences for testing the WordPieceTokenizer.
- `sample_vocabulary.txt:` This contains an example vocabulary file which servers as template for vocabulary_50.txt.
- `task1.py:` Source code for task1.
- `tokenized_50.json:` Output for `sample_test.json`.
- `vocabulary_50.txt:` Vocabulary extracted from the corpus.

### task2
- `idx_to_word.json:` Saves the index to word mapping obtained from preprocessing the data.
- `loss.png:` Train vs validation vs epoch graph obtained after training the model.
- `task2.py:` Source code for task2.
- `vocabulary_50.txt:` This servers as cache for vocabulary with vocabulary size 10000. This has to be deleted if the vocab_size if changed to reconstruct the vocabulary accordingly.
- `word_to_idx.json:` Saves the word to index mapping obtained from preprocessing the data.
- `word2vec_cbow.pth:` Trained Word2Vec Model checkpoint.

### task3
- `NeuralLM1.pth:` Trained model checkpoint for the first Neural Language Model.
- `NeuralLM2.pth:` Trained model checkpoint for the second Neural Language Model.
- `NeuralLM3.pth:` Trained model checkpoint for the third Neural Language Model.
- `sample_test.txt:` Sample file for testing the next token prediction task.
- `task3_loss_plot.png:` Train vs validation vs epoch graph for all the three models obtained after training.
- `task3.py:` Source code for task3.
- `vocabulary_50.txt:` This servers as cache for vocabulary with vocabulary size 10000. This has to be deleted if the vocab_size if changed to reconstruct the vocabulary accordingly.