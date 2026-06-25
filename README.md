Scam Text Transformer

This project is a neural network built with PyTorch to detect scam and phishing messages from text.

It uses a Transformer-style encoder to classify whether a message is safe or malicious.

Overview

The model learns patterns in text messages such as Discord messages and phishing attempts. It outputs a probability indicating whether a message is a scam.

Main idea

- Convert text messages into token sequences
- Embed tokens into vectors
- Process sequences using a Transformer encoder block
- Aggregate sequence information
- Predict a binary label (scam or not scam)

Dataset

The dataset is built from multiple sources including:
- Discord phishing datasets from Hugging Face
- Custom CSV datasets containing scam and non-scam messages

All messages are cleaned, tokenized, and converted into fixed-length sequences.

Preprocessing

- Text is lowercased
- Punctuation is separated from words
- A vocabulary is built from training data
- Each message is converted into a sequence of token IDs
- Sequences are padded or truncated to a fixed length of 50 tokens

Model architecture

The model contains:

- Token embedding layer
- Learnable positional embeddings
- Transformer encoder block with multi-head attention
- Feedforward neural network
- Layer normalization
- Mean pooling over sequence output
- Linear classifier for binary prediction

Training

The model is trained using binary cross entropy with logits loss.

Optimization is done using Adam optimizer.

The dataset is shuffled and split into training and testing sets.

Evaluation

After training, the model is evaluated on a held-out test set using:
- Loss
- Accuracy

Output is converted to probabilities using sigmoid and thresholded for classification.

Usage

Train the model using the provided script, then save the trained weights.

The model can then be loaded to classify new text messages as scam or not scam.

Limitations

This model is a basic baseline and not production-ready.

It may struggle with:
- Very short messages
- Highly obfuscated scam text
- Class imbalance in datasets
- Domain shift between different data sources

It is intended for educational and experimental use only.
