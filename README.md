# Build A Large Language Model from Scratch Implementation
This repository contains my full implementation of the chapters from Sebastian Raschka's book "Build a Large Language Model from Scratch". It serves as both a learning project and a complete reference for constructing a GPT-style language model using PyTorch.

This project walks through all the foundational steps in building a transformer-based language model — from tokenization and model architecture to pretraining, fine-tuning, and evaluation. It mirrors the structure and chapters of the book, ensuring each concept is implemented and explained with clarity. It also contains interactive notebooks 
that give the reader can use to further their understanding on complex topics presented in the book.

Using the concepts presented in this book, we also implemented a fully custom pretraining pipeline tailored for efficient and scalable training of GPT-style models from scratch. The pipeline mirrors real-world LLM training strategies and includes multiple components designed for modularity and performance.

## Chapters 2-7
This part of the project is the implementation of the chapters from the book, each chapter focuses on a different part of building and understanding large language models — from foundational concepts and attention mechanisms, to pretraining from scratch and fine-tuning for downstream tasks like classification and instruction following.

### Chapter 2 - Working with Text Data
* Explains why neural networks cannot work directly with raw text and require numeric embeddings
* Describes the process of text preprocessing: tokenization, token to token ID conversion, and embeddings
* Demonstrates practical implementation of tokenization and conversion of text into model-ready input sequences

### Chapter 3 - Coding Attention Mechanisms
* Introduces the problem of modeling long-range dependencies in sequences, highlighting the limitations of RNN-based encoder-decoder architectures
* Starts with a simple, non-trainable self-attention mechanism
* Builds toward multi-head self-attention, which serves as a key building block in transformers
* Emphasizes the importance of context vectors and attention weights in capturing relationships between tokens

### Chapter 4 – Implementing a GPT Model from Scratch to Generate Text
* Breaks down the internal structure of a transformer block, layer by layer
* Implements a compact GPTModel in PyTorch and demonstrates its use in generating text sequences token by token
* Focuses on the importance of causal masking and output projection to generate coherent language

### Chapter 5 – Pretraining on Unlabeled Data
* Sets up a training pipeline for autoregressive causal language modeling (CLM)
* Implements tokenization → embedding → forward pass → output → loss calculation
* Highlights the importance of monitoring loss curves and tuning learning rates during pretraining

### Chapter 6 – Fine-Tuning for Classification
* Explains different fine-tuning strategies: classification vs. instruction fine-tuning
* Demonstrates classification fine-tuning using a spam/ham text message dataset
* Fine-tunes the pretrained GPT model by attaching a classification head and training it using labeled examples

### Chapter 7 - Fine-Tuning to Follow Instructions
* Focuses on instruction fine-tuning using a dataset of input-instruction-output triples
* Explains how formatting (prompt engineering) influences LLM behavior
* Prepares the model for instruction-following behavior via supervised fine-tuning on these prompt-response pairs

## [GPT2 Pretraining Pipeline](https://github.com/sdhaduk/GPT2small-pretraining)
This is the link to the repository that contains the pipeline. This pipeline was created by taking the code we implemented from the book and modularizing it, then building upon it.

### Data Preprocessing 
* Chose the Project Gutenberg corpora as the pretraining dataset
* Cleaned the dataset by removing roman numeral tags, illustration captions, tables, ASCII dividers, decorative lines, and all-caps titles
* Filtered the dataset by removing non english books, picture books, books with low text count
* Tokenized the raw text books with BPE, turned them into tensors, then saved these tensors as files to be used for pretraining
* Combined cleaned documents, seperating them by <|endoftext|> to preserve semantic boundaries, into .pt files of customizable size

### Model Train Loop
* Added warmup, cosine decay, and gradient clipping to stabalize training
* Mixed precision training by using torch.amp for autocast and GradScaler to reduce memory and improve speed of training on an RTX GPU
* Added ability to interrupt model training and resume from where training left off
* Uses GaLoreAdamW for memory-efficient training with low-rank gradient projections

### Pretrained Model Evaluation
Our model, GPT2-small, was trained using this pipeline on about 5 GB of text data, which took roughly 100 hours on an RTX 4060 Ti 16GB.

#### Train and Validation Loss
This image below shows the loss on the train and validation set during training.
![loss](https://github.com/user-attachments/assets/03992f54-e045-4485-991b-edc578614a39)

The image shows how loss values for both validation and training had a major shift everytime a new file was opened to be trained on. This makes sense because text can vary significantly from book to book.

#### Learning Rate
This image below shows the value of the learning rate during training. 
![lr](https://github.com/user-attachments/assets/8e0c8ba8-7019-4bbd-b6e9-f8167f7d69e8)

The linear increase in learning rate is due to the linear warm up rate for the first 10% of steps, after which cosine decay kicks in and the learning rate decays until it hits the assigned minimum.

### Perplexity
We use the wikitext2 dataset to calculate the perplexity of the model, this dataset is commonly used to evaluate perplexity of LLMs after pretraining.

The perplexity of our pretrained model was 3485.89

This extremely high perplexity of our model is a clear sign that it was not trained on enough text data, as the original GPT2-small was trained on 40GB of data, while ours was only trained on 5GB. The quality of the data also factors into this as the raw text data from Project Gutenberg is variable, some texts being low quality while others are high quality.

