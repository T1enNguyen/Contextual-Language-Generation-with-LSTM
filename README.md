# 🇻🇳 Neural Text Generation using LSTM

## 🌟 Overview

[cite_start]This project implements a Recurrent Neural Network (RNN) model with a **Long Short-Term Memory (LSTM)** architecture to address the **Text Generation** problem[cite: 1, 3]. [cite_start]Text generation is a foundational task for Large Language Models (LLMs) [cite: 3][cite_start], where the model learns to predict the next word in a sequence based on a given context[cite: 4].

[cite_start]The model is trained on a substantial Vietnamese dataset, enabling it to automatically generate coherent, grammatically sound, and contextually relevant text sequences[cite: 4].

## 🎯 Project Goals

1.  [cite_start]**Model Training:** Design and train a high-performance LSTM model to learn the probability distribution of words based on the input context[cite: 5, 58].
2.  [cite_start]**Efficient Data Flow:** Implement a `data_generator` function to create training data batches randomly and continuously, optimizing memory usage for large datasets[cite: 35].
3.  [cite_start]**Application:** Utilize the trained model weights to automatically generate continuous sequences of Vietnamese text[cite: 60].

## ⚙️ Technology & Model Architecture

### 1. Input/Output Definition

* [cite_start]**Input (Context):** A sequence of the preceding 50 words (represented as integer IDs)[cite: 6, 40].
* [cite_start]**Output:** Prediction of the single subsequent word (represented via One-Hot Encoding)[cite: 7, 32].
* [cite_start]**Problem Type:** Multi-class classification, where the number of output classes equals the size of the vocabulary ($N_{vocab}$)[cite: 8, 42].

### 2. Model Architecture

The model is built using Keras/TensorFlow and includes the following key layers:

| Layer | Type | Purpose |
| :--- | :--- | :--- |
| **Input Layer** | `Embedding` | Converts word IDs into dense vector representations. [cite_start]**$Input\_length=50$**[cite: 37, 40]. |
| **Hidden Layers** | `LSTM` (2-3 layers) | [cite_start]Learns long-term dependencies and sequential patterns in the text[cite: 36]. |
| **Output Layer** | `Dense` | [cite_start]The classifier layer with **`softmax`** activation to predict the probability distribution across all $N_{vocab}$ words[cite: 41, 42]. |

### 3. Data Sources

The project utilizes pre-processed Vietnamese text data:

* [cite_start]**Sequences:** `sequences_digit.pkl` [cite: 10] [cite_start](Vietnamese sentences converted to numeric sequences [cite: 12]).
* [cite_start]**Tokenizer/Vocabulary:** `tokenizer.pkl` [cite: 15] [cite_start](A pre-built dictionary containing **8,962 Vietnamese words** [cite: 17]).

## 💻 Setup & Usage Guide

### 1. Prerequisites

* Install the required libraries:
    ```bash
    !pip install numpy keras tensorflow pyvi
    ```
* Download the data and tokenizer files:
    * [cite_start]`sequences_digit.pkl` [cite: 11]
    * [cite_start]`tokenizer.pkl` [cite: 16]

### 2. Training Process

1.  [cite_start]**Load Data:** Use the `pickle` library to load `sequences_digit` [cite: 13, 14] [cite_start]and `tokenizer`[cite: 18, 19].
2.  **Define Model:** Construct the LSTM model based on the chosen architecture.
3.  [cite_start]**Train:** Use the `data_generator` [cite: 23-34] [cite_start]function and `model.fit()` with `ModelCheckpoint` [cite: 49-52] [cite_start]callback to save the best model weights based on accuracy (`acc`)[cite: 54].
    ```python
    model.fit(data_generator(sequences_digit, batch_size), 
              steps_per_epoch=(len(sequences_digit)//batch_size), 
              epochs=10, 
              callbacks=callbacks_list)
    ```

### 3. Text Generation Application

[cite_start]After training, the saved weights (`model.h5` [cite: 46]) are used for generation:

1.  [cite_start]**Load Model:** Load the trained model using `keras.models.load_model()`[cite: 67, 69].
2.  [cite_start]**Preprocessing:** Define `clean_document` and `preprocess_input` functions [cite: 76, 86] [cite_start]which use `pyvi.ViTokenizer` [cite: 73] [cite_start]for Vietnamese word segmentation and `pad_sequences` [cite: 74, 91] to ensure input length is 50.
3.  [cite_start]**Generate Text:** The `generate_text(text_input, n_words)` function [cite: 94] [cite_start]iteratively predicts the next word (`np.argmax` [cite: 98][cite_start]), appends it to the sequence, and truncates the oldest word [cite: 99, 100][cite_start], effectively sliding the context window to generate a continuous stream of text[cite: 96, 97].
