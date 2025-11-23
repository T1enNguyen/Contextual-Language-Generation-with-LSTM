# 🇻🇳 Neural Text Generation using LSTM

## 🌟 Overview

This project implements a Recurrent Neural Network (RNN) model with a **Long Short-Term Memory (LSTM)** architecture to address the **Text Generation** problem. Text generation is a foundational task for Large Language Models (LLMs), where the model learns to predict the next word in a sequence based on a given context.

The model is trained on a substantial Vietnamese dataset, enabling it to automatically generate coherent, grammatically sound, and contextually relevant text sequences.

## 🎯 Project Goals

1.  **Model Training:** Design and train a high-performance LSTM model to learn the probability distribution of words based on the input context.
2.  **Efficient Data Flow:** Implement a `data_generator` function to create training data batches randomly and continuously, optimizing memory usage for large datasets.
3.  **Application:** Utilize the trained model weights to automatically generate continuous sequences of Vietnamese text.

## ⚙️ Technology & Model Architecture

### 1. Input/Output Definition

* **Input (Context):** A sequence of the preceding 50 words (represented as integer IDs).
* **Output:** Prediction of the single subsequent word (represented via One-Hot Encoding).
* **Problem Type:** Multi-class classification, where the number of output classes equals the size of the vocabulary ($N_{vocab}$).

### 2. Model Architecture

The model is built using Keras/TensorFlow and includes the following key layers:

| Layer | Type | Purpose |
| :--- | :--- | :--- |
| **Input Layer** | `Embedding` | Converts word IDs into dense vector representations. **$Input\_length=50$**. |
| **Hidden Layers** | `LSTM` (2-3 layers) | Learns long-term dependencies and sequential patterns in the text. |
| **Output Layer** | `Dense` | The classifier layer with **`softmax`** activation to predict the probability distribution across all $N_{vocab}$ words. |

### 3. Data Sources

The project utilizes pre-processed Vietnamese text data:

* **Sequences:** `sequences_digit.pkl` (Vietnamese sentences converted to numeric sequences).
* **Tokenizer/Vocabulary:** `tokenizer.pkl`(A pre-built dictionary containing **8,962 Vietnamese words**).

## 💻 Setup & Usage Guide

### 1. Prerequisites

* Install the required libraries:
    ```bash
    !pip install numpy keras tensorflow pyvi
    ```
* Download the data and tokenizer files:
    * `sequences_digit.pkl`
    * `tokenizer.pkl` 

### 2. Training Process

1.  **Load Data:** Use the `pickle` library to load `sequences_digit` and `tokenizer`.
2.  **Define Model:** Construct the LSTM model based on the chosen architecture.
3.  **Train:** Use the `data_generator` function and `model.fit()` with `ModelCheckpoint` callback to save the best model weights based on accuracy (`acc`).
    ```python
    model.fit(data_generator(sequences_digit, batch_size), 
              steps_per_epoch=(len(sequences_digit)//batch_size), 
              epochs=10, 
              callbacks=callbacks_list)
    ```

### 3. Text Generation Application

After training, the saved weights (`model.h5`) are used for generation:

1.  **Load Model:** Load the trained model using `keras.models.load_model()`.
2.  **Preprocessing:** Define `clean_document` and `preprocess_input` functions which use `pyvi.ViTokenizer` for Vietnamese word segmentation and `pad_sequences` to ensure input length is 50.
3.  **Generate Text:** The `generate_text(text_input, n_words)` function iteratively predicts the next word (`np.argmax`), appends it to the sequence, and truncates the oldest word, effectively sliding the context window to generate a continuous stream of text.
