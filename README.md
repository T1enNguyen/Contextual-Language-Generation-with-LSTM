# Vietnamese Text Generation using LSTM

This project implements a Deep Learning model using **Long Short-Term Memory (LSTM)** networks to perform **automated Vietnamese text generation** (Next Word Prediction). The model predicts the next word in a sequence based on a context of the preceding 50 words.

## 📌 Project Overview

The text generation problem is approached as a **multi-class classification** task, where the number of output classes equals the size of the vocabulary.

* **Input:** A sequence of 50 Vietnamese words.
* **Output:** The predicted next word in the dictionary.
* **Core Technology:** TensorFlow/Keras, LSTM, Word Embeddings.

## 📂 Dataset

The model is trained on a dataset of Vietnamese news articles, pre-processed into sequences:

* **`sequences_digit.pkl`**: Contains the dataset converted into integer sequences.
* **`tokenizer.pkl`**: A pre-built tokenizer containing **8,962 Vietnamese words**.

## 🛠️ Installation & Requirements

To run this project, you need Python installed along with the following libraries:

```bash
pip install numpy tensorflow keras pyvi
