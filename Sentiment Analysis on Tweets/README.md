# Sentiment Analysis on Tweets

This project focuses on sentiment classification of tweets into three categories: positive, negative, and neutral. It involves text preprocessing, exploratory data analysis (EDA), and the implementation of machine learning and deep learning models.

## Project Structure

- **Preprocessing**: Clean tweets using regular expressions, remove stop words, and apply stemming.
- **EDA**: Visualize sentiment distribution, tweet length, and most common words per sentiment.
- **Modeling**:
  - **Logistic Regression** with TF-IDF features
  - **LSTM Neural Network** with word embeddings
- **Evaluation**: Compare models using classification metrics including F1-score, Precision, Recall, Accuracy, and Confusion Matrix.

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt