# Sentiment Analysis on Tweets

This project aims to build a machine learning model to classify the sentiment of tweets into three categories: positive, negative, and neutral.

## Objective

The primary goal is to develop a classification model that can accurately predict tweet sentiment while addressing challenges related to noisy text data and class imbalance. The project focuses on data preprocessing, exploratory data analysis (EDA), model training, and evaluation using appropriate performance metrics.

## Dataset

- Source: [Publicly available Twitter Sentiment Analysis dataset (e.g., from Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Size: Approximately 80 MB
- Features: Tweet text and sentiment labels (positive, negative)

## Project Workflow

### 1. Data Preprocessing
- Clean tweets by removing URLs, mentions, hashtags, special characters, and stop words
- Perform tokenization and stemming
- Convert text data into numeric features using TF-IDF vectorization
- Split data into training and testing sets (80/20)

### 2. Exploratory Data Analysis (EDA)
- Visualize sentiment distribution using bar charts and pie charts
- Generate word clouds for each sentiment category
- Analyze tweet length distribution

### 3. Model Training
- Train two classification models:
  - Logistic Regression with TF-IDF features
  - LSTM neural network with word embeddings (e.g., GloVe)
- Tune hyperparameters for at least one model

### 4. Model Evaluation
- Evaluate models using metrics:
  - Precision
  - Recall
  - F1-score
  - Accuracy
- Visualize confusion matrices and compare F1-scores

## Key Findings

- Logistic Regression performs well with TF-IDF features on noisy text.
- LSTM benefits from word embeddings but requires more training time and tuning.
- F1-score is the preferred metric due to class imbalance.
- Preprocessing steps like stemming and stop word removal significantly improve model performance.

## Future Improvements

- Experiment with transformer-based models such as BERT
- Use pretrained embeddings like Word2Vec or FastText
- Apply data augmentation and advanced handling of social media language (slang, emojis)
- Explore ensemble models for improved accuracy

## Requirements

- Python 3.7+
- Libraries: pandas, numpy, matplotlib, seaborn, wordcloud, nltk, scikit-learn, tensorflow, keras, plotly

## How to Run

1. Install dependencies using pip:

```bash
pip install -r requirements.txt

2. Run the Jupyter Notebook or Python script:
jupyter notebook sentiment_analysis_tweets.ipynb

