# Email Spam Classifier

A machine learning project that classifies emails as spam or ham (non-spam) using Natural Language Processing and supervised learning algorithms.

## Overview

Built two classification models (Naive Bayes and Logistic Regression) to detect spam emails with 94% accuracy on a dataset of 5,000+ messages.

## Features

- **TF-IDF Vectorization**: Converts text to numerical features based on word importance
- **Dual Model Comparison**: Implements both Naive Bayes and Logistic Regression
- **Performance Metrics**: Detailed accuracy, precision, recall, and F1-score reporting
- **Real-time Prediction**: Test classifier on custom messages

## Tech Stack

- **Python 3.x**
- **scikit-learn**: Machine learning models and preprocessing
- **pandas**: Data manipulation and analysis
- **TfidfVectorizer**: Feature extraction from text

## Installation

```bash
pip install pandas scikit-learn
```

## Dataset

Uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) containing 5,574 SMS messages labeled as spam or ham.

## Usage

```bash
python spam_classifier.py
```

## Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 94.2% |
| Logistic Regression | 94.8% |

## How It Works

1. **Data Preprocessing**: Load and clean email/SMS messages
2. **Feature Extraction**: Convert text to TF-IDF vectors (top 3000 features)
3. **Model Training**: Train Naive Bayes and Logistic Regression classifiers
4. **Evaluation**: Compare model performance on test set
5. **Prediction**: Classify new messages in real-time

## Example

```python
Message: "Congratulations! You've won a free vacation."
Prediction: SPAM

Message: "Hey, want to grab lunch tomorrow?"
Prediction: HAM
```

## Future Improvements

- Add deep learning models (LSTM, BERT)
- Implement feature engineering for email metadata
- Build web interface for real-time classification
