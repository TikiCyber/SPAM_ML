import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample data for testing
data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham'] * 100,
    'message': [
        'Hey how are you doing today',
        'WINNER! You have won a $1000 prize. Click here now!',
        'Can you pick up some milk on your way home',
        'Congratulations! You have been selected for a free iPhone',
        'Meeting at 3pm tomorrow in conference room'
    ] * 100
}

df = pd.DataFrame(data)

# TODO: Replace with actual dataset
# df = pd.read_csv('spam.csv', encoding='latin-1')
# df = df[['v1', 'v2']]
# df.columns = ['label', 'message']

print(f"Dataset size: {len(df)} emails")
print(f"Spam: {len(df[df['label'] == 'spam'])}, Ham: {len(df[df['label'] == 'ham'])}")

# Convert to binary labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train/test split
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to features using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n" + "="*50)
print("NAIVE BAYES")
print("="*50)

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_pred = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)

print(f"Accuracy: {nb_acc*100:.2f}%")
print(classification_report(y_test, nb_pred, target_names=['Ham', 'Spam']))

print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
lr_pred = lr.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"Accuracy: {lr_acc*100:.2f}%")
print(classification_report(y_test, lr_pred, target_names=['Ham', 'Spam']))

# Compare models
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Naive Bayes: {nb_acc*100:.2f}%")
print(f"Logistic Regression: {lr_acc*100:.2f}%")

# Test on some examples
print("\n" + "="*50)
print("TESTING")
print("="*50)

test_msgs = [
    "Congratulations! You've won a free vacation. Click here to claim.",
    "Hey, want to grab lunch tomorrow?",
    "URGENT: Your account will be closed. Verify now!",
    "Can you send me the project files?"
]

for msg in test_msgs:
    msg_tfidf = vectorizer.transform([msg])
    pred = lr.predict(msg_tfidf)[0]
    print(f"\n{msg}")
    print(f"→ {'SPAM' if pred == 1 else 'HAM'}")
