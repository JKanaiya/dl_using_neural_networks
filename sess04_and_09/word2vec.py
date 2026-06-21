# Install necessary libraries
# pip install gensim scikit-learn
import nltk

# Download the 'punkt' resource
nltk.download("punkt")

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups

# Download the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

# Tokenize the text data
tokenized_text = [word_tokenize(text) for text in newsgroups.data]

# Train Word2Vec model
word2vec_model = Word2Vec(
    sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4
)

# Save or load the model
# word2vec_model.save("word2vec_model.bin")
# word2vec_model = Word2Vec.load("word2vec_model.bin")

# Get vector representation of a word
vector = word2vec_model.wv["example"]
print(vector)
