pip install nltk

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.corpus import wordnet, stopwords

# Download necessary datasets
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')

import nltk
nltk.download()

text = "NLTK is a leading platform for building Python programs to work with human language data."

# Word Tokenization
word_tokens = word_tokenize(text)
print("Word Tokens:", word_tokens)

# Sentence Tokenization
sentence_tokens = sent_tokenize(text)
print("Sentence Tokens:", sentence_tokens)

#whitespace
text = "Tokenization is the first step in NLP."
whitespace_tokens = text.split()
print("Whitespace Tokens:", whitespace_tokens)

#character
text = "Token"
character_tokens = list(text)
print("Character Tokens:", character_tokens)

text1 = "Tokenization is the first step in NLP."

#bigrams
b_bigrams=list(nltk.bigrams(text1))
print("b_bigrams",b_bigrams)
#trigrams
b_trigrams=list(nltk.trigrams(text1))
print("b_trigrams",b_trigrams)
#ngrams
b_ngrams=list(nltk.ngrams(text1,6))
print("b_ngrams",b_ngrams)

from nltk.tokenize import TweetTokenizer

tweet = "Loving the new features of #GPT4! Thanks @OpenAI 😊 https://openai.com/"

# Initialize the TweetTokenizer
tweet_tokenizer = TweetTokenizer()

# Tokenize the tweet
tweet_tokens = tweet_tokenizer.tokenize(tweet)
print("Tweet Tokens:", tweet_tokens)

# PorterStemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in word_tokens]
print("Porter Stemmer:", stemmed_words)

#LancasterStemming
lancaster_stemmer = LancasterStemmer()
lancaster_stemmed_words = [lancaster_stemmer.stem(word) for word in word_tokens]
print("Lancaster Stemmer:", lancaster_stemmed_words)

import nltk
from nltk.stem import SnowballStemmer

# Initialize the Snowball stemmer for English
stemmer = SnowballStemmer("english")

# Words to be stemmed
words = ["running", "runner", "ran", "easily", "fairly"]

# Apply stemming
stems = [stemmer.stem(word) for word in words]

print("Original Words:", words)
print("Stemmed Words:", stems)

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]
print("Lemmatized Words:", lemmatized_words)

pos_tags = pos_tag(word_tokens)
print("POS Tags:", pos_tags)

# Named Entity Recognition
named_entities = ne_chunk(pos_tags)
print("Named Entities:", named_entities)


text = "Apple is looking at buying U.K. startup for $1 billion."

# Tokenization
words = word_tokenize(text)

# POS Tagging
pos_tags = pos_tag(words)

# Stemming (Porter)
porter_stemmed = [PorterStemmer().stem(word) for word in words]

# Lemmatization
lemmatized = [WordNetLemmatizer().lemmatize(word) for word in words]

# Named Entity Recognition
named_entities = ne_chunk(pos_tags)

print("Words:", words)
print("POS Tags:", pos_tags)
print("Porter Stemmed:", porter_stemmed)
print("Lemmatized:", lemmatized)
print("Named Entities:", named_entities)


