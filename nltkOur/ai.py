import csv
import pandas
import pandas as pd
import numpy as np
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

max_words = 10000
random_state = 57

data = pd.read_excel("Learn(full).xlsx")
data['Desc'] = data['Desc'].fillna('')

print(data[0:3])


def preprocess(text, stop_words, punctuation_marks, morph):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text


punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']
stop_words = stopwords.words("russian")
morph = pymorphy3.MorphAnalyzer()

data['Preprocessed_Desc'] = data.apply(lambda row: preprocess(row['Desc'], punctuation_marks, stop_words, morph),
                                       axis=1)

words = Counter()

for txt in data['Preprocessed_Desc']:
    words.update(txt)

# Словарь, отображающий слова в коды
word_to_index = dict()
# Словарь, отображающий коды в слова
index_to_word = dict()

for i, word in enumerate(words.most_common(max_words - 2)):
    word_to_index[word[0]] = i + 2
    index_to_word[i + 2] = word[0]


def text_to_sequence(txt, word_to_index):
    seq = []
    for word in txt:
        index = word_to_index.get(word, 1)  # 1 означает неизвестное слово
        # Неизвестные слова не добавляем в выходную последовательность
        if index != 1:
            seq.append(index)
    return seq


data['Sequences'] = data.apply(lambda row: text_to_sequence(row['Preprocessed_Desc'], word_to_index), axis=1)

train, test = train_test_split(data, test_size=0.2)

x_train_seq = train['Sequences']
z_train = train['Cat']
x_test_seq = test['Sequences']
z_test = test['Cat']


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for index in sequence:
            results[i, index] += 1.
    return results


x_train = vectorize_sequences(x_train_seq, max_words)

x_test = vectorize_sequences(x_test_seq, max_words)

dt = DecisionTreeClassifier(class_weight='balanced')

dt1 = DecisionTreeClassifier(class_weight='balanced')

dt1.fit(x_train, z_train)


def test():
    outData = pd.read_csv("C.csv", encoding='Windows-1251', sep=';')

    outData['Preprocessed_desc'] = outData.apply(
        lambda row: preprocess(row['Desc'], punctuation_marks, stop_words, morph),
        axis=1)

    outData['sequences'] = outData.apply(lambda row: text_to_sequence(row['Preprocessed_desc'], word_to_index), axis=1)
    desc = outData['sequences']
    x_desc = vectorize_sequences(desc, max_words)

    predictionDt = dt.predict(x_desc)

    predictionDt1 = dt1.predict(x_desc)

    outDataDt = outData

    outDataDt['Cat'] = predictionDt1

    columns_to_include = ["Desc", "Cat"]

    filtered_dfDt = outDataDt[columns_to_include]
    filtered_dfDt.to_csv('NewDt.csv', index=False, sep=';', encoding='Windows-1251')


