import pandas
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pandas.read_excel('Learn.xlsx')
df = df.fillna('')
desc, cat = df["Desc"], df["Cat"]
stopwords_set = set(stopwords.words("russian"))
tokenize_desc = [word_tokenize(word) for word in desc.values]
for i in range(len(tokenize_desc)):
    tokenize_desc[i] = " ".join(word for word in tokenize_desc[i] if word not in stopwords_set)

vector = CountVectorizer()
X = vector.fit_transform(tokenize_desc)

X_train, X_test, y_train, y_test = train_test_split(X, cat, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)


def check_accessories():
    y_pred = model.predict(X_test)

    # Оценка качества модели
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)


def predict_category(user_input):
    user_input_processed = " ".join(word for word in word_tokenize(user_input) if word not in stopwords_set)
    user_vector = vector.transform([user_input_processed])
    prediction = model.predict(user_vector)
    return prediction[0]


def response_user():
    while True:
        user_question = input("Введите ваш запрос (или 'выход' для завершения): ")
        if user_question.lower() == "выход":
            break
        predicted_category = predict_category(user_question)
        print(f"Предсказанная категория: {predicted_category}")


if __name__ == '__main__':
    response_user()
