import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv

def txt_to_np(path):          ####functie pentru a incarca propozitiile din fisierul text in vectori de cuvinte de tip numpy
    f = open(path, 'r', encoding='utf-8')
    sentences = []
    sentence = f.readline().split()
    sentences.append(sentence)
    while sentence:
        sentence = f.readline().split()
        sentences.append(sentence)
    sentences.pop()
    for i in range(len(sentences)):
        sentences[i].pop(0)
        sentences[i] = [x.lower() for x in sentences[i] if not (len(x)==1)]
    return np.array(sentences)

###incarcarea fisierelor:
train_samples = txt_to_np('data/train_samples.txt')
train_labels = np.loadtxt('data/train_labels.txt', dtype='int', usecols=(1))

validation_samples = txt_to_np('data/validation_samples.txt')
validation_labels = np.loadtxt('data/validation_labels.txt', dtype='int', usecols=(1))
#test_samples = txt_to_np('data/test_samples.txt')

def create_corpus(input):        ###functie care returneaza un singur vector de string-uri(fiecare element este o propozitie obtinuta prin concatenarea cuvintelor dintr-un vector de cuvinte)
    sep = ' '
    corpus = []
    for i in range(len(input)):
        corpus.append(sep.join(input[i]))
    return corpus

###crearea vectorului de propozitii(corpus):
train_samples_joined = create_corpus(train_samples)
validation_samples_joined = create_corpus(validation_samples)
#test_samples_joined = create_corpus(test_samples)

###convertirea datelor(vectorului de propozitii) intr-o matrice de feature-uri de tip TF-IDF:
vectorizer = TfidfVectorizer(token_pattern='\S+')
X = vectorizer.fit_transform(train_samples_joined).toarray()
Y = vectorizer.transform(validation_samples_joined).toarray()
#Y = vectorizer.transform(test_samples_joined).toarray()


###definirea modelului, antrenarea acestuia si generarea predictiilor:
clf = MultinomialNB(alpha = 0.025)
clf.fit(X, train_labels)
pred = clf.predict(Y)

###afisarea scorului obtinut + confusion matrix + classification_report:
print(np.mean(pred == validation_labels))  #acelasi scor daca utilizam f1_score cu average='micro'
print("F1 score: ", f1_score(validation_labels, pred, average='binary'))
print("Confusion matrix:")
print(confusion_matrix(validation_labels, pred))
print("Classification report:")
print(classification_report(validation_labels,pred))

###cod utilizat pentru crearea fisierului .csv cu etichete:
"""id_list = []
id_file = open('data/test_samples.txt', 'r', encoding='utf-8')
for line in id_file:
    id_list.append(line.split(None, 1)[0])
id_file.close()

csvfile = open('data/test_predict.csv', 'w')
filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
filewriter.writerow(['id', 'label'])
for i in range(len(id_list)):
  filewriter.writerow([id_list[i], pred[i]])
csvfile.close()"""