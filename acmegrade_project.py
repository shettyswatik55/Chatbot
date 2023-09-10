data={"intents":[
    {"tag":"greeting",
     "patterns":["Hello","How are you?","Hi There","Hi","What's up"],
     "responses":["Howdy Partner!","Hello","How are you doing?","Greetings!","How do you do"]
        },
    {"tag":"age",
     "patterns":["how old are you","when is your birthday","when was you born"],
     "responses":["I am 20 years old","I was born in 2003","My birthday is June 2nd and I was born in 1996","02/06/2003"]
        },
    {"tag":"date",
     "patterns":["what are you doing this weekend",
                "do you want to hangout sometime?","what are your plans for this week"],
     "responses":["I am available this week","I don't have any plans","I am not busy"]
        },
    {"tag":"name",
     "patterns":["what's your name","what are you called","who are you"],
     "responses":["My name is Swastey","I'm Swastey","Swastey"]
        },
    {"tag":"goodbye",
     "patterns":["bye","g2g","see ya","adios","cya","hoghthini"],
     "responses":["It was nice speaking to you","See you later","Speak Soon"]
        },
]}
import json
import string
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
doc_x=[]
doc_y=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens=nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])


words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

words=sorted(set(words))
classes=sorted(set(classes))

print(words)
print(classes)
print(doc_x)
print(doc_y)

training=[]
out_empty=[0]*len(classes)


for idx, doc in enumerate(doc_x):
    bow=[]
    text=lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row=list(out_empty)
    output_row[classes.index(doc_y[idx])]=1

    training.append([bow, output_row])

random.shuffle(training)

training=np.array(training,dtype=object)

train_X=np.array(list(training[:,0]))
train_y=np.array(list(training[:,1]))

input_shape=(len(train_X[0]),)
output_shape=len(train_y[0])

epochs=500

model=Sequential()
model.add(Dense(128, input_shape=input_shape,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_shape,activation='softmax'))
adam=tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy',
             optimizer="adam",
             metrics="accuracy")

print(model.summary())

model.fit(x=train_X, y=train_y, epochs=500, verbose=1)

def clean_text(text):
    tokens=nltk.word_tokenize(text)
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text,vocab):
    tokens=clean_text(text)
    bow=[0]*len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word==w:
                bow[idx]=1
    return np.array(bow)

def pred_class(text, vocab,labels):
    bow=bag_of_words(text, vocab)
    result=model.predict(np.array([bow]))[0]
    thresh=0.2
    y_pred=[[idx,res] for idx, res in enumerate(result) if res>thresh]

    y_pred.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    tag=intents_list[0]
    list_of_intents=intents_json["intents"]
    for i in list_of_intents:
        if i["tag"]==tag:
            result=random.choice(i["responses"])
            break
    return result

while True:
    message=input("")
    intents=pred_class(message, words, classes)
    result=get_response(intents,data)
    print(result)

