import sys
import regex as re

import numpy as np
import pandas as pd

from nltk import word_tokenize

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def read_data(file, verobse=True):

    """
    Funkcja wczytujaca dane.

    Zalozenia o postaci danych:
    - jedna linia to jedna obserwacja z dokladnoscia bledow opisanych wyzej
    - dane maja postac: (id,etykieta, tresc)
    - zakladam ze pola sa oddzielone tabulatorem
    - zakladam ze tabulator moze znajdowac sie rowniez w tresci tekstu
    - zakladam ze tresc tweeta nie powinna byc objeta w cudzyslow
    """


    label_encoding = dict({"negative":0,"neutral":1,"positive":2})
    connection = open(file,"r")
    data = connection.read().split("\n")
    connection.close()
    

    if len(data[-1])==0: #poprawiamy wczytywanie koncowki pliku usuwajac pusty element z konca listy
        del data[-1]
        
    data = list(map(lambda x: x.split("\t"),data))

    
    # "przenosze" cudzyslowy, ktore przeskoczyly do nastepnej lini na wlasciwe miejsce
    lost_quotation_marks = [i for i,x in enumerate(data) if x[0]=="\"" ]
    for i in lost_quotation_marks:
        data[i-1].append("\"")
    data = [x for x in data if x[0] != "\""]
    
    #usuniecie nadmiarowych bialych znakow
    data = map(lambda x: (x[1], " ".join((" ".join(x[2:])).split())),data)
    
    # zamienienie podwojnych cudzyslowow na pojedyncze
    data = list(map(lambda x: (x[0], x[1].replace('""','"')),list(data)))
    
    X = [x[1][1:-1] if (x[1][0]=='"' and x[1][-1]=='"') else x[1] for x in data] #usuwamy cudzysowy z poczatku i konca o ile wystepuja (jednoczesnie)
    y = [label_encoding[x[0]] for x in data]
    

    if verobse>0:
        print ("Data report:\n","Number of obserwations: ", len(X))
    
    return X, y




def load_embeddings(embedding_path,emb_dim):

    words2ids = {}
    vectors = [np.zeros(emb_dim)] # pierwszy wektor jest zerowy na padding
    i = 0
    with open(embedding_path,"r") as f:
        for line in f:
            toks = line.split(" ")
            word = toks[0]
            if True:#word in words:
                v = list(map(float, toks[1:]))
                vectors.append(v)
                words2ids[word] = i
                i = i + 1

    vectors = np.array(vectors)
    
    # w embeddingach, z ktorych korzystam znaczniki specjalne maj postac "<x>",
    # natomiast w naszych danych "_x_". Zatem uzgadniamy:
    keys = list(words2ids.keys())
    for key in keys:
        if key[0]=="<" and key[-1]==">":
            new_key = "_" + key[1:-1] + "_"
            words2ids[new_key] = words2ids.pop(key)

    return words2ids, vectors



def transform(X,y,words2ids,maxlen):
    
    X_new = [[words2ids.get(x,words2ids["_unknown_"]) for x in word_tokenize(X[i])] for i in range(len(X))]
    
    X_new = pad_sequences(X_new,maxlen=maxlen)
    
    y_new = to_categorical(y)
    
    return X_new, y_new


"""
Ponizsze funkcje nie sa autorskie (Zmiany w stosunku do oryginaly sa minimalne)
Zrodlo: https://gist.github.com/ppope/0ff9fa359fb850ecf74d061f3072633a
"""


#FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["_hashtag_"] + re.split(r"(?=[A-Z])", hashtag_body, flags=re.MULTILINE | re.DOTALL))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " _allcaps_"


def preprocess_one_tweet(text):

    """
    Funkcja przetwarza podany text. Obrobka jest dedykowana tweetom - obrobka ta jest zgodna z obrobka, na ktorej uczone byly embeddingi dla tweetow.
    """


    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "_url_")
    text = re_sub(r"@\w+", "_user_")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "_smile_")
    text = re_sub(r"{}{}p+".format(eyes, nose), "_lolface_")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "_sadface_")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "_neutralface_")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","_heart_")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "_number_")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 _repeat_")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 _elong_")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

def preprocess(texts):
    return [preprocess_one_tweet(x) for x in texts]

