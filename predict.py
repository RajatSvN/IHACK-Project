
# REST API developed by Rajat, Rishabh, Divyam, Kshitij 
# necessary imports

from flask import Flask
from flask import request
from flask import json
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import pickle





app = Flask(__name__)

# loading required pickles , serialized object for ML Models

model = pickle.load(open("m.p","rb"))
token = pickle.load(open("t.p","rb"))

@app.route('/',methods=["POST"])

def predict():

    # preprocessing code , removing stopwords

    a = []
    a.append(str(request.data))
    a[0]=a[0][3:len(a[0])-2]
    

    b=[]
    bwords = stopwords.words('english')[:45]
    bwords.append('<user>')
    bwords.append('<url>')

    s=''
    for i in a[0].split():
      if i not in bwords:
        s=s+i+' '
    b.append(s.strip())


    seq1=token.texts_to_sequences(b)

    

    seq1 = pad_sequences(seq1,maxlen = 67,padding = 'post', truncating = 'post',value = 0)
    x1 = np.asmatrix(seq1)

    #return prediction to HTML page

    with graph.as_default():
        preds = (model.predict(x1))
        return str(preds[0][0])

    

if __name__ == "__main__":
    graph = tf.get_default_graph()
    app.run(host='0.0.0.0',debug=True)
