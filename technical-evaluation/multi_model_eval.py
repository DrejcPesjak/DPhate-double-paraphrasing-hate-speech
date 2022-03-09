'''Model averaging is the simplest form of ensemble learning.'''

''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
#https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer1 = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model1 = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
import torch
from torch.nn import functional as F

labs = ['Hate Speech','Normal','Offensive']

def detect_hate(text):
     inputs = tokenizer1(text, return_tensors="pt")
     labels = torch.tensor([1]).unsqueeze(0)
     outputs = model1(**inputs,labels=labels)
     sftmax = F.softmax(outputs.logits,dim=-1)[0]
     return sftmax


#def print_hate(torch_softmax):
#     hr = torch_softmax.cpu().detach().numpy()
#     return labs[np.argmax(hr)], list(hr)

def print_hate2(torch_softmax):
     hr = torch_softmax.cpu().detach().numpy()
     return (np.argmax(hr)+2)%3

preds1=[]
for k in dataset34.text:
	preds1.append(print_hate2(detect_hate(k)))
	if len(preds1)%100==0:
		print(len(preds1))

dataset34['pred1'] = preds1

#import json
#with open('dataset3/data3570.json') as jf:
#     data = json.load(jf)


#for k in data:
#     print(k)
#     print(print_hate(detect_hate(k)))
#     for e in data[k]:
#             print("    ",print_hate(detect_hate(e)))


''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
#https://huggingface.co/risingodegua/hate-speech-detector
#https://huggingface.co/uhhlt/bert-based-uncased-hatespeech-movies
#https://github.com/uhh-lt/hatespeech
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("risingodegua/hate-speech-detector")
model = TFAutoModelForSequenceClassification.from_pretrained("risingodegua/hate-speech-detector")
import torch
from torch.nn import functional as F
labels = ["Normal", "Offensive", "Hate Speech"]


#def make_prediction(text):
#    input_ids = tokenizer.encode(text)
#    input_ids = np.array(input_ids)
#    input_ids = np.expand_dims(input_ids, axis=0)
#    prediction_arr = model.predict(input_ids)[0][0]
#    prediction = labels[np.argmax(prediction_arr)]
#    return prediction


def make_prediction(text):
    input_ids = tokenizer.encode(text)
    input_ids = np.array(input_ids)
    input_ids = np.expand_dims(input_ids, axis=0)
    outputs = torch.from_numpy(model.predict(input_ids)[0])
    prediction = F.softmax(outputs,dim=-1)[0]
    return prediction

def print_hate(torch_softmax):
     hr = torch_softmax.cpu().detach().numpy()
     return np.argmax(hr)


preds2=[]
for k in dataset34.text:
	preds2.append(print_hate(make_prediction(k)))
	if len(preds2)%100==0:
		print(len(preds2))

dataset34['pred2'] = preds2

#def print_hate(torch_softmax):
#     hr = torch_softmax.cpu().detach().numpy()
#     return labels[np.argmax(hr)], list(hr)
#
#for k in data:
#     print(k)
#     print(print_hate(make_prediction(k)))
#     for e in data[k]:
#             print("    ",print_hate(make_prediction(e)))


''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
#https://github.com/Hironsan/HateSonar
#https://github.com/Hironsan/HateSonar/blob/master/notebooks/trial.ipynb
>>> pip install scikit-learn==0.22.1
 
from hatesonar import Sonar
sonar = Sonar()
#sonar.ping(text="At least I'm not a nigger")['top_class']
labs = {'hate_speech':2, 'offensive_language':1, 'neither':0}
def predict_hate(txt):
	return labs[sonar.ping(text=txt)['top_class']]

preds4=[]
for k in dataset34.text:
	preds4.append(predict_hate(k))
	if len(preds4)%100==0:
		print(len(preds4))

dataset34['pred4'] = preds4

>>> pip install scikit-learn==0.24.2
#  or
>>> pip install -U scikit-learn


''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
#https://github.com/NakulLakhotia/Hate-Speech-Detection-in-Social-Media-using-Python
import pandas as panda
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from textstat.textstat import *
from sklearn.svm import LinearSVC
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def preprocess(tweet):  
    #
    # removal of extra spaces
    regex_pat = re.compile(r'\s+')
    tweet_space = tweet.str.replace(regex_pat, ' ')
    #
    # removal of @name[mention]
    regex_pat = re.compile(r'@[\w\-]+')
    tweet_name = tweet_space.str.replace(regex_pat, '')
    #
    # removal of links[https://abc.com]
    giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweets = tweet_name.str.replace(giant_url_regex, '')
    #
    # removal of punctuations and numbers
    punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    # remove whitespace with a single space
    newtweet=punc_remove.str.replace(r'\s+', ' ')
    # remove leading and trailing whitespace
    newtweet=newtweet.str.replace(r'^\s+|\s+?$','')
    # replace normal numbers with numbr
    newtweet=newtweet.str.replace(r'\d+(\.\d+)?','numbr')
    # removal of capitalization
    tweet_lower = newtweet.str.lower()
    #
    # tokenizing
    tokenized_tweet = tweet_lower.apply(lambda x: x.split())
    #
    # removal of stopwords
    tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
    #
    # stemming of the tweets
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
    #
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
        tweets_p= tokenized_tweet
    #
    return tweets_p

def count_tags(tweet_c):  
    #
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', tweet_c)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def sentiment_analysis(tweet):   
    sentiment = sentiment_analyzer.polarity_scores(tweet)    
    twitter_objs = count_tags(tweet)
    features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],twitter_objs[0], twitter_objs[1],
                twitter_objs[2]]
    #features = pandas.DataFrame(features)
    return features

def sentiment_analysis_array(tweets):
    features=[]
    for t in tweets:
        features.append(sentiment_analysis(t))
    return np.array(features)

def additional_features(tweet): 
    #
    syllables = textstat.syllable_count(tweet)
    num_chars = sum(len(w) for w in tweet)
    num_chars_total = len(tweet)
    num_words = len(tweet.split())
    # avg_syl = total syllables/ total words
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(tweet.split()))
    #
    #  Flesch–Kincaid readability tests are readability tests 
    #  designed to indicate how difficult a passage in English is to understand. 
    # There are two tests, the Flesch Reading Ease, and the Flesch–Kincaid Grade 
    # A text with a comparatively high score on FRE test should have a lower score on the FKRA test.
    # Reference - https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    #
    ###Modified FK grade, where avg words per sentence is : just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    #
    add_features=[FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_words,
                num_unique_terms]
    return add_features

def get_additonal_feature_array(tweets):
    features=[]
    for t in tweets:
        features.append(additional_features(t))
    return np.array(features)


import pickle
support = pickle.load(open("svm_model.p", "rb"))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pk','rb'))

stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()
sentiment_analyzer = VS()

#def predict_hate(hate_sent):
#	twee = hate_sent
#	pro_twee = preprocess(panda.Series([hate_sent]))
#	#
#	fini_twee = sentiment_analysis(twee)
#	fF_twee = additional_features(pro_twee[0])
#	tfidf_twee = tfidf_vectorizer.transform(pro_twee).toarray().tolist()[0]
#	#
#	f = np.array(tfidf_twee + fini_twee + fF_twee)
#	#
#	pred = support.predict(f.reshape(1,-1))
#	names=['Hate Speech','Offensive','Normal']
#	return names[pred[0]]

def predict_hate(hate_sent):
	twee = hate_sent
	pro_twee = preprocess(panda.Series([hate_sent]))
	#
	fini_twee = sentiment_analysis(twee)
	fF_twee = additional_features(pro_twee[0])
	tfidf_twee = tfidf_vectorizer.transform(pro_twee).toarray().tolist()[0]
	#
	f = np.array(tfidf_twee + fini_twee + fF_twee)
	#
	pred = support.predict(f.reshape(1,-1))[0]
	if pred==2:
		return 0
	elif pred==0:
		return 2
	return pred

preds3=[]
for k in dataset34.text:
	preds3.append(predict_hate(k))
	if len(preds3)%100==0:
		print(len(preds3))

dataset34['pred3'] = preds3

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
maj_preds = []
data_preds = dataset34[['pred1','pred2','pred3','pred4']]
for i,preds in data_preds.iterrows():
	rota = preds.value_counts()
	if len(rota)==rota.iloc[0]:
		#if a tie, round up
		maj_preds.append(rota.index[1])
	else:
		maj_preds.append(rota.index[0])

dataset34['maj_preds']=maj_preds


'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
data_preds123 = dataset34[['pred1','pred2','pred3']]
maj_preds123 = []
for i,preds in data_preds123.iterrows():
	rota = preds.value_counts()
	maj_preds123.append(rota.index[0])

dataset34['maj_preds123']=maj_preds123

''' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X=dataset34[['pred1','pred2','pred3']]
y=dataset34.label.copy()
#y[y==2]=1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)
#clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_preds = clf.predict(X_test)

pickle.dump(clf, open('logreg_multi_model.p','wb'))

clf = pickle.load(open("logreg_multi_model.p", "rb"))

''' ++++++++++++++++++++++++++++++++final+++++++++++++++++++++++++++++++++++++ '''
def final_prediction(k):
	pred={}
	pred['pred1'] = [print_hate2(detect_hate(k))]
	pred['pred2'] = [print_hate(make_prediction(k))]
	pred['pred3'] = [predict_hate(k)]
	#pred = {'pred1':[0], 'pred2':[0], 'pred3':[1]}
	pr_df=panda.DataFrame(pred)
	return clf.predict(pr_df)[0]

#for r in data:
#	print(r)
#	print(labels[final_prediction(r)])
#	for e in data[r]:
#		print("     ",labels[final_prediction(e)])

import csv
i=0;j=0;table=[]
for r in data:
	table.append([r,-1,final_prediction(r)])
	for e in data[r]:
		table.append([e,i,final_prediction(e)])
	i+=1+len(data[r]);j+=1
	if j%100==0:
		df = panda.DataFrame(np.array(table),columns=['text','ref_ix','hate'])
		df = df.astype({'ref_ix':int, 'hate':int})
		df.to_csv("testing/datapredictions/hate" + str(j) + ".csv",index=False,quoting=csv.QUOTE_NONNUMERIC)
		print(i)

outs = df[df.ref_ix>=0]
better=[]
for i,h in outs.iterrows():
	df_hate = df.iloc[h.ref_ix].hate
	if df_hate==2:
		better.append(df_hate>h.hate)
	if len(better)%100==0:
		print(better[-10:])

print(sum(better)/len(better))
#>>> 0.7834065199



''' ++++++++++++++++++++++++++++++++similarity+++++++++++++++++++++++++++++++++++++ '''
#https://paperswithcode.com/task/semantic-textual-similarity
#https://github.com/Tiiiger/bert_score
datacsv = panda.read_csv('dataset3/data3570.csv')
from bert_score import score
#P,R,F1 = score(datacsv.iloc[40:60].source_text.tolist(), datacsv.iloc[40:60].target_text.tolist(), rescale_with_baseline=True, lang="en")#,verbose=True)
P,R,F1 = score(d['sentence1'].tolist(),d['sentence2'].tolist(),rescale_with_baseline=True, lang="en",verbose=True)
y_preds = F1.cpu().detach().numpy()

#accuracy on glue mrpc at threshold 0.56:   0.738
#0.55 0.26553980370774266
#0.56 0.26226826608506
#0.57 0.2658124318429662
#0.58 0.26990185387131954
#0.59 0.2778080697928026
#0.60 0.2840785169029444
#0.61 0.28544165757906215
#0.62 0.2936205016357688
#0.63 0.30016357688113415
y_preds2=y_preds.copy()
thresh=0.55
while thresh<0.66:
     y_preds2[np.where(y_preds>thresh)]=1
     y_preds2[np.where(y_preds<=thresh)]=0
     p = 1 - np.sum(np.abs(y_test-y_preds2))/len(y_test)
     print(thresh,p)
     thresh+=0.01


''' ++++++++++++++++++++++++++++++++similarity2+++++++++++++++++++++++++++++++++++++ '''
#https://github.com/princeton-nlp/SimCSE
#pip install simcse
from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
#sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
#sentences_b = ['He plays guitar.', 'A woman is making a photo.']
#similarities = model.similarity(sentences_a, sentences_b)

sim_scores=[]
for i,r in d[['sentence1','sentence2']].iterrows():
     sentences = r.tolist()
     s=model.similarity([sentences[0]],[sentences[1]])
     sim_scores.append(s[0][0])


#threshold of 0.75: accuracy => 0.762

''' ++++++++++++++++++++++++++++++++similarity3+++++++++++++++++++++++++++++++++++++ '''
#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
sim = cosine_similarity([embeddings[0]],embeddings[1:])

from datasets import load_dataset
#dataset = load_dataset("paws", "labeled_final")
dataset = load_dataset("glue", "mrpc")
import pandas as pd
d = pd.DataFrame(dataset['train'])
sim_scores=[]
for i,r in d[['sentence1','sentence2']].iterrows():
	sentences = r.tolist()
	embeddings = model.encode(sentences)
	sim = cosine_similarity([embeddings[0]],embeddings[1:])
	sim_scores.append(sim[0][0])


import matplotlib.pyplot as plt
plt.scatter(d.label.tolist(),sim_scores)
import numpy as np
y_test = d.label.to_numpy()
y_preds = np.array(sim_scores)
plt.hist(y_preds[np.where(y_test==0)], bins=100, alpha=0.5, label='different')
plt.hist(y_preds[np.where(y_test==1)], bins=100, alpha=0.5, label='similar')
plt.legend(loc='upper left')
plt.show()

