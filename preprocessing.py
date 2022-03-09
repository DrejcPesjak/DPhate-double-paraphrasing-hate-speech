import warnings
warnings.filterwarnings("ignore")


import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizerP = PegasusTokenizer.from_pretrained(model_name)
modelP = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences=20,num_beams=100, groups=25, diversityP=1.0):
  batch = tokenizerP([input_text],truncation=True,padding='longest', return_tensors="pt").to(torch_device)
  translated = modelP.generate(**batch,
  								num_beams=num_beams, 
  								num_return_sequences=num_return_sequences,
  								num_beam_groups=groups,
  								diversity_penalty=diversityP) 
  tgt_text = tokenizerP.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

def print_list(str_list):
	if len(str_list)==0:
		print('None')
	for p in str_list:
		print('> ', end='')
		print(p)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')
from collections import deque

from detoxify import Detoxify
modelD = Detoxify('original')
#modelD = Detoxify('original', device='cuda')

import numpy as np
import json

values= [[ 20., 100.,  25.,   1.],
		[ 30., 100.,  50.,   3.],
		[ 40., 100.,  50.,   2.],
		[ 40., 300., 150.,   2.]]

def similarity(base, phrases):
	sentences = deque(phrases)
	sentences.appendleft(base)
	sentences = list(sentences)
	sentence_embeddings = model.encode(sentences)
	sim = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
	return sim

from datasets import load_dataset
dataset = load_dataset("hatexplain")

def majority(lst):
     n=list(set(lst))
     n.reverse()
     return max(n, key=lst.count)

votes=[]
for i in range(len(dataset['train'])):
     lst = dataset['train'][i]['annotators']['label']
     votes.append(majority(lst))

votesNP = np.array(votes)
phrases=[]
selected = np.where(votesNP==0)[0]
for txtList in dataset['train'][selected]['post_tokens']:
     phrases.append(" ".join(txtList))

import re
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

names = ['Wade', 'Dave', 'Seth', 'Ivan', 'Riley', 'Gilbert', 'Jorge', 'Dan', 'Brian', 'Roberto', 'Ramon', 'Miles', 'Liam', 'Nathaniel', 'Ethan', 'Lewis', 'Milton', 'Claude', 'Joshua', 'Glen', 'Harvey', 'Blake', 'Antonio', 'Connor', 'Julian', 'Aidan', 'Harold', 'Conner', 'Peter', 'Hunter', 'Eli', 'Alberto', 'Carlos', 'Shane', 'Aaron', 'Marlin', 'Paul', 'Ricardo', 'Hector', 'Alexis', 'Adrian', 'Kingston', 'Douglas', 'Gerald', 'Joey', 'Johnny', 'Charlie', 'Scott', 'Martin', 'Tristin',
'Daisy', 'Deborah', 'Isabel', 'Stella', 'Debra', 'Beverly', 'Vera', 'Angela', 'Lucy', 'Lauren', 'Janet', 'Loretta', 'Tracey', 'Beatrice', 'Sabrina', 'Melody', 'Chrysta', 'Christina', 'Vicki', 'Molly', 'Alison', 'Miranda', 'Stephanie', 'Leona', 'Katrina', 'Mila', 'Teresa', 'Gabriela', 'Ashley', 'Nicole', 'Valentina', 'Rose', 'Juliana', 'Alice', 'Kathie', 'Gloria', 'Luna', 'Phoebe', 'Angelique', 'Graciela', 'Gemma', 'Katelynn', 'Danna', 'Luisa', 'Julie', 'Olive', 'Carolina', 'Harmony', 'Hanna', 'Anabelle']

def preprocess(phrase):
	t = ['<user>','<percent>','<number>']
	cont = [k in phrase for k in t]
	if(cont[0]==True):
		#insert a name from list
		i = np.random.randint(len(names))
		phrase = phrase.replace(t[0], names[i], 1)
		return preprocess(phrase)
	if(cont[1]==True):
		#insert a percentage
		p = '%.2f' % (np.random.rand()*100)
		phrase = phrase.replace(t[1], p+'%' , 1)
		return preprocess(phrase)
	if(cont[2]==True):
		#insert a random number
		n = str(np.random.randint(2025))
		phrase = phrase.replace(t[2], n ,1)
		return preprocess(phrase)
	#remove emojis
	phrase = remove_emojis(phrase)
	#remove double spaces
	return re.sub(' +', ' ', phrase)

for i in range(len(phrases)):
	phrases[i] = preprocess(phrases[i])


phrasesNP = np.array(phrases)
#np.savetxt('phrases.txt', phrasesNP, delimiter="\n", fmt="%s")
#phrasesNP = np.loadtxt('phrases.txt', dtype='str' , delimiter="\n")
'''
dataset2 = load_dataset("ethos", "binary")
ix = np.where(np.array(dataset2['train']['label'])==1)
hate = np.array(dataset2['train']['text'])[ix[0]]
hate1 = list(hate)
results = modelD.predict(hate1)
tox = np.array(results['toxicity'])
z = np.where(tox<0.5)[0]
tox[z]=0.5
div = ((tox-0.5)//0.125).astype(int)
'''
margin = 1000
tox = np.array([])
for s in range(0,len(phrases),margin):
	e = s+margin if s+margin < len(phrases) else len(phrases)
	results = modelD.predict(phrases[s:e])
	tox1 = np.array(results['toxicity'])
	tox = np.concatenate((tox,tox1))

ix = np.where(tox>0.5)[0]
div = ((tox[ix]-0.5)//0.125).astype(int)
hate = phrasesNP[ix]

#with open('data3570.json') as jf:
#	data = json.load(jf)
