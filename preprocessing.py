''' Preprocessing for the DPhate algorithm, 
	it processes the Hatexplain dataset and labeles the examples with the Detoxify hate speech algorithm.
	To run it, use:
		python3 preprocessing.py
'''
import numpy as np

#load the Hatexplain dataset
from datasets import load_dataset
dataset = load_dataset("hatexplain")

#a prittier print of a list 
def print_list(str_list):
	if len(str_list)==0:
		print('None')
	for p in str_list:
		print('> ', end='')
		print(p)

#get the majority vote
def majority(lst):
     n=list(set(lst))
     n.reverse()
     return max(n, key=lst.count)

#get majority vote for each example in Hatexplain dataset
votes=[]
for i in range(len(dataset['train'])):
     lst = dataset['train'][i]['annotators']['label']
     votes.append(majority(lst))

#get only the examples with the majority vote labeled as hateful==0
votesNP = np.array(votes)
phrases=[]
selected = np.where(votesNP==0)[0]
for txtList in dataset['train'][selected]['post_tokens']:
     phrases.append(" ".join(txtList))


#removes emojis, symbols, flags, chinese characters...
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

#a list of 20 most popular male names and 20 most popular female names
names = ['Wade', 'Dave', 'Seth', 'Ivan', 'Riley', 'Gilbert', 'Jorge', 'Dan', 'Brian', 'Roberto', 'Ramon', 'Miles', 'Liam', 'Nathaniel', 'Ethan', 'Lewis', 'Milton', 'Claude', 'Joshua', 'Glen', 'Harvey', 'Blake', 'Antonio', 'Connor', 'Julian', 'Aidan', 'Harold', 'Conner', 'Peter', 'Hunter', 'Eli', 'Alberto', 'Carlos', 'Shane', 'Aaron', 'Marlin', 'Paul', 'Ricardo', 'Hector', 'Alexis', 'Adrian', 'Kingston', 'Douglas', 'Gerald', 'Joey', 'Johnny', 'Charlie', 'Scott', 'Martin', 'Tristin',
'Daisy', 'Deborah', 'Isabel', 'Stella', 'Debra', 'Beverly', 'Vera', 'Angela', 'Lucy', 'Lauren', 'Janet', 'Loretta', 'Tracey', 'Beatrice', 'Sabrina', 'Melody', 'Chrysta', 'Christina', 'Vicki', 'Molly', 'Alison', 'Miranda', 'Stephanie', 'Leona', 'Katrina', 'Mila', 'Teresa', 'Gabriela', 'Ashley', 'Nicole', 'Valentina', 'Rose', 'Juliana', 'Alice', 'Kathie', 'Gloria', 'Luna', 'Phoebe', 'Angelique', 'Graciela', 'Gemma', 'Katelynn', 'Danna', 'Luisa', 'Julie', 'Olive', 'Carolina', 'Harmony', 'Hanna', 'Anabelle']

#replaces <user>,<percent>,<number> with actual values, and removes emojis
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
np.savetxt('data-generated/phrases.txt', phrasesNP, delimiter="\n", fmt="%s")
#phrasesNP = np.loadtxt('phrases.txt', dtype='str' , delimiter="\n")

'''get only examples that the hate speech detector labels as hateful (toxicity>0.5), and determine the toxCategory (variable div) for each'''
#load the hate speech detector
from detoxify import Detoxify
modelD = Detoxify('original')

#only label 1000 examples at once, because you may run out of RAM
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

np.savetxt('data-generated/div.txt', div, delimiter="\n", fmt="%s")
np.savetxt('data-generated/hate.txt', hate, delimiter="\n", fmt="%s")


''' do the same for ethos dataset '''
#dataset2 = load_dataset("ethos", "binary")
#ix = np.where(np.array(dataset2['train']['label'])==1)
#hate = np.array(dataset2['train']['text'])[ix[0]]
#hate1 = list(hate)
#results = modelD.predict(hate1)
#tox = np.array(results['toxicity'])
#z = np.where(tox<0.5)[0]
#tox[z]=0.5
#div = ((tox-0.5)//0.125).astype(int)
#
#np.savetxt('data-generated/div.txt', div, delimiter="\n", fmt="%s")
#np.savetxt('data-generated/hate.txt', hate, delimiter="\n", fmt="%s")


