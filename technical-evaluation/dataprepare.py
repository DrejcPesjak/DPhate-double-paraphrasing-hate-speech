import pandas as pd
dataset1 = pd.read_csv("testing/all_annotations.csv")
d1 = dataset1[['test_case','label_annot_maj']]
d1 = d1.rename(columns={'test_case':'text', 'label_annot_maj':'label'})
d1.loc[d1['label']=='hateful','label']=2
d1.loc[d1['label']=='non-hateful','label']=0
#d1['label'].unique()



dataset = pd.read_parquet("testing/measuring-hate-speech.parquet")
d2 = dataset[['text','hate_speech_score']]
#'sentiment','respect','insult','humiliate','status','dehumanize','violence','genocide','attack_defend','hatespeech','hate_speech_score','text','infitms','outfitms','annotator_severity','std_err','annotator_infitms','annotator_outfitms','hypothesis'
d2.loc[d2['hate_speech_score']<=0.2,'hate_speech_score'] = 0
d2.loc[(d2['hate_speech_score']>0.2) & (d2['hate_speech_score']<2.0),'hate_speech_score'] = 1
d2.loc[d2['hate_speech_score']>=2.0,'hate_speech_score'] = 2
d2 = d2.rename(columns={'text':'text', 'hate_speech_score':'label'})
d2['label'] = d2.label.astype(int)
d2['text'] = preprocess(d2.text) #def at line 23
d2 = d2.iloc[:40000]


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
    #remove retweet at the beginning
    tweet_re = tweet_lower.str.replace("\A(rt )+",'')
    #
    return tweet_re


dataset = pd.read_csv("testing/HateSpeechData.csv")
d3 = dataset[['tweet','class']]
d3 = d3.rename(columns={'tweet':'text', 'class':'label'})
d3['text'] = preprocess(d3.text)



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

#votesNP = np.array(votes)
phrases=[]
# (0-hate,1-normal,2-offensive)
for txtList in dataset['train']['post_tokens']:
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

d4 = pd.DataFrame({'text':phrases, 'label':votes})
rep = {1:0,0:2,2:1}
d4 = d4.replace({'label':rep})


#d1=3901
#d2=135556
#d3=24783
#d4=15383
#sum=179623
dataset = pd.concat([d1,d2,d3,d4])
dataset['label']=dataset['label'].astype(int)
dataset.reset_index(drop=True, inplace=True)
dataset34 = dataset[len(d1)+len(d2):].copy()
import csv
dataset.to_csv('testing/datafull.csv',index=False,quoting=csv.QUOTE_NONNUMERIC)
dataset34.to_csv('testing/data34.csv',index=False,quoting=csv.QUOTE_NONNUMERIC)
#dataset['predictions1']=list
g = dataset34.label.iloc[:24781]
g2 = g.replace(to_replace=[2,0], value=[0,2])
dataset34.label.iloc[:24781]=g2
