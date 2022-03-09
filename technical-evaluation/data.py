#https://hatespeechdata.com/#English-header

''' ------------------------------------------------------------------------------------ '''
#https://github.com/paul-rottger/hatecheck-data/blob/main/all_annotations.csv
#https://arxiv.org/pdf/2012.15606.pdf
import pandas as pd
dataset = pd.read_csv("all_annotations.csv")


''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
#https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech
from datasets import load_dataset
dataset = load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
df = dataset['train'].to_pandas()
df.describe()



''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
#https://huggingface.co/datasets/hate_speech_offensive
#https://github.com/t-davidson/hate-speech-and-offensive-language
#from datasets import load_dataset
#dataset = load_dataset("hate_speech_offensive")
dataset = panda.read_csv("HateSpeechData.csv")

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
    return tweets_p


''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ '''
#https://huggingface.co/datasets/hatexplain
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
