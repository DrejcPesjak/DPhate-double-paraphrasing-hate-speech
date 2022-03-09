'''
This file contains useful tools/libraries for the task of hate speech removal.
These tools were tested, and produce good results.

Each tool has a link to the source, but the main three are:
 - hate speech detector -> https://github.com/unitaryai/detoxify
 - hate speech dataset -> https://huggingface.co/datasets/hatexplain
 - paraphraser -> https://huggingface.co/tuner007/pegasus_paraphrase
'''

''':pegasus:'''
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizerP = PegasusTokenizer.from_pretrained(model_name)
modelP = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
#print(modelP.generate.__doc__)

# `num_beams` should be divisible by `num_beam_groups`
# `num_return_sequences` has to be smaller or equal to `num_beams`
def get_response(input_text,num_return_sequences=20,num_beams=100, groups=25, diversityP=1.0):
  batch = tokenizerP([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = modelP.generate(**batch,
  								max_length=60,
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

#generate new sentences and print
print_list( get_response(hate_phrases[3]) ) 
     
''':hatexplain dataset:'''
from datasets import load_dataset
dataset = load_dataset("hatexplain")

#get the majority number from list (the one with most occurances)
def majority(lst):
     n=list(set(lst))
     n.reverse()
     return max(n, key=lst.count)

#for each sentence calculate the majority vote (0-hate,1-normal,2-offensive)
votes=[]
for i in range(len(dataset['train'])):
     lst = dataset['train'][i]['annotators']['label']
     votes.append(majority(lst))

#make a list of all phrases that are "normal"
votesNP = np.array(votes)
phrases=[]
selected = np.where(votesNP==1)[0]
for txtList in dataset['train'][selected]['post_tokens']:
     phrases.append(" ".join(txtList))

''':toxic-bert:'''
from detoxify import Detoxify
import numpy as np
#check how many were predicted as hatefull
results = Detoxify('original').predict(phrases)
len( results['toxicity'])
sum([int(i>0.5) for i in results['toxicity']])
ix = np.where(np.array(results['toxicity'])>0.5)[0]
resultsALL = results

#for the phrase generate new ones, and keep only non hateful ones
for i in ix:
	print('-'*30)
	print('original phrase:', hate_phrases[i])
	print('toxicity:',resultsALL['toxicity'][i])
	pra = get_response(hate_phrases[i])
	results = Detoxify('original').predict(pra)
	tox = results['toxicity']
	good=[]
	tmp = [tox[i]<0.6 and good.append(pra[i]) for i in range(len(tox))]
	if len(good)==0:
		print("NOT")
		good = pra
	print_list(good)


''':sentence meaning similarity:'''
#https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')
#sentences = ["I really like this book.", "I really fucking like this book."]
#sentences = [hate_phrases[i], pra[14]]
from collections import deque
sentences = deque(pra)
sentences.appendleft(hate_phrases[i])
sentences = list(sentences)
sentence_embeddings = model.encode(sentences)
sim = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])



''':linguistic acceptability - grammar checking:'''
#https://huggingface.co/kamivao/autonlp-cola_gram-208681
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional as F
modelG = AutoModelForSequenceClassification.from_pretrained("kamivao/autonlp-cola_gram-208681")
tokenizerG = AutoTokenizer.from_pretrained("kamivao/autonlp-cola_gram-208681")
def grammar(text):
	inputs = tokenizerG(text, return_tensors="pt")
	outputs = modelG(**inputs)
	score = F.softmax(outputs.logits, dim=-1)
	score = score.tolist()[0]
	return score


''':spell check and correction:'''
from textblob import TextBlob
from textblob import Word
#phrase = "People that are idots, cannot come."
TextBlob(phrase).correct()
Word('idiotes').spellcheck()
Word('bich').spellcheck()



''':profanity censor:'''
#multiple profanity words, delete part of sentence for each
# might delete everything -> replace it only with a nice word
from better_profanity import profanity
profanity.load_censor_words()
p2=profanity.censor(phrase)
start = p2.find('****')
end = phrase.find(' ',start)
word = ''
if end == -1:
	word = phrase[start:]
else:
	word = phrase[start:end]
#phrase.count('****')
#phrase.replace('****','')



''':dependency tree - standford parser:'''
#https://stanfordnlp.github.io/CoreNLP/
import os
from nltk.parse.corenlp import CoreNLPServer
STANFORD = os.path.join("models", "stanford-corenlp-4.2.2")
server = CoreNLPServer(
   os.path.join(STANFORD, "stanford-corenlp-4.2.2.jar"),
   os.path.join(STANFORD, "stanford-corenlp-4.2.2-models.jar"),    
)
server.start()

from nltk.parse.corenlp import CoreNLPDependencyParser
from itertools import permutations
parser = CoreNLPDependencyParser()
#parse = next(parser.raw_parse(phrase))
#parses = parser.parse(phrase.split())
#trii = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
sent = phrase.split()
parses = list(parser.parse(sent))
#parses[0].tree()
#parses[0].tree().draw()
#parses[0].tree().pretty_print()

parse = parses[0].tree()
#word = "kill"
sub = parse.subtrees(filter=lambda t: t.label()==word)
for p in sub:
    words = str(p).replace('(','').replace(')','')
    word_list = words.split()

if(len(word_list)/len(sent) < 0.8 and len(word_list)<8): 
	for p in permutations(word_list):
		sub = ' '.join(p)
		phrase = phrase.replace(sub, '')
		print('OK')

server.stop()



''':POS TAGGING + punctuation removal:'''
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import RegexpTokenizer
st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
tokenizer = RegexpTokenizer(r'\w+')
phrase = 'Fuck that shit this bitch ugly and she has a penis,oh hell nawww!!!'
splt = tokenizer.tokenize(phrase)
tagged = st.tag(splt)
for e in tagged:
     print(e[1])



hate_phrases = [
"You forgot to mention just how smart the jews are, next they will make a deadly virus to kill ALL people BUT the jews!!",
"Queers are an abomination and need to be helped to go straight to Hell!",
"The Palestinians are beasts walking on two legs.",
"We have to kill all the Palestinians unless they are resigned to live here as slaves.",
"Now that Trump is president, I'm going to shoot you and all the blacks I can find",
"And isn’t it funny. I’ve got black accountants at Trump Castle and Trump Plaza. Black guys counting my money! I hate it!",
"The only kind of people I want counting my money are short guys that wear yarmulkes every day.",
"white males going up to women saying that it was now legal to grab them by the pussy",
"I will punish all females for the crime of depriving me of sex. They have starved me of sex for my entire youth, and gave that pleasure to other men. In doing so, they took many years of my life away. I cannot kill every single female on earth, but I can deliver a devastating blow that will shake all of them to the core of their wicked hearts. (He wonders why women don’t want anything to do with him)",
"Build a wall was chanted in our cafeteria at lunch. If you aren't born here, pack your bags was shouted in my own classroom. Get out spic was said in our halls.",
"According to another submission, a number of students were suspended from Millersberg Area High School in Pennsylvania for harassing a Latina student chanting Trump and cheering when she ran out crying.",
"Today a young Latino man in is 20's, and a co-worker of mine, was walking into work as a truck slowed down and two white men threw a bag of garbage onto him and yelled, you are going back to where you came from",
"After the election: This is their punishment for 8 years of black people.",
"A black truck with three white men pulled up to the red light. One of them yelled, Fuck your black life! The other two began to laugh. One began to chant Trump! as they drove away.",
"Wipe out the Jews.",
"The Jews are criminals and our enemies. Said by three year old girl.",
"You shit Jew, I'm going to kill you. As he plunged a knife into the man.",
"We’re going to take over all the political institutions of California. California is going to be a Hispanic state and anyone who doesn’t like it should leave. If they [Anglos] don’t like Mexicans, they ought to go back to Europe.",
"There is plenty to blame whiteness for. There is no crime that whiteness has not committed against people of color. There is no crime that we have not committed even against ourselves. We must blame whiteness for the continuing patterns today that deny the rights of those outside of whiteness and which damage and pervert the humanity of those of us within it.",
"The necessary re-education of Blacks and a possible solution of the racial crisis can begin, only when Blacks fully realize this central fact to their lives: the white man is their Bitter Enemy.",
"laziness is a trait in blacks. It really is, I believe that. It’s not anything they can control.",
"The Mexican government is much smarter, much sharper, much more cunning. They send the bad ones over because they don't want to pay for them. They don't want to take care of them",
"Women are like grass, they need to be beaten/cut regularly.",
"You just need to be raped.",
"You stupid ugly fucking slut I’ll go to your flat and cut your fucking head off you inbred whore.",
"Fucking faggots!",
"You're Asian, right? When they see your eyes you are going to be deported. Said to 13 year old girl.",
"Anonymous note: Your Muslim headscarf isn't allowed anymore. Why don’t you tie it around your neck & hang yourself with it, signed America!",
"The true Negro does not want integration. ... He realizes his potential is far better among his own race.",
"Gay folks would just as soon kill you as look at you",
"The National Organization of Women is actually the National Organization of Witches.",
"hurricanes and disasters and the 9/11 attacks are divine retribution for homos.",
"Check your white privileges you cis white male!"]
