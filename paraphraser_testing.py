''' 
Pegasus paraphraser parameter testing 

Different parameter values are tested on the earlier version of the DPhate algorithm, 
the ones used in the final version were picked according to personal judgement.  

If you want to test this yourself, just run it by hand in the python shell. :)
'''

#import Pegasus paraphraser
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


#import Detoxify hate speech detector
from detoxify import Detoxify


#import BERT embeddings + cosine similarity for checking similarity between the original an other sentences
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')
from collections import deque

def similarity(base, phrases):
	sentences = deque(phrases)
	sentences.appendleft(base)
	sentences = list(sentences)
	sentence_embeddings = model.encode(sentences)
	sim = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
	return sim


#import the autonlp-cola_gram grammar checker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional as F
modelG = AutoModelForSequenceClassification.from_pretrained("kamivao/autonlp-cola_gram-208681")
tokenizerG = AutoTokenizer.from_pretrained("kamivao/autonlp-cola_gram-208681")

def grammar(inputT):
	inputs = tokenizerG(inputT, return_tensors="pt")
	outputs = modelG(**inputs)
	score = F.softmax(outputs.logits, dim=-1)
	score = score.tolist()[0]
	return score


#generate acceptable value quadruplets for the paraphraser
import numpy as np
values=[]
nseq = np.arange(10,41,10)
beam = np.array([50,60,80,90,100,150,160,180,200,240,250,300])
div = np.arange(1.0,4.1,0.5)
div = np.delete(div, -2)
group = np.array([2,3,4,5])

for i in beam:
	for j in group:
		if (i%j==0 and (i//j)%10==0):
			for k in nseq:
				if(k < (i//j)):
					for l in div:
						values.append([k, i, i//j, l])


#the old version of the DPhate algorithm
phrase = "this nigger calling me juicy lips over the phone wtf"
goodN = []
scr = []
l = 0
for i in range(len(values)):
	print(100*'*')
	l+=1
	print(l, values[i])
	x,y,z,t = values[i]
	pra = get_response(phrase, num_return_sequences=int(x), num_beams=int(y), groups=int(z), diversityP=t)
	results = Detoxify('original').predict(pra)
	tox = np.array(results['toxicity'])
	#print(tox)
	ix = np.where(tox < 0.6 )[0]
	praGood = list(np.array(pra)[ix])
	#print(list(tox[ix]))
	#print(praGood)
	if(len(praGood)>0):
		sim = similarity(phrase, praGood)
		#print(sim)
		ixSim = np.where(sim[0]>0.6)[0]
		g=[]
		for el in ixSim:
			g.append( grammar(praGood[el])[1] )
		#print(g) #good grammar if >0.5
		g=np.array(g)
		ixG = np.where(g>0.5)[0]
		if(len(ixG)>0):
			toxN = tox[ix][ixSim][ixG]
			simN = sim[0][ixSim][ixG]
			grN = g[ixG]
			score = ((1.0-toxN) + simN + grN ) 
			score = sum(score)/ len(grN)
			goodN.append(len(grN))
			scr.append(score)
			print_list(np.array(pra)[ix][ixSim][ixG])
		else:
			goodN.append(0)
			scr.append(0.0)
	else:
		goodN.append(0)
		scr.append(0.0)



#graph analysis of parameter pairs from:
	#goodN : how many of the new sentences were deemed good (tox<0.6, sim>0.6, gram>0.5)
	#scr: average score (1-tox +sim +gram)/N of the acceptable new sentences (the higher the better, max==3.0)
 # -> results can be seen in folder pictures: values.png, values2.png, numberG.png, numberG2.png, scoreG.png



#good values:
values= [[ 20., 100.,  50.,   1.],
		[ 30.,  80.,  40.,   3.],
		[ 30., 100.,  50.,   3.],
		[ 30., 200., 100.,   2.],
		[ 40., 100.,  50.,   2.],
		[ 40., 200., 100.,   2.],
		[ 40., 300., 150.,   2.]]

#values used in the final version:
values= [[ 20., 100.,  25.,   1.],
		[ 30., 100.,  50.,   3.],
		[ 40., 100.,  50.,   2.],
		[ 40., 300., 150.,   2.]]


