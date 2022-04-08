''' *************************DPhate algorithm**************************** '''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
import os
import torch
import re
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from detoxify import Detoxify
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from better_profanity import profanity

class DPhate:
	def __init__(self):
		''' Initialize DPhate, by loading the necessary models. '''
		#load the PEGASUS paraphraser
		model_name = 'tuner007/pegasus_paraphrase'
		self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.tokenizerP = PegasusTokenizer.from_pretrained(model_name)
		self.modelP = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)
		
		#4 sets of paraphraser parameters
		self.values= [[ 20., 100.,  25.,   1.],
						[ 30., 100.,  50., 3.],
						[ 40., 100.,  50., 2.],
						[ 40., 300., 150., 2.]]
		
		#load the Detoxify hate speech detector
		self.modelD = Detoxify('original', device=self.torch_device)
		
		#load BERT for assessing similarity on embeddings
		self.model = SentenceTransformer('bert-base-nli-mean-tokens')
		
		self.tokenizer = RegexpTokenizer(r'\w+')
		profanity.load_censor_words()
	
	
	def paraphrase(self,input_text,num_return_sequences=20,num_beams=100, groups=25, diversityP=1.0):
	  ''' 
	   Performs a paraphrasal 
	    - `num_beams` should be divisible by `num_beam_groups`
	    - `num_return_sequences` has to be smaller or equal to `num_beams`
	  '''
	  batch = self.tokenizerP([input_text],truncation=True,padding='longest', return_tensors="pt").to(self.torch_device)
	  translated = self.modelP.generate(**batch,
	  								num_beams=num_beams, 
	  								num_return_sequences=num_return_sequences,
	  								num_beam_groups=groups,
	  								diversity_penalty=diversityP) 
	  tgt_text = self.tokenizerP.batch_decode(translated, skip_special_tokens=True)
	  return tgt_text
	
	def paraphrase_toxic(self,phrase,x,y,z,t):
		''' Performs a paraphrasal and toxicity assesment. '''
		pra = self.paraphrase(phrase, num_return_sequences=int(x), num_beams=int(y), groups=int(z), diversityP=t)
		results = self.modelD.predict(pra)
		tox = np.array(results['toxicity'])
		ix = np.where(tox < 0.5 )[0]
		return pra,tox,ix
	
	def similarity(self,base, phrases):
		''' Calculates cosine similarity on BERT embeddings. '''
		sentences = deque(phrases)
		sentences.appendleft(base)
		sentences = list(sentences)
		sentence_embeddings = self.model.encode(sentences)
		sim = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
		return sim
	
	def similar(self,pra,ix, phraseO,simStep):
		''' Returns only the phrases that are similar enough, 
			the more toxic the original sentence the more dissimilar the generated one can be. 
		
			Parameters:
			 `pra`	   - list of paraphrases
			 `ix`	   - indexes of non toxic ones
			 `phraseO` - the original sentence
			 `simStep` - toxCategory of the original sentence
		'''
		praGood = list(np.array(pra)[ix])
		sim = self.similarity(phraseO, praGood)[0]
		threshold = 0.57 + 0.1*(3-simStep)
		if len(phraseO.split()) <= 4:
			threshold = 0.9
		ixSim = np.where(sim>threshold)[0]
		return praGood,ixSim
	
	
	def decontracted(self,phrase):
		''' Extends the shortened phrases. '''
		#https://stackoverflow.com/a/47091490
		# specific
		phrase = re.sub(r"won\'t", "will not", phrase)
		phrase = re.sub(r"can\'t", "can not", phrase)
		# general
		phrase = re.sub(r"n\'t", " not", phrase)
		phrase = re.sub(r"\'re", " are", phrase)
		phrase = re.sub(r"\'s", " is", phrase)
		phrase = re.sub(r"\'d", " would", phrase)
		phrase = re.sub(r"\'ll", " will", phrase)
		phrase = re.sub(r"\'t", " not", phrase)
		phrase = re.sub(r"\'ve", " have", phrase)
		phrase = re.sub(r"\'m", " am", phrase)
		return phrase
	
	def delete_vulgar_adj(self,phrase):
		''' Decontract the phrase than replace vulgar adjectives and adverbs.
			The conclusion to remove adj. and adv. was made according to the information analysis in shannon_info.py. '''
		phrase = phrase.replace("’","'").replace("' ","'")
		decon = self.decontracted(phrase)
		## pos tagging:
		splt = self.tokenizer.tokenize(decon)
		tagged = pos_tag(splt)
		## censored:
		prof = profanity.censor(decon)
		profToken = self.tokenizer.tokenize(prof)
		## get censored words:
		intersect = list(set(splt)-set(profToken))
		new_sent = ""
		for e in tagged:
			tag = e[1]
			word = e[0]
			if word not in intersect:
				new_sent+=(word+" ")
				continue;
			if tag.startswith('JJ') or tag.startswith('RB'):
				pass
			else:
				new_sent+=(word+" ")
		return new_sent
	
	def post_processing(self,plist):
		''' Remove sentences containing NationMaster or an american phone number, 
			also remove the sentence from list if its all caps.
			These are artifacts of the paraphraser. '''
		post = [s for s in plist if not(("NationMaster" in s) or ("888-" in s) or ("800-" in s)) and not(s.isupper())]
		return post
	
	
	
	def predict(self, text, toxCategory):
		''' Generates a list of similar nonhateful comments. '''
		#choose paraphrase parameters based on the toxicity of the input sentence
		x,y,z,t = self.values[toxCategory]
		#decontract and delete vulgar adj. and adv.
		newText = self.delete_vulgar_adj(text)
		#paraphrase and asses toxicity
		paraList,toxList,ix = self.paraphrase_toxic(newText,x,y,z,t) 
		if len(ix) > 0: #if there are any non toxic paraphrases
			#get paraphrases that are similar to the original sentence
			simList,ixSim = self.similar(paraList,ix,text,toxCategory)
			if len(ixSim) > 0: #if there are any similar
				simNonToxList = np.array(simList)[ixSim]
				post = self.post_processing(simNonToxList) #post processing
				if len(post)>0:
					return post
					
		#choose the least toxic paraphrased sentence
		simList = self.similarity(text, paraList)[0]
		cond = list(set(np.where(toxList>0.5)[0]).intersection(set(np.where(simList>0.57)[0])))
		if len(cond)==0:
			return [];
		minTox = paraList[np.argmin(toxList[cond])] 
		
		#second paraphrase (same as above)
		paraList,toxList,ix = self.paraphrase_toxic(minTox,x,y,z,t)
		if len(ix) > 0:
			simList,ixSim = self.similar(paraList,ix,text,toxCategory)
			if len(ixSim) > 0:
				simNonToxList = np.array(simList)[ixSim]
				post = self.post_processing(simNonToxList)
				if len(post)>0:
					return post
		
		return [];
	


if __name__ == "__main__":
	#load data (hateful comments, toxCategory for each
	hate = np.loadtxt('data-generated/hate.txt', dtype='str' , delimiter="\n")
	div = np.loadtxt('data-generated/div.txt', dtype='str' , delimiter="\n")
	div = div.astype(int)
	#create a folder for newly generated comments
	if not os.path.exists('dataset'):
		os.makedirs('dataset')
	data={}
	dphate = DPhate() #DPhate init - load models
	for i in range(len(div)):
		print(100*'-')
		print(hate[i])
		data[hate[i]] = dphate.predict(hate[i],div[i])	#predict (generate friendly versions of the input comment)
		print(data[hate[i]])
		#save everything after every tenth example
		if i%10==0:
			fname = "dataset/data" + str(i) + ".json"
			with open(fname,'w') as fp:
				json.dump(data,fp,indent=4)
	
	#with open('data3570.json') as jf:
	#	data = json.load(jf)
