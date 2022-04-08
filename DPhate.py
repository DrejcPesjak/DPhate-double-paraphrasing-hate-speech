''' *************************DPhate algorithm**************************** '''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
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
		#
		model_name = 'tuner007/pegasus_paraphrase'
		self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.tokenizerP = PegasusTokenizer.from_pretrained(model_name)
		self.modelP = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)
		
		self.model = SentenceTransformer('bert-base-nli-mean-tokens')
		
		self.modelD = Detoxify('original', device=self.torch_device)

		self.values= [[ 20., 100.,  25.,   1.],
				[ 30., 100.,  50.,   3.],
				[ 40., 100.,  50.,   2.],
				[ 40., 300., 150.,   2.]]
		
		self.tokenizer = RegexpTokenizer(r'\w+')
		profanity.load_censor_words()

	def get_response(self,input_text,num_return_sequences=20,num_beams=100, groups=25, diversityP=1.0):
	  batch = self.tokenizerP([input_text],truncation=True,padding='longest', return_tensors="pt").to(self.torch_device)
	  translated = self.modelP.generate(**batch,
	  								num_beams=num_beams, 
	  								num_return_sequences=num_return_sequences,
	  								num_beam_groups=groups,
	  								diversity_penalty=diversityP) 
	  tgt_text = self.tokenizerP.batch_decode(translated, skip_special_tokens=True)
	  return tgt_text


	def similarity(self,base, phrases):
		sentences = deque(phrases)
		sentences.appendleft(base)
		sentences = list(sentences)
		sentence_embeddings = self.model.encode(sentences)
		sim = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
		return sim



	def decontracted(self,phrase):
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


	def paraphrase_toxic(self,phrase,x,y,z,t):
		pra = self.get_response(phrase, num_return_sequences=int(x), num_beams=int(y), groups=int(z), diversityP=t)
		results = self.modelD.predict(pra)
		tox = np.array(results['toxicity'])
		ix = np.where(tox < 0.5 )[0]
		return pra,tox,ix

	def similar(self,pra,ix, phraseO,simStep):
		praGood = list(np.array(pra)[ix])
		sim = self.similarity(phraseO, praGood)[0]
		threshold = 0.57 + 0.1*(3-simStep)
		if len(phraseO.split()) <= 4:
			threshold = 0.9
		print(threshold, sim)
		ixSim = np.where(sim>threshold)[0]
		return praGood,ixSim

	def post_processing(self,plist):
		post = [s for s in plist if not(("NationMaster" in s) or ("888-" in s) or ("800-" in s)) and not(s.isupper())]
		return post

	def delete_vulgar_adj(self,h):
		h = h.replace("’","'").replace("' ","'")
		dph = self.decontracted(h)
		## pos tagging:
		splt = self.tokenizer.tokenize(dph)
		tagged = pos_tag(splt)
		## censored:
		p2 = profanity.censor(dph)
		p2token = self.tokenizer.tokenize(p2)
		## get censored words:
		intersect = list(set(splt)-set(p2token))
		new_sent = ""
		for e in tagged:
			tag = e[1]
			word = e[0]
			if word not in intersect:
				new_sent+=(word+" ")
				continue;
			if tag.startswith('JJ') or tag.startswith('RB'): #or tag == 'FW'
				pass
			else:
				new_sent+=(word+" ")
		return new_sent



	def predict(self, text, toxCategory):
	
		x,y,z,t = self.values[toxCategory]
		
		newText = self.delete_vulgar_adj(text)
		paraList,toxList,ix = self.paraphrase_toxic(newText,x,y,z,t)
		if len(ix) > 0:
			print("first: ix > 0")
			simList,ixSim = self.similar(paraList,ix,text,toxCategory)
			if len(ixSim) > 0:
				print("first: ixSim > 0")
				simNonToxList = np.array(simList)[ixSim]
				post = self.post_processing(simNonToxList)
				if len(post)>0:
					return post
					
		print("going second paraphrase")
		simList = self.similarity(text, paraList)[0]
		cond = list(set(np.where(toxList>0.5)[0]).intersection(set(np.where(simList>0.57)[0])))
		if len(cond)==0:
			return [];
		minTox = paraList[np.argmin(toxList[cond])]
		
		paraList,toxList,ix = self.paraphrase_toxic(minTox,x,y,z,t)
		if len(ix) > 0:
			print("second: ix > 0")
			simList,ixSim = self.similar(paraList,ix,text,toxCategory)
			if len(ixSim) > 0:
				print("second: ixSim > 0")
				simNonToxList = np.array(simList)[ixSim]
				post = self.post_processing(simNonToxList)
				if len(post)>0:
					return post
		
		return [];
	


if __name__ == "__main__":
	print("main-start")
	hate = np.loadtxt('data-generated/hate.txt', dtype='str' , delimiter="\n")
	div = np.loadtxt('data-generated/div.txt', dtype='str' , delimiter="\n")
	div = div.astype(int)
	data={}
	dphate = DPhate()
	for i in range(len(div)):
		print(100*'-')
		print(hate[i])
		data[hate[i]] = dphate.predict(hate[i],div[i])
		print(data[hate[i]])
		if i%10==0:
			fname = "dataset3/data" + str(i) + ".json"
			with open(fname,'w') as fp:
				json.dump(data,fp,indent=4)
	
	#with open('data3570.json') as jf:
	#	data = json.load(jf)
