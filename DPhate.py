
''' *************************DPhate algorithm**************************** '''

import re
def decontracted(phrase):
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


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk import pos_tag
from better_profanity import profanity
profanity.load_censor_words()

def paraphrase_toxic(phrase,x,y,z,t):
	pra = get_response(phrase, num_return_sequences=int(x), num_beams=int(y), groups=int(z), diversityP=t)
	results = modelD.predict(pra)
	tox = np.array(results['toxicity'])
	ix = np.where(tox < 0.5 )[0]
	return pra,tox,ix

def similar(pra,ix, phraseO,simStep):
	praGood = list(np.array(pra)[ix])
	sim = similarity(phraseO, praGood)[0]
	threshold = 0.57 + 0.1*(3-simStep)
	if len(phraseO.split()) <= 4:
		threshold = 0.9
	print(threshold, sim)
	ixSim = np.where(sim>threshold)[0]
	return praGood,ixSim

def post_processing(plist):
	post = [s for s in plist if not(("NationMaster" in s) or ("888-" in s) or ("800-" in s)) and not(s.isupper())]
	return post

def delete_vulgar_adj(h):
	h = h.replace("’","'").replace("' ","'")
	dph = decontracted(h)
	## pos tagging:
	splt = tokenizer.tokenize(dph)
	tagged = pos_tag(splt)
	## censored:
	p2 = profanity.censor(dph)
	p2token = tokenizer.tokenize(p2)
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


hate = np.loadtxt('hate.txt', dtype='str' , delimiter="\n")
div = np.loadtxt('div.txt', dtype='str' , delimiter="\n")
div = div.astype(int)
data={}
#for i in range(len(hate)):
for i in range(len(div)):
	print(100*'*', i)
	x,y,z,t = values[div[i]]
	phraseO = hate[i]
	print(phraseO)
	phrase = delete_vulgar_adj(phraseO)
	pra,tox,ix = paraphrase_toxic(phrase,x,y,z,t)
	if len(ix) > 0:
		print("first: ix > 0")
		praGood,ixSim = similar(pra,ix,phraseO, div[i])
		if len(ixSim) > 0:
			print("first: ixSim > 0")
			plist = np.array(praGood)[ixSim]
			post = post_processing(plist)
			if len(post)>0:
				print(post)
				data[phraseO]=post
				if i%20==0:
					fname = "dataset3/data" + str(i) + ".json"
					with open(fname,'w') as fp:
						json.dump(data,fp,indent=4)
				continue;
	print("going second paraphrase")
	#phraseO = phrase
	sim = similarity(phraseO, pra)[0]
	#cond = np.where(tox>0.5) and np.where(sim>0.57)
	cond = list(set(np.where(tox>0.5)[0]).intersection(set(np.where(sim>0.57)[0])))
	if len(cond)==0:
		continue;
	phrase = pra[np.argmin(tox[cond])]
	#phrase = pra[np.argmin(tox[np.where(tox>0.5)])]
	pra,tox,ix = paraphrase_toxic(phrase,x,y,z,t)
	if len(ix) > 0:
		print("second: ix > 0")
		praGood,ixSim = similar(pra,ix,phraseO,div[i])
		if len(ixSim) > 0:
			print("second: ixSim > 0")
			plist = np.array(praGood)[ixSim]
			post = post_processing(plist)
			if len(post)>0:
				print(post)
				data[phraseO]=post
	if i%20==0:
		fname = "dataset3/data" + str(i) + ".json"
		with open(fname,'w') as fp:
			json.dump(data,fp,indent=4)

