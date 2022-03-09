import numpy as np
phrasesNP = np.loadtxt('dataset/phrases.txt', dtype='str' , delimiter="\n")
phrases = list(phrasesNP)

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

alls = " ".join(phrases)
wordlist = tokenizer.tokenize(alls)
uwordlist = set(wordlist)
dict_words = dict.fromkeys(uwordlist,0)

for i in wordlist:
	dict_words[i] += 1

prob = {}
for i in dict_words:
	prob[i] = dict_words[i] / sum(dict_words.values())


# “Shannon information,” “self-information,” or simply the “information,”
# information(x) = -log( p(x) )
import math
theinfo = -math.log(prob['the'],2)
info = {}
for i in dict_words:
	info[i] = -math.log(prob[i],2)





''' *************************************************** '''
from better_profanity import profanity
profanity.load_censor_words()

i=-1
i+=1
phrase = hate[i]
p2=profanity.censor(phrase)

h = tokenizer.tokenize(phrase)
g = tokenizer.tokenize(p2)
l = list(set(h)-set(g))
s_info = [info[x.lower()] for x in l]
sum(s_info)

phraseO = phrase
phrase = p2
#phrase = phrase.replace('**** ','')

pra = get_response(phrase, num_return_sequences=int(x), num_beams=int(y), groups=int(z), diversityP=t)
results = modelD.predict(pra)
tox = np.array(results['toxicity'])
ix = np.where(tox < 0.5 )[0]
len(ix)

praGood = list(np.array(pra)[ix])
sim = similarity(phraseO, praGood)[0]
ixSim = np.where(sim>0.57)[0]
len(ixSim)

print(phrase)
print(phraseO)

print_list(np.array(praGood)[ixSim])
phrase.count('****') / len(phrase.split())
sum(s_info)


''' *************************calc info of censored words**************************** '''

#end=0
#while True:
#	start = p2.find('****', end)
#	end = phrase.find(' ',start)
#	print(start, end)
#	word = ''
#	if end == -1:
#		word = phrase[start:]
#		break;
#	else:
#		word = phrase[start:end]
#	print(info[word.lower()])

h = tokenizer.tokenize(phrase)
g = tokenizer.tokenize(p2)
l = list(set(h)-set(g))
i = [info[x.lower()] for x in l]



''' **********************analysis of info per POS tag******************************* '''

from nltk.tag.stanford import StanfordPOSTagger
st = StanfordPOSTagger('english-bidirectional-distsim.tagger')
#from nltk.tokenize import RegexpTokenizer
#tokenizer = RegexpTokenizer(r'\w+')

pos_info = {}
pos_count = {}
for h in hate:
	splt = tokenizer.tokenize(h)
	tagged = st.tag(splt)
	## censored:
	p2 = profanity.censor(h)
	p2token = tokenizer.tokenize(p2)
	intersect = list(set(splt)-set(p2token))
	for e in tagged:
		tag = e[1]
		word = e[0]
		if word not in intersect:
			continue;
		w_info = 0.0
		try:
			w_info = info[word.lower()]
		except KeyError:
			continue;
		if tag == 'FW':
			print(h, word, w_info)
		if tag in pos_info:
			pos_info[tag] += w_info
			pos_count[tag] += 1
		else:
			pos_info[tag] = w_info
			pos_count[tag] = 1


i_per_pos = {}
for tg in pos_info:
	f = pos_info[tg]/pos_count[tg]
	#print(tg, f)
	i_per_pos[tg] = f
	
#all_sorted = {k: v for k, v in sorted(i_per_pos.items(), key=lambda item: item[1])}
relevant_tags = {k:i_per_pos[k] for k,v in pos_count.items() if v>30}
relevant_sorted = {k: v for k, v in sorted(relevant_tags.items(), key=lambda item: item[1])}
# {'DT': 6.23161312176529, 'PRP': 6.663108188877318, 'IN': 8.024913727923057, 'RB': 9.395283695147434, 'VBP': 9.920054652224884, 'VB': 10.13504759704436, 'JJ': 11.282893558661753, 'NN': 11.979298992357021, 'NNS': 12.021336296417894}
# determiner , personal pronoun , preposition , adverb , verb , adjective , noun


''' censored '''
#>>> pos_info
#{'NNS': 156.75022004436792, 'VBG': 20.88505777349939, 'VBD': 9.810621449838727, 'NN': 524.6288146731758, 'VB': 110.04383509127192, 'JJ': 201.9265734869597, 'VBP': 49.267278475478435, 'VBN': 59.51416286625181, 'FW': 65.64212194977752, 'RB': 6.3613928089523455, 'VBZ': 12.236886204540825, 'NNP': 32.25944750179015, 'WRB': 8.793942708692096, 'TO': 5.500484273222535}
#>>> pos_count
#{'NNS': 12, 'VBG': 2, 'VBD': 1, 'NN': 48, 'VB': 12, 'JJ': 20, 'VBP': 5, 'VBN': 6, 'FW': 7, 'RB': 1, 'VBZ': 1, 'NNP': 3, 'WRB': 1, 'TO': 1}
#  verb, adjective, noun
#>>> i_per_pos
#{'NNS': 13.06251833703066, 'VBG': 10.442528886749695, 'VBD': 9.810621449838727, 'NN': 10.92976697235783, 'VB': 9.170319590939327, 'JJ': 10.096328674347985, 'VBP': 9.853455695095686, 'VBN': 9.9190271443753, 'FW': 9.37744599282536, 'RB': 6.3613928089523455, 'VBZ': 12.236886204540825, 'NNP': 10.753149167263382, 'WRB': 8.793942708692096, 'TO': 5.500484273222535}
''' c. corrected '''
#>>> join_count
#{'NN': 66, 'VB': 36, 'JJ': 15, 'RB': 9, 'MD': 1, 'WR': 1, 'TO': 1}
#>>> i_per_pos_join
#{'NN': 10.980905868262893, 'VB': 9.523328078596386, 'JJ': 9.843543969991522, 'RB': 9.151486294218174, 'MD': 12.49992061037462, 'WR': 8.793942708692096, 'TO': 5.500484273222535}


# adjective, adverb -> descriptive non essential for sentence generation
# FW foreign word -> check what is classified as foreign and if it can be deleted
# verb, noun - to important/essential

''' **********************analysis of info per POS tag - corrected (decontracted + nltk.pos_tag)******************************* '''
pos_info = {}
pos_count = {}
for h in hate:
	dph = decontracted(h.replace("’","'"))
	splt = tokenizer.tokenize(dph)
	tagged = pos_tag(splt)
	## censored:
	p2 = profanity.censor(h)
	p2token = tokenizer.tokenize(p2)
	intersect = list(set(splt)-set(p2token))
	for e in tagged:
		tag = e[1]
		word = e[0]
		if word not in intersect:
			continue;
		w_info = 0.0
		try:
			w_info = info[word.lower()]
		except KeyError:
			continue;
		if tag == 'FW':
			print(h, word, w_info)
		if tag in pos_info:
			pos_info[tag] += w_info
			pos_count[tag] += 1
		else:
			pos_info[tag] = w_info
			pos_count[tag] = 1

join_i = {}
join_c = {}
for tag in pos_info:
	new_tag = tag[:2]
	if new_tag in join_i:
		join_i[new_tag] += pos_info[tag]
		join_c[new_tag] += pos_count[tag]
	else:
		join_i[new_tag] = pos_info[tag]
		join_c[new_tag] = pos_count[tag]


i_per_pos_join = {}
for tg in join_i:
	f = join_i[tg]/join_c[tg]
	i_per_pos_join[tg]=f


