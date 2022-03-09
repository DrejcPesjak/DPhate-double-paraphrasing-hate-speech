
''' Pegasus paraphraser parameter testing '''

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




def similarity(base, phrases):
	sentences = deque(phrases)
	sentences.appendleft(base)
	sentences = list(sentences)
	sentence_embeddings = model.encode(sentences)
	sim = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
	return sim

def grammar(inputT):
	inputs = tokenizerG(inputT, return_tensors="pt")
	outputs = modelG(**inputs)
	score = F.softmax(outputs.logits, dim=-1)
	score = score.tolist()[0]
	return score



values=[]
nseq = np.arange(10,41,10)
#beam = np.arange(60,301,10)
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



values= [[ 20., 100.,  50.,   1.],
		[ 30.,  80.,  40.,   3.],
		[ 30., 100.,  50.,   3.],
		[ 30., 200., 100.,   2.],
		[ 40., 100.,  50.,   2.],
		[ 40., 200., 100.,   2.],
		[ 40., 300., 150.,   2.]]
		
values= [[ 20., 100.,  25.,   1.],
		[ 30., 100.,  50.,   3.],
		[ 40., 100.,  50.,   2.],
		[ 40., 300., 150.,   2.]]


