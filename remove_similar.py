def str_equal(word1, word2):
	'''
	 count number of differences in two strings
	'''
	if word1!=word2:
		count = sum(1 for a, b in zip(word1, word2) if a != b) + abs(len(word1) - len(word2))
		return count
	else:
		return 0


def para_diff(lscomp):
	'''
	 return a list of pariwise correspondece indexes of sentences with less than two differences
	'''
	dif_c = [[] for x in range(len(lscomp))]
	for i in range(0,len(lscomp)):
		for j in range(i+1,len(lscomp)):
			c = str_equal(lscomp[i],lscomp[j])
			if c <= 2:
				dif_c[i].append(j)
				dif_c[j].append(i)
	return dif_c


for key in data:
	r = data[key]
	r = [x.lower() for x in r]
	ls = para_diff(r)
	#count how many correspondences each sentence has
	lenls = [len(x) for x in ls]
	#start of with unique sentences (num.of corr. == 0) 
	para = [r[i] for i in range(len(r)) if lenls[i]==0 ]
	while sum(lenls)>0:
		#get the one with most corr.
		M = max(lenls)
		ixM = lenls.index(M)
		#add it
		para.append(r[ixM])
		#eliminate the corr. sentences
		for i in ls[ixM]:
			lenls[i] = 0
		lenls[ixM] = 0
	data[key] = para


