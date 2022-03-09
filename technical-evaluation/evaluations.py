#datacsv.to_csv('testing/datapredictions/simSCE05.csv',index=False,quoting=csv.QUOTE_NONNUMERIC)
datacsv = pd.read_csv('testing/datapredictions/simSCE05.csv')
hatecsv = pd.read_csv("testing/datapredictions/hate3269.csv")
outs = hatecsv[hatecsv.ref_ix>=0].copy()
out2 = outs.reset_index(drop=True)
datacsv["hate"]=out2.hate
datacsvC = datacsv.drop('sim',1)
#datacsvC.to_csv('testing/datapredictions/combined.csv',index=False,quoting=csv.QUOTE_NONNUMERIC)
datacsvC = pd.read_csv('testing/datapredictions/combined.csv')
df = hatecsv



>>> len(datacsvC[datacsvC.sim_pred==1])/len(datacsvC)
0.9230825084105816
>>> len(datacsvC[datacsvC.sim_pred==0])/len(datacsvC)
0.0769174915894184


>>> len(datacsvC[datacsvC.hate>0])/len(datacsvC)
0.3113004332357141
>>> len(datacsvC[datacsvC.hate<2])/len(datacsvC)
0.8177021564973256
>>> len(datacsvC[datacsvC.hate==2])/len(datacsvC)
0.18229784350267444
>>> len(datacsvC[datacsvC.hate==1])/len(datacsvC)
0.12900258973303966
>>> len(datacsvC[datacsvC.hate==0])/len(datacsvC)
0.6886995667642859



>>> len(datacsvC[(datacsvC.sim_pred==1) & (datacsvC.hate<2)])/len(datacsvC)
0.7462545683374882
>>> len(datacsvC[(datacsvC.sim_pred==1) & (datacsvC.hate<1)])/len(datacsvC)
0.6230365224967931


outs = df[df.ref_ix>=0]
better=[]
for i,h in outs.iterrows():
	df_hate = df.iloc[h.ref_ix].hate
	if df_hate==2:
		better.append(df_hate>h.hate)
	if len(better)%100==0:
		print(better[-10:])

print(sum(better)/len(better))
#>>> 0.7834065199


better=[]
for i,h in outs.iterrows():
	df_hate = df.iloc[h.ref_ix].hate
	if df_hate>0:
		better.append(df_hate>h.hate)
	if len(better)%1000==0:
		print(better[-25:])

>>> sum(better)/len(better)
0.7791662501249625


better=[]
for i,h in out2.iterrows():
	df_hate = df.iloc[h.ref_ix].hate
	if df_hate>0:
		better.append((df_hate>h.hate) & (datacsvC.iloc[i].sim_pred==1))
	if len(better)%1000==0:
		print(better[-25:])

>>> sum(better)/len(better)
0.7074127761671498

datacsvC['better'] = False
better=[]
for i,h in out2.iterrows():
	df_hate = df.iloc[h.ref_ix].hate
	if df_hate>0:
		better.append((df_hate>h.hate) & (datacsvC.iloc[i].sim_pred==1))
		if better[-1]==True:
			datacsvC.at[i,'better'] = True
	if len(better)%1000==0:
		print(better[-25:])

#datacsvC.to_csv('testing/datapredictions/combinedFinal.csv',index=False,quoting=csv.QUOTE_NONNUMERIC)


ins = datacsvC.source_text.unique()
ins = pd.DataFrame(ins,columns=['input_text'])
ins['better']=False;j=0
for i,r in datacsvC.iterrows():
	if r.source_text != ins.input_text[j]:
		j+=1
	if r.better == True:
		ins.at[j,'better'] = True


>>> ins.better.sum()/len(ins)
0.8436830835117773



#povprecno stevilo generiranih povedi za en input
>>> outs.ref_ix.value_counts().mean()
12.639033343530132
