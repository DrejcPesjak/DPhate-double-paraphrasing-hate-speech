import numpy as np
import pandas as pd
microDF = pd.read_csv('microWorkers/CSVReport_a36dfe57fa1a_HG_Page#1_With_PageSize#5000.csv', skiprows=4)
#len(microDF[microDF['Original phrase']=="the pentagon is like a street nigger constantly bumming cigarettes spare change"])
#len(microDF['New phrase'].unique())
#['Original phrase', 'New phrase', 'Similarity (1-5)', 'Hatefulness (1-5)']
microDF = microDF.rename(columns={'Original phrase':'original', 'New phrase':'new', 'Similarity (1-5)':'similarity', 'Hatefulness (1-5)':'hatefulness'})



newDF = pd.DataFrame(columns=microDF.columns)
for i,r in microDF.iterrows():
	ixs = np.where((newDF.original==r.original) & (newDF.new==r.new))[0]
	#if the pair does not exist in the new df, add new row
	if len(ixs) == 0:
		newDF.loc[len(newDF)] = [r.original,r.new, [r.similarity], [r.hatefulness]]
	else:
		ix = ixs[-1]
		#if the pair has already 3 reviews, add a new row
		if len(newDF.iloc[ix].similarity)==3:
			newDF.loc[len(newDF)] = [r.original,r.new, [r.similarity], [r.hatefulness]]
		else:
			#if pair already exists, append sim and hate scores
			newDF.iloc[ix].similarity.append(r.similarity)
			newDF.iloc[ix].hatefulness.append(r.hatefulness)


'''
llen = pd.DataFrame([len(k) for k in newDF.similarity])[0]
newnewDF = newDF[llen>1]
oldDF = newDF.copy()
newDF=newnewDF
'''

#round(sum([1,3,3])/3)
#newDF['sim_avg']= round(sum(newDF.similarity)/3)
sim_avg=[]
for lst in newDF.similarity:
	sim_avg.append( round(sum(lst)/len(lst)) )

newDF['sim_avg']=sim_avg


hate_avg=[]
for lst in newDF.hatefulness:
	hate_avg.append( round(sum(lst)/len(lst)) )

newDF['hate_avg']=hate_avg

'''
for i,r in newDF.iterrows():
     print(r.hatefulness)

for i,r in newDF.iterrows():
     for k in r.hatefulness:
             if k=='\xa0':
                     print(r);print(i)
'''

import matplotlib.pyplot as plt
plt.hist(newDF.sim_avg,bins=[0.5,1.5,2.5,3.5,4.5,5.5])
plt.show()
plt.hist(newDF.hate_avg,bins=[0.5,1.5,2.5,3.5,4.5,5.5])
plt.show()

#similar and off/normal
len(newDF[(newDF.sim_avg>=3) & (newDF.hate_avg<=3)])/len(newDF)
#>>> 0.4007352941176471

#atleast one acceptable
newDF['good'] = (newDF.sim_avg>=3) & (newDF.hate_avg<=3)
ins= newDF.original.unique()
ins = pd.DataFrame(ins,columns=['original'])
atleast = []
for i in range(len(ins)):
	atleast.append(any(newDF[newDF.original==ins.iloc[i].original].good))

sum(atleast)/len(atleast)
#>>> 0.8148148148148148


## ALL, drejc + bosnic account
#similar
len(newDF[newDF.sim_avg>=3])/len(newDF)
#>>> 0.5126146788990825
len(newDF[newDF.hate_avg<=3])/len(newDF)
#>>> 0.8176605504587156
# good: 0.36353211009174313
# atleast: 0.6790123456790124


#count examples: 872 (600+272)
#count people: 61 (15+7+9+13+17)
#countries: ['United States', 'Jamaica', 'United Kingdom', 'Canada', 'Trinidad and Tobago'], ['United Kingdom', 'United States', 'Canada', 'Jamaica'], ['Canada', 'United States', 'Jamaica'], ['Canada', 'United States', 'Australia', 'Jamaica', 'Ireland'], ['United States', 'Jamaica', 'United Kingdom', 'Canada','Trinidad and Tobago', 'Australia']
microDF = pd.read_csv('~/Downloads/CSVReport_3b41d4c96bac_HG_Page#1_With_PageSize#5000(1).csv',skiprows=4)
len(microDF[microDF.STATUS=='OK'].USER_ID.unique())
microDF[microDF.STATUS=='OK'].COUNTRY.unique()

#CSVReport_3de27bb1c7a8_HG_Page#1_With_PageSize#5000.csv
#CSVReport_8ef33c3264d6_HG_Page#1_With_PageSize#5000.csv
#CSVReport_a36dfe57fa1a_HG_Page#1_With_PageSize#5000.csv
#CSVReport_e178db2ff50e_HG_Page#1_With_PageSize#5000.csv
#>>> allDF1.shape
#(100, 6)
#>>> allDF2.shape
#(100, 6)
#>>> allDF3.shape
#(200, 6)
#>>> allDF4.shape
#(200, 6)
allDF = pd.concat([allDF1,allDF2,allDF3,allDF4])
allDF.to_csv('microWorkers/combined.csv',index=False,quoting=csv.QUOTE_NONNUMERIC)
