import json
import pandas as pd
import numpy as np

data={}
with open('dataCorr.json') as jf:
	data = json.load(jf)

restructured = {'source_text':[], 'target_text':[]}

for k in data:
	for e in data[k]:
		restructured['source_text'].append(k)
		restructured['target_text'].append(e)

df = pd.DataFrame(restructured)

# get the index, for 95%-5% train-eval split
s = restructured['source_text']
l = len(s)
last_text = ""
index = 0
for i in range(l-1):
	curr_text = s[l-i-1]
	if curr_text != last_text:
		index = l-i-1
		last_text = curr_text
	if i >= l*0.05:
		break;

fname = "restructured4_" + str(index) + ".pkl"
df.to_pickle(fname,protocol=4)



from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained("t5","t5-base")
#df[:index+1]
#df[index+1:]
#model.train(train_df=df[:index+1], eval_df=df[index+1:], source_max_token_len=300, target_max_token_len=200,max_epochs=2,outputdir="outputs",use_gpu=False)

model.train(train_df=df[:index+1], 
            eval_df=df[index+1:], 
            source_max_token_len = 512, 
            target_max_token_len = 128,
            batch_size = 8,
            max_epochs = 5,
            use_gpu = False,
            outputdir = "outputs")

#model.predict(sentence)



