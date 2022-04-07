# DPhate-double-paraphrasing-hate-speech
Bachelor's thesis on removing hate from online comments using paraphrasing: algorithm DPhate.


## Usage
To recreate the data generated in the research paper (also available [here](data-generated/data3570.json)), where the input are hateful sentences from the Hatexplain dataset, use:
```python
python3 DPhate.py
```

To test the algorithm on your own examples use the followoing python code:
```python
from DPhate import DPhate
dphate = DPhate()
phrase = "I fucking love your mother."
toxicity = dphate.modelD.predict(phrase)['toxicity']
toxCategory = int((toxicity-0.5)//0.125)
dphate.predict(phrase,toxCategory)
```

Fine-tuned T5 models are too big for GitHub and can be downloaded [here](https://drive.google.com/file/d/16d7xA2BU_-vIsG0Efvw-4JdxyQm9Jwsb/view). It is a 2.3GB zip file, which contains 3 different T5 models.
