import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
stop_words = stopwords.words('english')

second_dict = { 'hi' : [87, 23],  'hello' : [19, 53], 'bye' : [12, 64]}
first_dict = { 'hi' : [87, 23],  'B' : [19, 53], 'C' : [34, 14]}

for key, value in first_dict.items():
    value.append(55)

print(first_dict)