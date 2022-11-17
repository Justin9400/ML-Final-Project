import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
stop_words = stopwords.words('english')

# # For each text in the v2 column
# for text in list
#     tokenizedText = word_tokenize(text) # Tokenizes the text 
#     # print(tokenizedText)
#     for word in tokenizedText # For each word in the text
#         # print(word)
#         if word.lower() in stop_words # Checks if the word is a stop word
#             tokenizedText.remove(word) # Removes the word from the text if it is a stop word

#     cleanedText.append(tokenizedText) # Appends the new text without the stop words to a list

    # for text in list
    #     tokenizedText = word_tokenize(text) # Tokenizes the text 
    #     for word in tokenizedText
    #         if word.lower() not in stop_words
    #             nonStopWord = word
    #     cleanedText.append(nonStopWord)
