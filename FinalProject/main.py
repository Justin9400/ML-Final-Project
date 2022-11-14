import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = stopwords.words('english')
removedStopWords = []

# Loads the CSV file
def loadCSV(path):
    df = pd.read_csv(path)
    return df

def splitDf(df):
    for row in df:
        df.loc[df['favorite_color']] == 'yellow'

# Tokenizes and removes all of the stop words from each text
def removeStopWords(df):
    # For each text in the v2 column
    for text in df["v2"]:
        tokenizedText = word_tokenize(text) # Tokenizes the text 
        for word in tokenizedText: # For each word in the text
            if word in stop_words: # Checks if the word is a stop word
                tokenizedText.remove(word) # Removes the word from the text if it is a stop word
        removedStopWords.append(tokenizedText) # Appends the new text without the stop words to a list

def wordStemming():
    for text in removedStopWords:
        for words in text:
            print(words, " : ", ps.stem(words))

def frequentWordsIdentification():


def main():
    path = "spam.csv"
    df = loadCSV(path)
    removeStopWords(df)
    wordStemming()

main()