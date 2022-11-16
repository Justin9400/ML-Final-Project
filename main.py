import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = stopwords.words('english')

# Loads the CSV file
def loadCSV(path):
    df = pd.read_csv(path)
    return df

def splitSpamHam(df):
    spamText = df.loc[df['v1'] == 'spam']
    hamText = df.loc[df['v1'] == 'ham']
    spamText = spamText['v2'].tolist()
    hamText = hamText['v2'].tolist()
    return spamText, hamText
        

# Tokenizes and removes all of the stop words from each text
def removeStopWords(list):
    cleanedText = []
    for text in list:
        tokenizedText = word_tokenize(text) # Tokenizes the text 
        for word in tokenizedText:
            if word.lower() not in stop_words:
                nonStopWord = word
        cleanedText.append(nonStopWord)

    return cleanedText

def wordStemming(cleanedText):
    print(cleanedText)
    for text in cleanedText:
        print("hello")
        # for words in text:
            # print(words)
            # print(words, " : ", ps.stem(words))

# def frequentWordsIdentification():


def main():
    # print(stop_words)
    path = "spam.csv"
    df = loadCSV(path)
    spamText, hamText = splitSpamHam(df)
    cleanedSpamText = removeStopWords(spamText)
    cleanedHamText = removeStopWords(hamText)
    print(cleanedSpamText, cleanedHamText)
    wordStemming(cleanedSpamText)
    wordStemming(cleanedHamText)


main()