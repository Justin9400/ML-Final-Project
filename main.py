import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from itertools import islice
# nltk.download('stopwords')
ps = PorterStemmer()
stop_words = stopwords.words('english')

'''
    Loads the dataset as a dataframe 
'''
def loadCSV(path):
    df = pd.read_csv(path)
    return df

'''
    Splits the dataset into spam and ham text
'''
def splitSpamHam(df):
    # Gets all of the text that is classified as spam and ham
    spamText = df.loc[df['v1'] == 'spam']
    hamText = df.loc[df['v1'] == 'ham']

    # Converts the text from column v2 to a list
    spamText = spamText['v2'].tolist()
    hamText = hamText['v2'].tolist()

    return spamText, hamText
        
'''
    Tokenizes and removes all of the stop words from the given text
'''
def removeStopWords(list):
    cleanedText = []
    # For each sentence in the list 
    for text in list:
        # Tokenizes the current sentence
        tokenizedText = word_tokenize(text) 
        # For each word in the current sentence
        for word in tokenizedText:
            # If the word is not a stop word then add it to the list
            nonStopWord = [word.lower() for word in tokenizedText if word.lower() not in stop_words]

        cleanedText.append(nonStopWord)

    return cleanedText

'''
    Stems every word in the given text
'''
def wordCount(cleanedText):
    stemDict = {}
    # For each sentence in the data set
    for text in cleanedText:
        # For each word in the current sentence
        for word in text:
            # stems the current word
            stem = ps.stem(word)
            # If the stem is in the dictionary then increment the count 
            if stem in stemDict:
                stemDict[stem] += 1
            # If the stem is not in the list then add it to the dictionary
            else:
                stemDict[stem] = 1

    return stemDict

'''
    Sorts the dictionary of stem counts in descending order
'''
def rankStems(stemDict):
    # Sorts the dictionary passed in as a parameter in ascending order
    rankedDescending = dict(sorted(stemDict.items(), key=lambda y: y[1], reverse=True))
    return rankedDescending

'''
    Gets the top 50 most frequent stems for a given dictionary
'''
def frequentWordsIdentification(rankStems):
    # Gets the top 50 most frequent stems in the dictionary passed in
    topFifty = dict(islice(rankStems.items(), 50))
    return topFifty

'''
    Gets the top 100 unique stems from both spam and ham
'''
def getUniqueSpamHamStems(spamStemDict, hamStemDict):

    # Gets values that are in ham but not in spam and in spam but not in ham
    uniqueHam = { k : hamStemDict[k] for k in set(hamStemDict) - set(spamStemDict) }
    uniqueSpam = { k : spamStemDict[k] for k in set(spamStemDict) - set(hamStemDict) }

    # Ranks the stems for spam and ham
    uniqueSpam = rankStems(uniqueSpam)
    uniqueHam = rankStems(uniqueHam)

    # Gets the top 50 results from spam and ham
    uniqueSpam = dict(islice(uniqueSpam.items(), 50))
    uniqueHam = dict(islice(uniqueHam.items(), 50))

    return uniqueSpam , uniqueHam

def getTopOneHundredUniqueStems(uniqueSpam, uniqueHam):
# Combines the spam and ham dictionaries
    totalStemDict = uniqueSpam | uniqueHam

    # Gets the top 100 
    topOneHundredStems = dict(islice(totalStemDict.items(), 100))
    return topOneHundredStems 

'''
    Creates a result dataframe from the top 100 stems
'''
def createDataframe(topOneHundredStems):
    stems = []
    counts = []

    # For each key and value in the dictionary append the keys to a list and the values to a list
    for key, value in topOneHundredStems.items():
        stems.append(key)
        counts.append(value)

    # Create the final dictionary
    resultsDF = {"Stems" : stems, "Counts" : counts}

    # Create the dataframe
    df = pd.DataFrame(resultsDF)

    return df

def tokenizeDf(df):
    tokenizedLine = []
    dfAsList = []
    # For each stem in the data set
    for text in df['v2']:
        temp = []
        tokenizedLine = word_tokenize(text) 
        # For each word in the current sentence
        for word in tokenizedLine:
            # stems the current word
            word = ps.stem(word)
            temp.append(word)

        dfAsList.append(temp)
    return dfAsList

def createMatrix(df, topOneHundredStems):
    dfAsList = tokenizeDf(df)

    for line in dfAsList:
        for stem, value in topOneHundredStems.items():
            count = line.count(stem)
            value.append(count)

    return topOneHundredStems

def getKeys(topOneHundredStems):
    stems = {}
    for key, value in topOneHundredStems.items():
        stems[key] = []

    return stems

def main():
    # Loads the dataset as a dataframe
    path = "spam.csv"
    df = loadCSV(path)

    # Splits the dataset into spam and ham
    spamText, hamText = splitSpamHam(df)

    # Removes the stop words from both spam and ham texts
    cleanedSpamText = removeStopWords(spamText)
    cleanedHamText = removeStopWords(hamText)

    # Stems all the remaining words in spam and ham texts
    spamStemDict = wordCount(cleanedSpamText)
    hamStemDict = wordCount(cleanedHamText)

    # Sorts the stem counts in descending order
    spamStemDict = rankStems(spamStemDict)
    hamStemDict = rankStems(hamStemDict)

    uniqueSpam, uniqueham = getUniqueSpamHamStems(spamStemDict, hamStemDict)

    # print(uniqueSpam)
    # Gets the top 100 unique stems, 50 from spam and 50 from ham
    topOneHundredStems = getTopOneHundredUniqueStems(uniqueSpam, uniqueham)

    topOneHundredStemsList = getKeys(topOneHundredStems)
    top50SpamStemsList = getKeys(uniqueSpam)
    top50HamStemsList = getKeys(uniqueham)
    # # Converts the dictionary to a dataframe
    # df = createDataframe(topOneHundredStems)
    # print(topOneHundredStems)
    # # Exports the top 100 stems from the data set to a csv

    top100Dict = createMatrix(df, topOneHundredStemsList)
    top50SpamDict = createMatrix(df, top50SpamStemsList)
    top50HamDict = createMatrix(df, top50HamStemsList)

    df1 = pd.DataFrame.from_dict(top100Dict)
    df2 = pd.DataFrame.from_dict(top50SpamDict)
    df3 = pd.DataFrame.from_dict(top50HamDict)

    df1["Class"] = df['v1']
    df1.to_csv('Top 100 Stems.csv', index = False, encoding='utf-8')

    df2.to_csv('Top 50 Spam Stems.csv', index = False, encoding='utf-8')
    df3.to_csv('Top 50 Ham Stems.csv', index = False, encoding='utf-8')



main()