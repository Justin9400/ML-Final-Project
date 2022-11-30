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
def getTopOneHundredUniqueStems(spamStemDict, hamStemDict):

    # Gets values that are in ham but not in spam and in spam but not in ham
    uniqueHam = { k : hamStemDict[k] for k in set(hamStemDict) - set(spamStemDict) }
    uniqueSpam = { k : spamStemDict[k] for k in set(spamStemDict) - set(hamStemDict) }

    # Ranks the stems for spam and ham
    uniqueSpam = rankStems(uniqueSpam)
    uniqueHam = rankStems(uniqueHam)

    # Gets the top 50 results from spam and ham
    uniqueSpam = dict(islice(uniqueSpam.items(), 50))
    uniqueHam = dict(islice(uniqueHam.items(), 50))

    # Combines the spam and ham dictionaries
    totalStemDict = uniqueSpam | uniqueHam

    # # Creates the classifier list to tell which stem is spam or ham
    # spam = ['spam'] * 50
    # ham = ['ham'] * 50
    # classifier = spam + ham 

    # Gets the top 100 
    topOneHundredStems = dict(islice(totalStemDict.items(), 100))
    return topOneHundredStems #, classifier

'''
    Creates a result dataframe from the top 100 stems
'''
def createDataframe(topOneHundredStems, classifier):
    stems = []
    counts = []

    # For each key and value in the dictionary append the keys to a list and the values to a list
    for key, value in topOneHundredStems.items():
        stems.append(key)
        counts.append(value)

    # Create the final dictionary
    resultsDF = {"Stems" : stems, "Counts" : counts, "Class" : classifier}

    # Create the dataframe
    df = pd.DataFrame(resultsDF)

    return df

def createMatrix(topOneHundredStems, spamStemDict, hamStemDict):
    unionDict = spamStemDict | hamStemDict
    unionDict = list(unionDict.keys())
    print(unionDict)
    # # For each stem in the data set
    # for stem in unionDict:
    #     print(stem) 


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

    # Gets the top 50 stems from the spam and ham 
    topFiftyStemsSpam = frequentWordsIdentification(spamStemDict)
    topFiftyStemsHam = frequentWordsIdentification(hamStemDict)

    # Gets the top 100 unique stems from the whole data set
    topOneHundredStems, classifierList = getTopOneHundredUniqueStems(spamStemDict, hamStemDict)

    # Converts the dictionary to a dataframe
    df = createDataframe(topOneHundredStems, classifierList)

    # Exports the top 100 stems from the data set to a csv
    df.to_csv('Top 100 Stems.csv', index = False, encoding='utf-8')

    createMatrix(topOneHundredStems, spamStemDict, hamStemDict)


main()