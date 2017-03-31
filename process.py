
from import_data import df_train
from import_data import df_test

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation
from tqdm import tqdm
import pandas as pd


def apply_and_add_function(df, function, col1, col2, newname1, newname2):
    '''arguments:
    the df you want to add the new info onto
    the function you want to apply
    the columns (in the case two) you want to
    apply this function to, row by row
    new names for the new columns created

    this works when you are applying a
    separate function to two columns.
    it does not work well when you are
    applying one function to multiple cols

    '''
    names = [col1, col2]
    wcs = df[names].progress_applymap(function)
    wcs.columns = [newname1, newname2]
    df = pd.concat([df, wcs], axis=1)
    return(df)



def text_to_wordlist(row, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    # text = str(row['question1']).split()
    text = str(row).split()

    text = [word.lower() for word in text]

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i m", "i am", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"usa", "America", text)
    text = re.sub(r"canada", "Canada", text)
    text = re.sub(r"japan", "Japan", text)
    text = re.sub(r"germany", "Germany", text)
    text = re.sub(r"burma", "Burma", text)
    text = re.sub(r"rohingya", "Rohingya", text)
    text = re.sub(r"zealand", "Zealand", text)
    text = re.sub(r"cambodia", "Cambodia", text)
    text = re.sub(r"zealand", "Zealand", text)
    text = re.sub(r"norway", "Norway", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"pakistan", "Pakistan", text)
    text = re.sub(r"britain", "Britain", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iphone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"iii", "3", text)
    text = re.sub(r"california", "California", text)
    text = re.sub(r"texas", "Texas", text)
    text = re.sub(r"tennessee", "Tennessee", text)
    text = re.sub(r"the us", "America", text)
    text = re.sub(r"trump", "Trump", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

# df_train['cleaned_question1'] = df_train.apply(text_to_wordlist, axis=1)

tqdm.pandas()
df_train = apply_and_add_function(df_train,
                                  text_to_wordlist,
                                  'question1',
                                  'question2',
                                  'cleaned_question1',
                                  'cleaned_question2')

tqdm.pandas()
df_test = apply_and_add_function(df_test,
                                  text_to_wordlist,
                                  'question1',
                                  'question2',
                                  'cleaned_question1',
                                  'cleaned_question2')

cachedStopWords = stopwords.words("english")



def return_words(row):
    '''this function returns 
    each word in a question, individually'''
    try:
        words = str(row).split()
        text = [word for word in words if word not in cachedStopWords]
        lower_text = [word.lower() for word in text]
        return(lower_text)
    except:
        return('.')


tqdm.pandas()
df_train = apply_and_add_function(df_train,
                                  return_words,
                                  'question1',
                                  'question2',
                                  'q1words',
                                  'q2words')


tqdm.pandas()
df_test = apply_and_add_function(df_test,
                                  return_words,
                                  'question1',
                                  'question2',
                                  'q1words',
                                  'q2words')

print(df_train.head())
