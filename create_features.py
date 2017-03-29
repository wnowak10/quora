
from import_data import df_train
from import_data import df_test

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import compress
from collections import Counter
import warnings


print('imports finished')

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


############################
####### Function 1. ######## 
############################

# nltk.download("stopwords") # if i need to redownload

print('finding word counts')
stops = set(stopwords.words("english"))


def word_and_punct_count(row):
    '''
    word count -- Return a count for
    the number of words and
    punctuation marks in each question.

    split the text using nltk tokenize.
    each symbol is its own 'word'...including
    . and ,'s and such
    '''
    try:
        word_and_punct_count = wordpunct_tokenize(row)
        # just return the length
        return(len(word_and_punct_count))
    except:
        pass


tqdm.pandas()
df_train = apply_and_add_function(df_train,
                                  word_and_punct_count,
                                  'question1', 'question2',
                                  'q1 word count', 'q2 word count')


tqdm.pandas()
df_test = apply_and_add_function(df_test,
                                 word_and_punct_count,
                                 'question1',
                                 'question2',
                                 'q1 word count',
                                 'q2 word count')


print('completed function 1 for train and test')

############################
####### Function 2. ######## 
############################

# cache stop words
cachedStopWords = stopwords.words("english")


def return_words(row):
    '''this function just returns each word in a question'''
    try:
        words = wordpunct_tokenize(row)
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

print('completed function 2 for train and test')


# ############################
# ####### Function 3. ######## 
# ############################


# def number_shared_words(row):
#     '''
#     written by ME
#     how many words, in total,
#     are shared btw two questions
#     '''
#     try:
#         q1 = row['q1words']
#         q2 = row['q2words']
#         num_common_words = sum(pd.Series(q1).isin(q2))
#         return(num_common_words)
#     except:
#         pass

def word_match_share(row):
    '''
    written by anokas
    '''
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in cachedStopWords:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in cachedStopWords:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes 
        # a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / \
        (len(q1words) + len(q2words))
    return R

df_train['word_match_share'] = df_train.progress_apply(word_match_share, axis=1)
df_test['word_match_share'] = df_test.progress_apply(word_match_share, axis=1)



# mine
# df_train['number shared words'] = df_train.progress_apply(number_shared_words, axis=1)
# df_test['number shared words'] = df_test.progress_apply(number_shared_words, axis=1)

print('completed function 3 for train and test')


############################
####### Function 4. ######## 
############################


# new feature -- what fraction of question
# lengths (on average) is shared btw questions
# df_train['percent shared words'] = \
#     ((2 * df_train['number shared words']) /
#      (df_train['q1 word count'] + df_train['q2 word count']))

# df_test['percent shared words'] = \
#     ((2 * df_test['number shared words']) /
#      (df_test['q1 word count'] + df_test['q2 word count']))

# print('completed feature 4 for train and test')

###########################
###### Function 5. ######## 
###########################


def how_match(row):
    '''return a 1 if the questions
    both contain word 'how', and 0 if not'''
    try:
        if sum(pd.Series(row['q1words']).isin(['how'])) >= 1 and sum(pd.Series(row['q2words']).isin(['how'])) >= 1:
            match = 1
        else:
            match = 0
        return(match)
    except:
        pass

# tqdm.pandas()
df_train['how match?'] = df_train.progress_apply(how_match, axis=1)
# tqdm.pandas()
df_test['how match?'] = df_test.progress_apply(how_match, axis=1)

print('completed function 5 for train and test')

############################
####### Function 6. ######## 
############################


def what_match(row):
    '''return a 1 if the questions both contain word how, and 0 if not'''
    try:
        if sum(pd.Series(row['q1words']).isin(['what'])) >= 1 and sum(pd.Series(row['q2words']).isin(['what'])) >=1:
            match = 1
        else:
            match = 0
        return(match)
    except:
        pass


# tqdm.pandas()
df_train['what match?'] = df_train.progress_apply(what_match, axis=1)
# tqdm.pandas()
df_test['what match?'] = df_test.progress_apply(what_match, axis=1)


print('completed function 6 for train and test')

############################
####### Function 7. ######## 
############################


def where_match(row):
    '''return a 1 if the questions both contain word how, and 0 if not'''
    try:
        if sum(pd.Series(row['q1words']).isin(['where'])) >= 1 and sum(pd.Series(row['q2words']).isin(['where'])) >=1:
            match = 1
        else:
            match = 0
        return(match)
    except:
        pass


df_train['where match?'] = df_train.progress_apply(where_match, axis=1)
df_test['where match?'] = df_test.progress_apply(where_match, axis=1)


print('completed function 7 for train and test')

############################
####### Function 8. ######## 
############################


def why_match(row):
    '''return a 1 if the questions both contain word how, and 0 if not'''
    try:
        if sum(pd.Series(row['q1words']).isin(['why'])) >= 1 and sum(pd.Series(row['q2words']).isin(['why'])) >=1:
            match = 1
        else:
            match = 0
        return(match)
    except:
        pass

df_train['why match?'] = df_train.progress_apply(why_match, axis=1)
df_test['why match?'] = df_test.progress_apply(why_match, axis=1)

print('completed function 8 for train and test')


############################
####### Function 9.0 - TF IDF for train ######## 
############################

### Use Anokas code instead of nltk...

train_qs = pd.Series(df_train['question1'].tolist() +
                     df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() +
                    df_test['question2'].tolist()).astype(str)

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

stops = stopwords.words("english")

# handle np.warning that results from division by NA 
# (i assume that is the problem)

warnings.filterwarnings("error")

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    try:
        R = np.sum(shared_weights) / np.sum(total_weights)
    except RuntimeWarning:
        R = 0
    return R

tqdm.pandas()
df_train['td idf '] = df_train.progress_apply(tfidf_word_match_share, axis=1, raw=True)
tqdm.pandas()
df_test['td idf '] = df_test.progress_apply(tfidf_word_match_share, axis=1, raw=True)






#####
## ME
####

# # use itertools to gather all words in a list
# q1words_set = list(chain.from_iterable(df_train['q1words']))
# q2words_set = list(chain.from_iterable(df_train['q2words']))
# # combine list 1 and list 2. use list, not set because we want duplicates!
# all_words = q1words_set + q2words_set
# # len(all_words)

# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(all_words)
# idf = vectorizer.idf_
# td_idfs = dict(zip(vectorizer.get_feature_names(), idf))


# def number_shared_words_ididf(row):
#     try:
#         q1 = row['q1words']
#         q2 = row['q2words']
#         common_words_index = pd.Series(q1).isin(q2)
#         shared = list(compress(q1, common_words_index))
#         summm = 0
#         for k in shared:
#             if k in td_idfs:
#                 summm += td_idfs[k]
#         return(summm)
#     except:
#         pass

# # tqdm.pandas()
# df_train['td-idf shared word score'] = df_train.progress_apply(number_shared_words_ididf, axis=1)

# print(df_train.head())
# ############################
# ####### Function 9.1 - TF IDF for test set ######## 
# ############################


# q1words_set_test = list(chain.from_iterable(df_test['q1words']))
# q2words_set_test = list(chain.from_iterable(df_test['q2words']))
# # combine list 1 and list 2. use list, not set because we want duplicates!
# all_words_test = q1words_set_test + q2words_set_test
# len(all_words_test)


# # from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer_test = TfidfVectorizer(min_df=1)
# Y = vectorizer_test.fit_transform(all_words_test)
# idf_test = vectorizer_test.idf_
# td_idfs_test = dict(zip(vectorizer_test.get_feature_names(), idf))


# def number_shared_words_ididf(row):
#     try:
#         q1 = row['q1words']
#         q2 = row['q2words']
#         common_words_index = pd.Series(q1).isin(q2)
#         shared = list(compress(q1, common_words_index))
#         summm = 0
#         for k in shared:
#             if k in td_idfs_test:
#                 summm += td_idfs_test[k]
#         return(summm)
#     except:
#         pass

# tqdm.pandas()
# df_test['td-idf shared word score'] = df_test.progress_apply(number_shared_words_ididf, axis=1)

print('creating features done!')

print('testcols', df_test.columns.values)
print('traincols', df_train.columns.values)

print('testshape', df_test.shape)
print('train shape', df_train.shape)
