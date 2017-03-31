from process import df_train
from process import df_test
from process import apply_and_add_function

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
# from nltk.tokenize import wordpunct_tokenize
import nltk
from tqdm import tqdm
# from itertools import chain
# from sklearn.feature_extraction.text import TfidfVectorizer
# from itertools import compress
from collections import Counter
import warnings


print('imports finished')


############################
####### Feature 1. ######## 
############################
# nltk.download("stopwords") # if i need to redownload
print('finding word counts')


def word_count_diff(row):
    '''
    split each string and return length
    '''
    try:
        l1=len(row['q1words'])
        l2=len(row['q2words'])
        dl = np.abs(l1 - l2)
    except:
        dl = 0
    # l = len(str(row).split())
    return dl


tqdm.pandas()
df_train['word count dif'] = df_train.progress_apply(word_count_diff, axis=1)
df_test['word count dif'] = df_test.progress_apply(word_count_diff, axis=1)


print('completed function 1 for train and test')


############################
####### Feature 2. ######## 
############################
cachedStopWords = stopwords.words("english")

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


print('completed function 3 for train and test')



###########################
###### Feature 3. ######## 
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

tqdm.pandas()
df_train['how match?'] = df_train.progress_apply(how_match, axis=1)
tqdm.pandas()
df_test['how match?'] = df_test.progress_apply(how_match, axis=1)

print('completed function 5 for train and test')

############################
####### Feature 4. ######## 
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


tqdm.pandas()
df_train['what match?'] = df_train.progress_apply(what_match, axis=1)
tqdm.pandas()
df_test['what match?'] = df_test.progress_apply(what_match, axis=1)


print('completed function 6 for train and test')

############################
####### Feature 5. ######## 
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

tqdm.pandas()
df_train['where match?'] = df_train.progress_apply(where_match, axis=1)
df_test['where match?'] = df_test.progress_apply(where_match, axis=1)


print('completed function 5 for train and test')

############################
####### Feature 6. ######## 
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

tqdm.pandas()
df_train['why match?'] = df_train.progress_apply(why_match, axis=1)
tqdm.pandas()
df_test['why match?'] = df_test.progress_apply(why_match, axis=1)

print('completed function 8 for train and test')


############################
####### Feature 7 - TF IDF ######## 
############################

# ### Use Anokas code instead of nltk...

train_qs = pd.Series(df_train['question1'].tolist() +
                     df_train['question2'].tolist()).astype(str)
# test_qs = pd.Series(df_test['question1'].tolist() +
                    # df_test['question2'].tolist()).astype(str)


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
    '''
    from anokas to find td idf score for each word
    then divide sum of shared td idf by total td idf weights
    '''
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

############################
####### Feature 8 - Number of noun difference  ######## 
############################

def shared_nouns(row):
    try:
        num_nouns_q1 = len([w for w, t in nltk.pos_tag(row['q1words']) if t[:1] in ['N']])
        num_nouns_q2 = len([w for w, t in nltk.pos_tag(row['q2words']) if t[:1] in ['N']])
        n_dif = np.abs(num_nouns_q1 - num_nouns_q2)
        total_nouns = (num_nouns_q1 + num_nouns_q2)
        frac = n_dif / total_nouns
    except:
        frac = 0
    return(frac)

tqdm.pandas()
df_train['num noun diff'] = df_train.progress_apply(shared_nouns, axis=1, raw=True)
tqdm.pandas()
df_test['num noun diff'] = df_test.progress_apply(shared_nouns, axis=1, raw=True)


############################
####### Feature 9 - Number of verb difference  ######## 
############################


def shared_verbs(row):
    try:
        num_verbs_q1 = len([w for w, t in nltk.pos_tag(row['q1words']) if t[:1] in ['V']])
        num_verbs_q2 = len([w for w, t in nltk.pos_tag(row['q2words']) if t[:1] in ['V']])
        v_dif = np.abs(num_verbs_q1 - num_verbs_q2)
        total_verbs = (num_verbs_q1 + num_verbs_q2)
        frac = v_dif / total_verbs
    except:
        frac = 0
    return(frac)


tqdm.pandas()
df_train['num verb diff'] = df_train.progress_apply(shared_verbs, 
                                                    axis=1, 
                                                    raw=True)
tqdm.pandas()
df_test['num verb diff'] = df_test.progress_apply(shared_verbs,
                                                axis=1,
                                                raw=True)


############################
####### Done ######## 
############################

print('creating features done! create features csv')

df_train.to_csv('df_train.csv', index=False)
df_test.to_csv('df_test.csv', index=False)

print(df_train.head())

# print('testcols', df_test.columns.values)
print('traincols', df_train.columns.values)

# print('testshape', df_test.shape)
print('train shape', df_train.shape)
