#nns.py
from import_data import df_train
from import_data import df_test
import pandas as pd
from nltk import word_tokenize

# from create_features import words
# from gensim import corpora, models, similarities
import gensim

# i could just import this from import_data, right?
train_qs = pd.Series(df_train['question1'].tolist() +
                     df_train['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
w = set(words) # if i want as a set



# tokenize. might be able to do this another way
# tok = [word_tokenize(sent) for sent in train_qs]
# # create word2vec model from my words
# model = gensim.models.Word2Vec(tok, min_count=1, size = 32)
# model.most_similar('hi')

# load badass google word2vec model with 1 million words
# first had to download .gz file. this stuff is all in quora dir
mdl = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

mdl.most_similar('king') # works but takes a while




# trying to get 300 value vector for each word in my corpus. 
# how to handle when word isnt there?
# not working because of isnt in this????

word_vecs = []
for word in w:
	try:
		word_vecs.append(mdl[word])
	except KeyError:
		word_vecs.append(np.zeros(300,))


# this is a way to get every word assigned to 300 length vector



# longest sentence in df train
q1_max_question_len = df_train.q1words.map(len).max()
q2_max_question_len = df_train.q2words.map(len).max()

# widest_filter
widest_filter = 5

# folling arch from 
# https://arxiv.org/pdf/1610.08229.pdf


# df_train['q1words'][0]

sent1 = np.empty([300,1])
for word in ['hi','hello']:
	try:
		word_vec = np.reshape(mdl[word], [300,1])
	except KeyError:
		word_vec = np.zeros(300,1)
	except TypeError:
		word_vec = np.zeros(300,1)
	arr = np.concatenate((sent1, word_vec), axis=1)




sent = np.reshape(sent1, [300,1])
hi = np.reshape(mdl['hi'], [300,1])

np.concatenate((sent,hi),axis=1).shape

def create_train_matrix(row):
	sent1 = np.empty([300,])
	for word in row['q1words']:
	try:
		sent1.append(mdl[word])
	except KeyError:
		word_vecs.append(np.zeros(300,))


df_train.apply(create_train_matrix, axis = 1)
# for each word, make a vector and then combine into a matrix





# import small df train
# process it so i have q1 words col
