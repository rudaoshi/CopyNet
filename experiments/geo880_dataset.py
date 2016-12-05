# coding=utf-8
import sys
import numpy as np
import jieba as jb
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file

word2idx = dict()
wordfreq = dict()
word2idx['<eol>'] = 0
word2idx['<unk>'] = 1

segment  = True # True

# training set


at    = 2
lines = 0

import logging


VALID_GEN = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'X',  '(', ')', 'IS', '_', ' ', 'NOT'}

import itertools

jb.suggest_freq(('总', '人口'), True)
jb.suggest_freq(('总', '长度'), True)
def read_dataset(path):

    f     = open(path , 'r')

    pairs = []

    while True:

        line = f.readline()

        if not line: break

        line = line.strip().decode('utf-8')
        if not line:
            continue

        eng_question = line
        eng_logic = f.readline().strip().decode('utf-8')
        possible_chn_sent = f.readline().strip().decode('utf-8')

        if possible_chn_sent:
            raw_chn_question = possible_chn_sent
            chn_logic = f.readline().strip().decode('utf-8')
            chn_free_logic = f.readline().strip().decode('utf-8')

            if segment:
                chn_question = [w for w in jb.cut(raw_chn_question)]
            else:
                chn_question = raw_chn_question

            for w in chn_question:
                if w not in wordfreq:
                    wordfreq[w]  = 1
                else:
                    wordfreq[w] += 1

            if segment:
                chn_free_logic = [w for w in jb.cut(chn_free_logic) if w.strip()]


            i = 0
            merged_chn_free_logic = []
            while i < len(chn_free_logic):
                word = chn_free_logic[i]

                j = i + 1
                while word not in chn_question and word not in VALID_GEN and j <len(chn_free_logic):
                    word = word + chn_free_logic[j]
                    j += 1

                if word in chn_question or word in VALID_GEN:
                    merged_chn_free_logic.append(word)

                else:

                    logging.error("unknow words in logic: " + word.encode("utf8") + " in sentence : " + u" ".join(
                        chn_question).encode("utf8"))
                    logging.error("logic form is : " + u" ".join(chn_free_logic).encode("utf8"))
                    raise Exception("Invalid input ")

                i = j

            for w in chn_free_logic:
                if w not in wordfreq:
                    wordfreq[w]  = 1
                else:
                    wordfreq[w] += 1

            pair    = (chn_question, merged_chn_free_logic)
            pairs.append(pair)

    return pairs

training = read_dataset('./dataset/geo880/geo880.zh.train.txt')
testing = read_dataset('./dataset/geo880/geo880.zh.test.txt')

print len(training), len(testing)

# sort the vocabulary
wordfreq = sorted(wordfreq.items(), key=lambda a:a[1], reverse=True)
for w in wordfreq:
    word2idx[w[0]] = at
    at += 1

idx2word = {k: v for v, k in word2idx.items()}
Lmax     = len(idx2word)
print 'read dataset ok.'
print Lmax
for i in xrange(Lmax):
    print idx2word[i].encode('utf-8')

# use character-based model [on]
# use word-based model     [off]


def build_data(data):
    instance = dict(text=[], summary=[], source=[], target=[], target_c=[])
    for pair in data:
        source, target = pair
        A = [word2idx[w] for w in source]
        B = [word2idx[w] for w in target]
        # C = np.asarray([[w == l for w in source] for l in target], dtype='float32')
        C = [0 if w not in source else source.index(w) + Lmax for w in target]

        instance['text']      += [source]
        instance['summary']   += [target]
        instance['source']    += [A]
        instance['target']    += [B]
        # instance['cc_matrix'] += [C]
        instance['target_c'] += [C]

#    print instance['target'][5000]
#    print instance['target_c'][5000]
    return instance


train_set = build_data(training)
test_set  = build_data(testing)
serialize_to_file([train_set, test_set, idx2word, word2idx], './dataset/geo880/data-word-full.pkl')
