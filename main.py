from time import clock
from collections import defaultdict
from sklearn.feature_extraction import text
from operator import itemgetter
from math import log, log10, exp

myStopWords = text.ENGLISH_STOP_WORDS  # .union(["donald","trump","donald trump"])
ngram = 1
s = clock()

'for defaultdict'


def zeror():
    return 0


def zeror2():
    return [0, 0]


def unionVocab(fakeV, realV, words):
    """
    This function merge realV and fakeV
    :param fakeV: frequency dictionary of fake words'
    :param realV: frequency dictionary of real words'
    :param words: all words from fake and real headlines
    :return: Union of realV and fakeV {word:[fake,real]}
    """

    uni = dict()

    for word in words:
        if word not in fakeV:
            uni[word] = (0, realV[word])
        elif word not in realV:
            uni[word] = (fakeV[word], 0)
        else:
            uni[word] = (fakeV[word], realV[word])

    return uni


def probabilty(word_count_dict, countReal, countFake):
    """
    Calculation of P(word|class)
    :param word_count_dict: frequency dictionary of train dataset
    :param countReal: total counts of headlines from real headlines
    :param countFake: total counts of headlines from fake headlines
    :return: probabilty of words given class { word:[ P(word|fake), P(word|real) ] }
    """

    dc = defaultdict(zeror2)

    for word, counts in word_count_dict.items():
        dc[word][0] = counts[0] / countFake
        dc[word][1] = counts[1] / countReal

    return dc


def createDict(corpus, ngram):
    """
    Creation of words dictionary
    :param corpus: all headlines
    :param ngram:  an n-gram is a contiguous sequence of n items from a given sample of text or speech
    :return: frequency dictionary of words { word: number of headlines with word }
    """

    counter = defaultdict(zeror)

    for doc in corpus:
        docs = doc.split()
        sd = []
        for i in range(len(docs) - ngram + 1):
            sd.append(' '.join(docs[i:i + ngram]))
        '''when we determined the class of news, the important thing is a number of headlines with word, not counts of word.
         For example, the first option is one headline contain 10- times "okan",
         the second option is 10 headlines contain 1-time "okan". We have same counts "okan" that is 10
         but the important thing is how many diffirent headlines containing "okan" '''
        sd = set(sd)
        for k in sd:
            counter[k] += 1

    return counter


# ----------------------------------------TRAIN -READ FİLE-----------------------------------------------------#

corpus_fake = []
corpus_real = []
count_realNews = 0
count_fakeNews = 0

with open('../clean_fake-Train.txt') as FileObj:
    for line in FileObj:
        count_fakeNews += 1
        corpus_fake.append(line.strip())

with open('../clean_real-Train.txt') as FileObj:
    for line in FileObj:
        count_realNews += 1
        corpus_real.append(line.strip())

fakeVocab = createDict(corpus_fake, ngram)
fakeCount = sum([x for x in fakeVocab.values()])

realVocab = createDict(corpus_real, ngram)
realCount = sum([x for x in realVocab.values()])

'word:( count in fake, count in real  )'
vocabTrain = unionVocab(fakeVocab, realVocab, set(fakeVocab.keys()) | set(realVocab.keys()))
vocCount = len(vocabTrain.items())

# ---------------------------------------- TEST READ FİLE-----------------------------------------------------#

corpus_test = []
count_testNews = 0

with open('../test.csv') as FileObj:
    FileObj.readline()
    for line in FileObj:
        count_testNews += 1
        corpus_test.append(line.strip().split(','))

# ---------------------------------------- PREDICT -----------------------------------------------------#
'P(real)'
P_real = count_realNews / (count_realNews + count_fakeNews)
'P(fake)'
P_fake = 1 - P_real


def conditionalProbabilty(cr, cf, vocab, wordsCount):
    """
    Calculate word given class
     P_hat(word|class) = (count(w,c)+1) / (count(w)+|V|)
    :param cr: words count in real
    :param cf: words count in fake
    :param vocab: frequency dictionary
    :param wordsCount: words count in train
    :return: { word: (P_hat(word|fake) , P_hat(word|real)) }
    """

    cp = {}

    for key, val in vocab.items():
        cp[key] = ((val[0] + 1) / (cf + wordsCount), (val[1] + 1) / (cr + wordsCount))

    return cp


laplace = conditionalProbabilty(realCount, fakeCount, vocabTrain, vocCount)
'laplace = conditionalProbabilty(count_realNews,count_fakeNews,vocabTrain,vocCount)'


def naiveBayes(testNews, lap, pr, pf, cr, cf, cw):
    """
    Naive Bayes implementation.
    I compute the log probabilities to prevent numerical underflow when calculating multiplicative probabilities.
    :param testNews:
    :param lap: { word: (P_hat(word|fake) , P_hat(word|real)) }
    :param pr: P(real)
    :param pf: P(fake)
    :return: 'real' or 'fake' string
    """

    P_word_fake = 0
    P_word_real = 0
    for word in testNews:
        if word in lap.keys():
            P_word_fake += log10(lap[word][0])
            P_word_real += log10(lap[word][1])
        else:
            P_word_fake += log10(1 / (cf + cw))
            P_word_real += log10(1 / (cr + cw))

    P_fake_words = pf + P_word_fake
    P_real_words = pr + P_word_real

    return 'fake' if P_fake_words > P_real_words else 'real'


"""
    for word,pro in lap.items():
        if word in testNews:
            P_word_fake+=log(pro[0])
            P_word_real+=log(pro[1])
        else:
            P_word_fake+=log(1-pro[0])
            P_word_real+=log(1-pro[1])

    P_fake_words = pf * exp(P_word_fake)
    P_real_words = pr * exp(P_word_real)

    return 'fake' if P_fake_words>P_real_words else 'real' 
"""

acc = 0
"""outfile=open('okanalan.csv','w')
print('Id,Category',file=outfile)"""
for document in corpus_test:
    doc = createDict([document[0]], ngram).keys()
    pre = naiveBayes(doc, laplace, P_real, P_fake, realCount, fakeCount, vocCount)
    "print(document[0]+','+pre,file=outfile)"
    if document[1] == pre:
        acc += 1
"outfile.close()"
print("<<<<<<<<<<", acc * 100 / count_testNews, "<<<<<<<<<\n")

# ---------------------------------------- PRESENCE , ABSENCE -----------------------------------------------------#

for word, pro in vocabTrain.items():
    if pro[0] == 0:
        vocabTrain[word] = (1, vocabTrain[word][1])
    elif pro[1] == 0:
        vocabTrain[word] = (vocabTrain[word][0], 1)


def PresenceAbsence(pr, pf, fnewa, rnews, vocab):
    'P(fake|word),presence fake'
    P_fake_given_word = {}
    'P(fake|~word),absence fake'
    P_fake_given_not_word = {}
    'P(real|word),presence real'
    P_real_given_word = {}
    'P(real|~word),absence real'
    P_real_given_not_word = {}

    'P(word|class)'
    prob = probabilty(vocab, rnews, fnewa)
    'P(word)'
    p_word = {}
    'P(~word)'
    not_word = {}
    'P(~word|class)'
    not_prob = {}

    global count_realNews, count_fakeNews
    tp = count_realNews + count_fakeNews
    for key, pro in prob.items():
        _over = (pro[0] * pf) + (pro[1] * pr)
        'P(class|word)=P(word|class)*P(class)/(P(word|class1)*P(class1)+P(word|class2)*P(class2))'
        P_fake_given_word[key] = (pro[0] * pf) / _over
        P_real_given_word[key] = (pro[1] * pr) / _over
        p_word[key] = sum(pro) / tp

        not_prob[key] = [1 - pro[0], 1 - pro[1]]
        not_word[key] = 1 - p_word[key]

    for key, pro in not_prob.items():
        P_fake_given_not_word[key] = (pro[0] * pf) / not_word[key]
        P_real_given_not_word[key] = (pro[1] * pr) / not_word[key]

    return P_fake_given_word, P_fake_given_not_word, P_real_given_word, P_real_given_not_word


presence_fake, absence_fake, presence_real, absence_real = PresenceAbsence(P_real, P_fake, count_fakeNews,
                                                                           count_realNews, vocabTrain)

presence_real1 = sorted(list(presence_real.items()), key=itemgetter(1))[::-1]
absence_real1 = sorted(list(absence_real.items()), key=itemgetter(1))[::-1]
presence_fake1 = sorted(list(presence_fake.items()), key=itemgetter(1))[::-1]
absence_fake1 = sorted(list(absence_fake.items()), key=itemgetter(1))[::-1]


def pri(ls, na):
    print(na)
    for el in ls:
        print(el)


"""pri(presence_real1[:10],'\n>>presence real')
pri(absence_real1[:10],'\n>>absence real')
pri(presence_fake1[:10],'\n>>presence fake')
pri(absence_fake1[:10],'\n>>absence fake')"""


def noner(dic):
    """
    If word not in stopwords, I keep it.
    :param dic: presence or absence dictionary with stopwords
    :return: presence or absence dictionary without stopwords
    """
    dc = []
    for key in dic:
        flg = True
        for wrd in key[0].split():
            if wrd in myStopWords:
                flg = False
                break
        if flg:
            dc.append(key)
    return dc


non_presencereal = noner(presence_real1)
non_presencefake = noner(presence_fake1)

"""pri(non_presencereal[:10],'\n>>NON presence real')
pri(non_presencefake[:10],'\n>>NON presence fake')"""

print("\nExecution Time: ", clock() - s)

