from nltk.tag import str2tuple
from collections import Counter, namedtuple, defaultdict
from sklearn.feature_extraction import DictVectorizer
from copy import deepcopy
from jellyfish import jaro_winkler as jwd
import pickle
from operator import itemgetter
import os
from itertools import product
import csv
from sklearn.cluster import KMeans as KM
from datetime import datetime as DT

'''
paston1 = Corpus('corpora/paston', windowsize=0, bigramweight=1,
    posweight=0, include_JWD=False, include_bigrams=True, curr_POS_weight=0)

paston1 = Corpus('corpora/paston', windowsize=1, bigramweight=1,
    posweight=1, include_JWD=True, include_bigrams=True, curr_POS_weight=1.5)

bncCorpus = Corpus('corpora/bnc', windowsize=1, bigramweight=1,
    posweight=1, include_JWD=False, include_bigrams=False, curr_POS_weight=1)
'''

class Corpus:
    def __init__(self, filename, windowsize=1, bigramweight=1, posweight=1,
        include_JWD=True, include_bigrams=True, curr_POS_weight=1):
        #example junk: junk={'FW', '.', ',', "'", '"'}
        self.windowsize = windowsize
        self.bigramweight = bigramweight
        self.posweight = posweight
        self.include_JWD = include_JWD
        self.include_bigrams = include_bigrams
        self.curr_POS_weight = curr_POS_weight

        def process_paston_data():
            with open(filename) as file:
                raw_text = file.read()#load raw file
            letters_temp = raw_text.split('<Q')#split into letters based on <Q text
            letters_temp.pop(0) #remove the first item, which is blank
            letters = ["<Q"+i for i in letters_temp]#put the <Q back on
            letters2 = [i.splitlines() for i in letters]#split into lines
            letters3 = [i[8::6] for i in letters2]#select the lines which correspond to letters
            for letter in letters3:
                for index, sentence in enumerate(letter):
                    letter[index] = sentence.split()#splits each sentence into word_POS chunks
            for l in letters3:
                for s in l:
                    for i in range(1,windowsize+1):
                        s.append("ENDPAD+" + str(i) + "_" + 'END+' + str(i))
                        s.insert(0,"STARTPAD-" + str(i) + "_" + 'START-' + str(i))#This adds padding
            data = []
            for letter in letters3:
                for sent in letter:
                    data.append(sent) #This makes a flat list of the letters, with sentences as the items.
            data2 = []
            for i in range(0,len(data)):
                data2.append([str2tuple(x, sep="_") for x in data[i]])#This splits each "word_POS" into (word, POS)
            data3 = []
            for sent in data2:
                for pair in sent:
                    data3.append(pair)#This flattens the whole thing into a big long list of tuples

            #self.data = [pair for pair in data3]# if pair[1] not in junk]#This returns everything, removing junk things and punk
            print('Processing', filename)
            return [(x.lower(),y) for (x,y) in data3]

        def process_bnc_data():
            with open(filename, 'rb') as file:
                raw_data = pickle.load(file)
            for s in raw_data:
                for i in range(1,windowsize+1):
                    s.append(("ENDPAD+" + str(i) , 'END+' + str(i)))
                    s.insert(0,("STARTPAD-" + str(i) , 'START-' + str(i)))
            data1 = []
            for s in raw_data:
                for pair in s:
                    data1.append(pair)
            print('Processing', filename)
            return [(x.lower(),y) for (x,y) in data1]


        if filename == 'corpora/paston':
            self.data = process_paston_data()
            self.filename_nopath = filename.split('/')[1]
        if filename == 'corpora/bnc':
            self.data = process_bnc_data()
            self.filename_nopath = filename.split('/')[1]
        else:
            self.data = filename
            self.filename_nopath = "VarObjOutput"

        def count_all_words(data):
            '''
            Creates a dictionary of words in the corpus
            k = word
            v = count of that word in the corpus
            All lower case. Does not include the padding
            '''
            word_list = {}
            for pair in data:
                word_list[pair] = word_list.get(pair,0) + 1
            junk = []
            for k in word_list.keys():
                if k[0].startswith('startpad') or k[0].startswith('endpad') or k[1] == 'FW':#remove foreign words
                    junk.append(k)
            return {k:v for k,v in word_list.items() if k not in junk}

        def sweep(data, spread):
            '''
            Returns a list of dictionariies. One dictionary per position determined by windowsize.
            For windowsize of n, sweep will return a list of size 2n+1
            For each dictionary in the list:
            k = a word in the word list
            v = a dictionary, where
                k = POS tag at position determined by windowsize (w=1 means n=3 and this is equiv to the word and one on either side of it)
                v = a count for each POS tag
            '''
            windows = []
            for window in spread:
                store = {k:defaultdict(int) for k in data}
                store = {k:v for k,v in store.items() if k[0].startswith('startp') == False and k[0].startswith('endpad') == False}
                for index, pair in enumerate(data):
                    if pair[0][0:6] not in ['startp','endpad']:
                        store[pair][data[index+window][1]] += 1
                windows.append({k:dict(v) for k,v in store.items()})
            return windows

        def counts_to_probs(pos_count_list, posweight):
            '''
            Takes a list of POS counts and turns them into probability lists
            k = word from word list
            v = dict, where:
                k = POS
                v = conditional freq 0 < v <= 1
            Can apply a weighting to this. Default is 1.
            '''
            pos_count_list_COPY = [deepcopy(i) for i in pos_count_list]
            for thing in pos_count_list_COPY:
                for k,v in thing.items():
                    total = sum(n for n in v.values())
                    for k2,v2 in v.items():
                      v[k2] = posweight*round(v2/total,8)
            return pos_count_list_COPY

        self.all_word_counts = count_all_words(self.data)#create word count list
        print('Counting words')
        self.word_list = list(self.all_word_counts.keys())#create word list
        print('Creating word list')

        def generate_jwd_data(words):
            '''
            I should try rewriting this with generators and yield?
            Words = a list of words, not (word, POS) tuples!

            '''
            with open('data/moderndictionary', 'r') as file:#This is hardcoded....
                modern_dictionary = file.read()
            modern_dictionary = modern_dictionary.splitlines()
            store = defaultdict(list)
            for i, w in enumerate(words):
                print('Doing', w, ':', i+1,'/', len(words))
                store[w].append(sorted([[m, jwd(w,m)] for m in modern_dictionary], key=itemgetter(1),reverse=True)[0:5])
            return store

        def make_filename():
            name = BLAH


        curr = [0]
        prev = [-1*i for i in range(1,self.windowsize+1)]
        post = [i for i in range(1,self.windowsize+1)]
        self.distances = prev + curr + post#create window positions. Distance = how far to each side of the current word to look at
        print('Creating window span')

        curr = ['curr']
        prev = ['prev'*i for i in range(1,self.windowsize+1)]
        post = ['next'*i for i in range(1,self.windowsize+1)]
        self.tag_prefixes = prev[::-1] + curr + post #create tag prefixes for use in the combining of POS counts later
        print('Generating tag prefixes')

        self.pos_count_list = [i for i in sweep(self.data, self.distances)]# use sweep() to create the full pos count list
        print("Generating POS counts in each word's window")

        self.prob_list = counts_to_probs(self.pos_count_list, self.posweight) #convert the count list to a prob list
        print('Converting counts to conditional frequencies')

        def load_jwd_data():
            if include_JWD == True:
                if os.path.isfile('pickled/jwd_data_' + self.filename_nopath + ".pickle") == True:
                    print('Found JWD_data: loading it')
                    return  pickle.load(open('pickled/jwd_data_' + self.filename_nopath + ".pickle",'rb'))
                else:
                    print('No JWD_data found: generating (this will take one million years)')
                    jwd_stuff = generate_jwd_data([i[0] for i in self.word_list])
                    print('Pickling it.')
                    pickle.dump(jwd_stuff, open('pickled/jwd_data_' + self.filename_nopath + ".pickle", 'wb'))
                    return jwd_stuff
            else:
                print('Skipping JWD step')

        self.jwd_data = load_jwd_data()

        def padded_ngram_dict(word):
            '''
            Takes a word and returns a dictionary where:
            k = character bigram with padding (i.e. cat > $cat$ > $c, ca, at, t$)
            v = count of each bigram in the word (usually=1)
            It is a horrible mess and I am not going to rewrite it because it works.
            '''
            def bigrams_with_padding(word):
                store = []
                current_index = 0
                padded = '$' + word + '$'
                for i in range(0,len(padded)-1):
                    store.append(padded[i:current_index+2])
                    current_index += 1
                return store
            def padd_list_to_dict(list):
                bigramdict = {}
                for item in list:
                    bigramdict[item] = bigramdict.get(item, 0) + 1
                #This next bit norms the dict counts
                for k,v in bigramdict.items():
                    bigramdict[k] = round(bigramdict[k]/len(bigramdict),8)
                return bigramdict
            return padd_list_to_dict(bigrams_with_padding(word))


        # def populate_old(word, prob_list, tag_prefix):
        #     '''
        #     A single word, a single list of probabilities and a single tag prefix are used
        #     to create a single dictionary for that word, position and tag.
        #     k = POS tag combined with a positional prefix
        #     v = count of that POS tag in that position
        #     '''
        #     temp = {}
        #     for k,v in prob_list[word].items():
        #             temp[tag_prefix+k] = v
        #     return temp

        def populate(word, prob_list, tag_prefix, ind):
            '''
            This lets you assign a weight to the POS tag of the word (whilst not giving weight to neighbouring word POS)
            Which has the result of separating clusters into fairly strong POS-groups.

            A single word, a single list of probabilities and a single tag prefix are used
            to create a single dictionary for that word, position and tag.
            k = POS tag combined with a positional prefix
            v = count of that POS tag in that position
            '''
            temp = {}
            if ind == self.windowsize:
                for k,v in prob_list[word].items():
                    temp[tag_prefix+k] = v*self.curr_POS_weight
            else:
                for k,v in prob_list[word].items():
                    temp[tag_prefix+k] = v
            return temp

        self.raw_features = []
        self.labels = []

        for word in self.word_list:
            store = []
            combo = {}
            self.labels.append(word)
            for i in range(len(self.tag_prefixes)):
                temp = populate(word, self.prob_list[i], self.tag_prefixes[i], i)#YOU ADDED i HERE!!! So that you can weight the current POS feature!
                store.append(temp)
                if include_bigrams == True:
                    bg = {k:self.bigramweight*v for k,v in padded_ngram_dict(word[0]).items()}
                    store.append(bg)
            if include_JWD == True:
                for jwditem in self.jwd_data[word[0]]:
                    for pair in jwditem:
                        store.append({'JWD'+pair[0]:pair[1]})
            for d in store:
                for k,v in d.items():
                    combo[k] = v
            self.raw_features.append(combo)
        print('Creating label list')
        print('Combing feature sets')


        '''
        Use DictVectorizer to normalise the feature vectors for use in sklearn
        '''
        self.vectoriserObject = DictVectorizer(sparse=False)
        #self.vectors = self.vectoriserObject.fit_transform(self.raw_features)
        #Took this out because it creates massive vectors based on the entire data set
        #when really all I want is a subset (i.e. beginning with r) and I don't want
        #to have lots of DictVectoriser features caused by non-present words
        #Rewrite find_by_start() to make sure it does this work now
        print('Converting feature sets to vectors')

        print('All done')

    def find(self, word):
        '''
        Public function to search the label/feature lists for a specific word..
        '''
        for i,n in enumerate(self.labels):
            if n[0] == word:
                return self.raw_features[i]

    def find_by_start(self, n, vectors=True):
        '''
        Returns labels and features for words beginning with n
        n can be one char or more.
        '''
        l = []
        f = []
        for i,x in enumerate(self.labels):
            if x[0][0:len(n)] == n:
                l.append(x)
                f.append(self.raw_features[i])
        if vectors == True:
            return l, self.vectoriserObject.fit_transform(f)
        else:
            return l, f

    def dump(self, labs, vects, filename):
        fname = 'csv/' + filename + ".csv"
        with open(fname, 'w') as file:
            out = csv.writer(file, delimiter=",", quoting=csv.QUOTE_ALL)
            temp = zip(labs,vects)
            temp = sorted(temp, key=itemgetter(1))
            for i in temp:
                out.writerow(i)
        print('Wrote to', fname)

    def do_KM(self, letter, k, iter=200, dump=True, parr=-2):
        l,v = self.find_by_start(letter)
        km_obj = KM(n_clusters=k, max_iter=iter, n_jobs=parr)
        results = km_obj.fit(v)
        if dump == True:
            filename = DT.now().strftime('%d%m%y-%H%M%S') + "-" + letter.upper() + '-' + 'W' + str(self.windowsize) + str(self.include_JWD)[0] + str(self.include_bigrams)[0] + '-k' + str(k)
            self.dump(l, results.labels_, filename)
            return l, results.labels_
        else:
            return l, results.labels_

    def print_config(self):
        print("Window size       ", self.windowsize)
        print("Bigram weight     ", self.bigramweight)
        print("POS weight        ", self.posweight)
        print("CurrPOS weight    ", self.curr_POS_weight)
        print("Include_JWD       ", self.include_JWD)
        print("Include_Bigrams   ", self.include_bigrams)
        print("corpus            ", self.filename_nopath)







'''
from Corpusobject import Corpus

paston1 = Corpus('corpora/paston', windowsize=1, bigramweight=1,
    posweight=1, include_JWD=True, include_bigrams=True)

paston2 = Corpus('corpora/paston', windowsize=1, bigramweight=1,
    posweight=1, include_JWD=True, include_bigrams=False)

paston3 = Corpus('corpora/paston', windowsize=1, bigramweight=1,
    posweight=1, include_JWD=False, include_bigrams=True)

paston4 = Corpus('corpora/paston', windowsize=0, bigramweight=1,
    posweight=1, include_JWD=False, include_bigrams=False)


'''


















