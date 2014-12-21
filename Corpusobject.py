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
p1 = Corpus('corpora/paston', windowsize=1, bigramweight=1,
    posweight=0, include_JWD=False, include_bigrams=True, curr_POS_weight=0, use_pos=False)

b1 = Corpus('corpora/bnc', windowsize=0, bigramweight=1,
    posweight=0, include_JWD=True, include_bigrams=True, curr_POS_weight=0, use_pos=False)

paston1 = Corpus('corpora/paston', windowsize=1, bigramweight=1,
    posweight=1, include_JWD=True, include_bigrams=True, curr_POS_weight=1.5)

bncCorpus = Corpus('corpora/bnc', windowsize=1, bigramweight=1,
    posweight=1, include_JWD=False, include_bigrams=False, curr_POS_weight=1)
'''

class Corpus:
    def __init__(self, filename, windowsize=1, bigramweight=1, posweight=1,
        include_JWD=True, include_bigrams=True, curr_POS_weight=1, use_pos=True):
        #example junk: junk={'FW', '.', ',', "'", '"'}
        self.windowsize = windowsize
        self.bigramweight = bigramweight
        self.posweight = posweight
        self.include_JWD = include_JWD
        self.include_bigrams = include_bigrams
        self.curr_POS_weight = curr_POS_weight

        def process_paston_data(filename):
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
                    if windowsize == 0:
                        s.append("ENDPAD_END")
                        s.insert(0,("STARTPAD_START"))
                    if windowsize > 0:
                        for i in range(1,windowsize+1):
                            s.append("ENDPAD+" + str(i) + '_END+' + str(i))
                            s.insert(0,"STARTPAD-" + str(i) + '_START-' + str(i))
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

        def process_bnc_data(filename):
            with open(filename, 'rb') as file:
                raw_data = pickle.load(file)
            for s in raw_data:
                if windowsize == 0:
                    s.append(("ENDPAD" , 'END'))
                    s.insert(0,("STARTPAD" , 'START'))
                if windowsize > 0:
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
            self.data = process_paston_data(filename)
            self.filename_nopath = filename.split('/')[1]
        if filename == 'corpora/bnc':
            self.data = process_bnc_data(filename)
            self.filename_nopath = filename.split('/')[1]
        if filename not in ['corpora/paston', 'corpora/bnc']:
            self.data = filename
            self.filename_nopath = "VarObjOutput"#FIX THIS AT SOME POINT

        def count_all_words(data):
            '''
            Creates a dictionary of words in the corpus
            k = word
            v = count of that word in the corpus
            All lower case. Does not include the padding

            Two versions. use_pos makes lexical items that are distinguished by POS.
                So (look, noun) and (look, verb) get an entry each.
            The other version combines those by orthography, into one entry.
            '''
            if use_pos == True:
                word_list = dict(Counter(data))
                for junk in [i for i in word_list if i[0].startswith('startpad') or i[0].startswith('endpad')]:
                    del word_list[junk]

            else:
                word_list = dict(Counter([i[0] for i in data]))
                for junk in [i for i in word_list if i.startswith('startpad') or i.startswith('endpad')]:
                    del word_list[junk]

                '''SHOULD I REMOVE FOREIGN WORDS?'''
                #Arguments:
                #   In Paston Letters, yes. Because they are not POS-tagged well. They are all as FW.
                #   No in modern texts, as they are POS-tagged properly? I dunno...not checked.
                #   Fuuuuuck.
            return word_list

        def sweep(distances):
            windows = []

            if use_pos == True:
                for position in distances:
                    store = {k:defaultdict(int) for k in self.data if k[0][0:6] not in ['startp','endpad']}
                    for index, pair in enumerate(self.data):
                        if pair in store:
                            store[pair][self.data[index+position][1]] += 1
                    windows.append({k:dict(v) for k,v in store.items()})
                return windows

            if use_pos == False:
                for position in distances:
                    store = {k[0]:defaultdict(int) for k in self.data if k[0][0:6] not in ['startp','endpad']}
                    for index, pair in enumerate(self.data):
                        if pair[0] in store:
                            store[pair[0]][self.data[index+position][1]] += 1
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

        if use_pos == True:
            self.vocabulary = [i[0] for i in self.word_list]
        if use_pos == False:
            self.vocabulary = self.word_list

        print('Creating word list')

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

        self.pos_count_list = [i for i in sweep(self.distances)]# use sweep() to create the full pos count list
        print("Generating POS counts in each word's window")

        self.prob_list = counts_to_probs(self.pos_count_list, self.posweight) #convert the count list to a prob list
        print('Converting counts to conditional frequencies')

        def do_jwd():
            if include_JWD == True:
                print('Loading JWD_data from pickle')
                with open('data/jwd.pickle', 'rb') as file:
                    jwd_data = pickle.load(file)
                with open('data/moderndictionary.pickle', 'rb') as file:
                    modern_dictionary = pickle.load(file)

                #So now all current JWD info is loaded, and the dictionary.
                #Check to make sure it has 100% coverage of the word_list
                #Take the symmetric difference of set of JWD_data keys and word_list
                temp = list(set(self.vocabulary).difference(set(jwd_data)))
                if len(temp) > 0:
                    print(len(temp), 'new words need to be added')
                    for i, new_word in enumerate(temp):
                        if use_pos == True:
                            print('Doing new word +pos', new_word, ":", i+1, '/', len(temp))
                            jwd_data[new_word[0]] = sorted([[m, jwd(new_word[0],m)] for m in modern_dictionary], key=itemgetter(1),reverse=True)[0:5]
                        if use_pos == False:
                            print('Doing new word -pos', new_word, ":", i+1, '/', len(temp))
                            jwd_data[new_word] = sorted([[m, jwd(new_word,m)] for m in modern_dictionary], key=itemgetter(1),reverse=True)[0:5]
                    with open('data/jwd.pickle', 'wb') as file:
                        pickle.dump(jwd_data, file)
                        print('Updating JWD pickle')

                else:
                    print('No new words needed to be added.')

                return jwd_data

            if include_JWD == False:
                print('Skipping JWD step')



        self.jwd_data = do_jwd()

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

        # self.raw_features = []
        # self.labels = []

        # for word in self.word_list:
        #     store = []
        #     combo = {}
        #     self.labels.append(word)
        #     for i in range(len(self.tag_prefixes)):
        #         temp = populate(word, self.prob_list[i], self.tag_prefixes[i], i)#YOU ADDED i HERE!!! So that you can weight the current POS feature!
        #         store.append(temp)
        #         if include_bigrams == True:
        #             bg = {k:self.bigramweight*v for k,v in padded_ngram_dict(word[0]).items()}
        #             store.append(bg)
        #     if include_JWD == True:
        #         for jwditem in self.jwd_data[word[0]]:
        #             for pair in jwditem:
        #                 store.append({'JWD'+pair[0]:pair[1]})
        #     for d in store:
        #         for k,v in d.items():
        #             combo[k] = v
        #     self.raw_features.append(combo)

        def compile_labels_and_features():
            raw_features = []
            labels = []

            if use_pos == True:

                for word in self.word_list:
                    store = []
                    combo = {}
                    labels.append(word)
                    for i in range(len(self.tag_prefixes)):
                        temp = populate(word, self.prob_list[i], self.tag_prefixes[i], i)#YOU ADDED i HERE!!! So that you can weight the current POS feature!
                        store.append(temp)
                        if include_bigrams == True:
                            bg = {k:bigramweight*v for k,v in padded_ngram_dict(word[0]).items()}
                            store.append(bg)
                    if include_JWD == True:
                        for jwditem in self.jwd_data[word[0]]:
                            store.append({'JWD'+jwditem[0]:jwditem[1]})
                    for d in store:
                        for k,v in d.items():
                            combo[k] = v
                    raw_features.append(combo)

            else:

                for word in self.word_list:
                    store = []
                    combo = {}
                    labels.append(word)
                    for i in range(len(self.tag_prefixes)):
                        temp = populate(word, self.prob_list[i], self.tag_prefixes[i], i)#YOU ADDED i HERE!!! So that you can weight the current POS feature!
                        store.append(temp)
                        if include_bigrams == True:
                            bg = {k:bigramweight*v for k,v in padded_ngram_dict(word).items()}
                            store.append(bg)
                    if include_JWD == True:
                        for jwditem in self.jwd_data[word]:
                            store.append({'JWD'+jwditem[0]:jwditem[1]})
                    for d in store:
                        for k,v in d.items():
                            combo[k] = v
                    raw_features.append(combo)

            return labels, raw_features

        print('Creating label list')
        print('Combing feature sets')
        self.labels, self.raw_features = compile_labels_and_features()

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


















