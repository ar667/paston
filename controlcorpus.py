import pandas as PD
from collections import Counter, defaultdict
import string
from random import sample
from string import ascii_letters

Pwords = list(set(i[0] for i in paston1.word_list if i[0][0] in letts))
Mwords = list(set(i[0] for i in pastonBNC.word_list if i[0][0] in letts))
#This is the list of unique words by orthography (i.e. record appears once)



Pwords2 = list(set(i for i in paston1.word_list if i[0][0] in letts))
Mwords2 = list(set(i for i in pastonBNC.word_list if i[0][0] in letts))
#This is the list of unique words by POS (i.e. record appears twice, as N and V)

inter = set(Pwords).intersection(set(Mwords))
#This is the set of words that appear in both corpora


#Find the intersection of each word set, for each letter. So, which words do they each have in common.

inter2 = Counter(i[0] for i in inter)
#This is the count of words by initial letter, that appear in both corpora




Pcount_inter = Counter(i[0] for i in paston1.data if i[0] in inter)
Mcount_inter = Counter(i[0] for i in pastonBNC.data if i[0] in inter)
#The counts of each shared word in each corpus


temp = {}
for i in inter:
    temp[i] = {'BNC':Mcount_inter[i] ,'Paston':Pcount_inter[i]}
data = pandas.DataFrame(temp)
#A dataframe showing the counts of each shared word in each corpus

#Now I want to select a selection of these, according to whatever criteria I can think of
#1 = I want it distributed over the alphabet, so I need to divide data into alphabetic chunks

alphabetical = defaultdict(list)
for k,v in temp.items():
    alphabetical[k[0]] = ({k:v})

#2 = I think there are three types of singleton. A: appears in each text once,
#                                                B/C: appears in either text only once
#   Since this is all based on jointly shared list, only the A type will be in temp

#So, for each letter, find items which have values P1B1

singletons = defaultdict(list)
for k,v in alphabetical.items():
    for i in v:
        for k2,v2 in i.items():
            if v2['Paston'] == 1 and v2['BNC'] == 1:
                singletons[k].append({k2:v2})

#These are orthographic singletons. Which means that if they appear once in BNC, they are true singletons too.
#But in the Paston Letters, they are only true singletons if they ALSO have no variant forms.
#I know that 'riotous' in PL is definitely not a true singleton. But 'ripe' probably is, as it is alone in
#its cluster in R-W1-TT-k350.

#What about the most common shared words? Does it matter?

for k1,v1 in alphabetical.items():
    for i in v1:
        for k,v in i.items():
            sorted(v.items(), key=itemgetter(1), reverse=True)

#Just the word list: 105 total
singles = []
for k,v in singletons.items():
    for i in v:
        for k2 in i:
            singles.append(k2)

#Let's select...25 of these. These will be the words which, when put into the BNC_fake text, will be
#clustered by themselves - not with anything else.

picked_singles = rand.sample([i for i in singles if 4 < len(i) < 12], 25)

['sadler',
 'customer',
 'vigil',
 'pilgrimage',
 'grudge',
 'riotous',
 'revoke',
 'spies',
 'convenient',
 'descendant',
 'dwelling',
 'supervisor',
 'excuses',
 'sergeant',
 'endorse',
 'strokes',
 'stresses',
 'verse',
 'prayer',
 'spouse',
 'tallest',
 'appeased',
 'contented',
 'lame',
 'personable']

#Since these are hapax legomena in the BNC_orig corpus, there is no need to do anything other than identify
#them. In BNC_var, these words should always be in their own cluster, alone. Forever.

#This makes me think that for other words, I can just count words that appear 2,3,4...n times in the BNC.
#Then I can replace them with however many variants. So for words that appear twice, I can replace one with
#a variant. For count n, n-1 will be changed. So instead of that word having its own cluster with one member,
#it will have n members.

#cat cat cat > cat kat katt : k=1m=1 > k=1m=3

#Should I always do n-1 variants? Or should I randomise this? Let's randomise.

#So, do a word count on BNC_orig.

wc = pastonBNC.all_word_counts

wc2 = defaultdict(int)
for k,v in wc.items():
    wc2[k[0]] += v#word:count dict

wc3 = defaultdict(list)
for k,v in wc2.items():
    wc3[v].append(k)#count:words dict

w4 = defaultdict(list)
for k,v in wc3.items():
    if len(v) > 3:
        w4[k].append(v)#count:words for words that are longer than 3 chars

w5 = defaultdict(int)
for k,v in w4.items():
    w5[k] = len(v)#count:number of words with that count that are longer than 3 chars

w6 = defaultdict(list)
for k,v in wc3.items():
    if len([i for i in v if len(i) > 3]) != 0:
        w6[k] = [i for i in v if len(i) > 3 and i[0] in string.ascii_letters]
        #This is all words beginning with A-Z which are longer than 3
        #Select from THIS list and set the value of m













