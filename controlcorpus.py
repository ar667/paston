import pandas as PD
from collections import Counter
import string
from random import sample


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