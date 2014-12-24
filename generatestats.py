from Corpusobject import Corpus
from VariantObjects import Instance, Container, Variant, Type, Text
from itertools import product


#Make the initial Corpus from BNC_orig to grab the data item.

b0 = Corpus(filename='corpora/bnc', windowsize=0, bigramweight=1,
    posweight=1, include_JWD=False, include_bigrams=False, curr_POS_weight=1, use_pos=False, bigram_norm=False)

b0_data = b0.data

#Feed this into a Text and extract the variant output

v0 = Text(b0_data)

v0.cause_variation('r', no_output=True, number=1220)

#print(v0.variation_info)

output = v0.output()

s = v0.variation_info['selection']

for i in s:
    for v in i:
        print(i.ID, v.name)

s2 = []
for i in s:
    for v in i:
        s2.append(v.name)

s
s2


def make_corpus_list():
    settings_values = [i for i in product([output], [0,1], [1], [1], [True, False], [True, False], [1], [False], [True, False])]

    corpus_list = []

    for config in settings_values:
        corpus_list.append(Corpus(
        filename=config[0],
        windowsize=config[1],
        bigramweight=config[2],
        posweight=config[3],
        include_JWD=config[4],
        include_bigrams=config[5],
        curr_POS_weight=config[6],
        use_pos=config[7],
        bigram_norm=config[8]))#There surely has to be a better way to do this...

    return corpus_list

#Want to use output from above as the source for Corpus object



corpus_list = make_corpus_list()

kmresults = []
for corp in corpus_list:
    kmresults.append(corp.do_KM(letter=s2, k=1220, dump=False))


b0.predicted_results




#Now what do I do with the kmresults....


def compare(expected_list, CorpusObject):
    count = 0
    over1 = 0
    countover1 = 0

    for i in expected_list:
        if len(i) > 1:
            over1 += 1
    for i in CorpusObject.predicted_clusters:
        if i in expected_list and len(i) > 1:
            countover1 += 1
        if i in expected_list:
            count += 1

    stats = []
    stats.append('{} out of {} are perfect matches, in total'.format(count, len(expected_list)))
    stats.append('{} out of {} clusters in the original are > 1'.format(over1, len(expected_list)))
    stats.append('{} out of {} perfect clusters are >1'.format(countover1, count))
    return stats



for c in corpus_list:
    print(compare(v0.expected_clustering, c))





