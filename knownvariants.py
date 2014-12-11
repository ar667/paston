with open('data/knownvariants.csv', 'r') as file:
    x = csv.reader(file)
    kv = [i for i in x]

for line in kv:
    line[0] = line[0].lower()
    line[1] = line[1].lower()
    line[2] = int(line[2])

from collections import defaultdict
from string import ascii_letters

kv2 = []

a = set(ascii_letters)

for line in kv:
    b = set(line[0].join(line[1]))
    if b.issubset(a) == True:
        kv2.append(line)

variants = defaultdict(list)

for line in kv2:
    variants[line[1]].append(line[0])#Keys = standard form : Values = variant forms

variants_per_standard_form = defaultdict(int)

for k,v in variants.items():
    variants_per_standard_form[len(v)] += 1#Keys = variants : Values = standard forms with that many variants

#The intersection of the set of variants and BNC_words is the shared vocabulary
#that I can target for variation insertion

bwords = [i[0] for i in bnc.word_list if i[0][0] in ascii_letters]
kvwords = list(variants.keys())

kvwords = set(kvwords)
bwords = set(bwords)

inter = bwords.intersection(kvwords)#~7500 words, around 37% of BNC is available for target

#What is the distribution available for selection from?
#i.e. are they all just singletons (one variant per standard) or do I have rich selection of variants?

shared_standard_forms = defaultdict(list)

shared_standard_forms_count = defaultdict(int)







