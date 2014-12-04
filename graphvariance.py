'''
Functions to generate variance graphs for settings

Run as:

settings = [list of tuples (A,B,C)], where:
    A = a list of ints for windowsize
    B = list of boolean True/False for include_JWD setting
    C = list of boolean True/False for include_bigrams setting

ranges = list of ints, for use as K setting in kmeans

settings = [i for i in product([0,1], [True, False],[True, False])]

k_ranges = [200,300,400,500,600,700,800,900,1000,1100,1200]

results = run_tests(settings, k_ranges)

do_graph(results,settings)

-----
settings2 = [i for i in product([0,1,2,3,4], [False],[True,False])]

k_ranges2 = [100,150,200,250,300,350,400,450,500,550,600]

results2 = run_tests(settings2, k_ranges2)

do_graph(results2,settings2)
-------
settings3 = [i for i in product([0,1,2,3,4], [True,False], [False])]

k_ranges3 = [100,150,200,250,300,350,400,450,500,550,600]

results3 = run_tests(settings3, k_ranges3)

do_graph(results3,settings3)
------
settings4 = [i for i in product([0,1,2,3,4], [True], [True])]

k_ranges4 = [100,150,200,250,300,350,400,450,500,550,600]

results4 = run_tests('r', settings4, k_ranges4)

do_graph(results4,settings4)

------
settings5 = [i for i in product([0], [True, False], [True,False])]

k_ranges5 = [100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]

results5 = run_tests('r', settings5, k_ranges5)

do_graph(results5,settings5)
------
settings6 = [i for i in product([1], [True], [True], [0.5,1.0], [0.5,1.0])]

k_ranges6 = [100,150,200,250,300,350,400,450,500,550,600,650,700,750]

results6 = run_tests('r', settings6, k_ranges6)

do_graph(results6,settings6)

------
settings7 = [i for i in product([1], [True], [True], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])]

k_ranges7 = [300,600]

results7 = run_tests('r', settings7, k_ranges7)

do_graph(results7,settings7)

------

Proof case that w=0, J=F, B=F results in k=number_of_POS_tags_used=~23

settings8 = [i for i in product([0], [False], [False],[1],[1])]

k_ranges8 = [10,20,30,40,50,60]

results8 = run_tests('r', settings8, k_ranges8)

do_graph(results8,settings8)


'''


import numpy as np
from scipy.spatial.distance import cdist, pdist
from Corpusobject import Corpus
from itertools import product
from sklearn.cluster import KMeans
import pylab
from itertools import cycle


def data_for_graph(steps, vects):
    '''
    From sarguido/k-means-clustering
    '''
    k_range = steps
    print('Doing all KM models')
    k_means_var =[KMeans(n_clusters=k, n_jobs=-1).fit(vects) for k in k_range]
    print('Doing stats on them')
    centroids = [X.cluster_centers_ for X in k_means_var]
    k_euclid = [cdist(vects, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke,axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(vects)**2)/vects.shape[0]
    bss = tss - wcss
    return [k_means_var, k_range, bss/tss*100]

def run_tests(letter, corpora_settings, k_ranges):
    corps = []
    for setting in corpora_settings:
        corps.append(Corpus('corpora/paston', windowsize=setting[0], include_JWD=setting[1], include_bigrams=setting[2], posweight=setting[3], bigramweight=setting[4]))
    vects = []
    for i, corp in enumerate(corps):
        print('Getting vects', i, 'of', len(corps))
        a,b = corp.find_by_start(letter)
        vects.append([a,b])
    results = []
    for i, e in enumerate(vects):
        print('Getting data for corpus settings', i, 'of', len(vects))
        t = data_for_graph(k_ranges, e[1])
        results.append(t)
    return results

def do_graph(data, settings):
    lines = cycle(["-","--","-.",":"])
    markers = cycle(['H', '+', '*', 'o', 's', 'D', 'x', '.',','])
    marker_space = cycle([1,2,3])
    for e,i in enumerate(data):
        pylab.plot(i[1],i[2], linestyle=next(lines), marker=next(markers), markevery=next(marker_space), label=settings[e])
    pylab.xlabel('Number of clusters')
    pylab.ylabel('Percentage of variance explained')
    pylab.title('Variance Explained vs. k')
    pylab.legend(loc='best')


# ra = [results2[x][1:3] for x in [0,1]]
# rb = [results2[x][1:3] for x in [2,3]]
# rc = [results2[x][1:3] for x in [4,5]]
# rd = [results2[x][1:3] for x in [6,7]]
# re = [results2[x][1:3] for x in [8,9]]

# for i in range(1,6):
#     for A, B in zip([a,b,c,d,e], [ra,rb,rc,rd,re]):
#         figure(i)
#         pylab.plot(B[0],B[1],A,label=A)


# counter = 1
# for x in [ra,rb,rc,rd,re]:
#     for i in :
#         figure(counter)
#         pyplot(i[0],i[1])
#         counter += 1

'''

s4 = []
r4 = []
for i, x in enumerate(settings4):
    if x[1] == False:
        s4.append(x)
        r4.append(results4[i])
do_graph(r4,s4)

s4 = []
r4 = []
for i, x in enumerate(settings4):
    if x[1] == True:
        s4.append(x)
        r4.append(results4[i])

s1 = []
r1 = []
for i, x in enumerate(settings7):
    if x[3] == 1.0:
        s1.append(x)
        r1.append(results7[i])

s2 = []
r2 = []
for i, x in enumerate(settings7):
    if x[4] == 1.0:
        s2.append(x)
        r2.append(results7[i])

'''
