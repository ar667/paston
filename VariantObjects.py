from collections import defaultdict
from string import ascii_letters, ascii_lowercase, digits
from itertools import islice
import random
import pandas as pd
import pickle

class Instance:
    #An instance is an item in a text
    #Its position in the text is unique.
    #It has an associated form for that position
    def __init__(self, form, position, PoS):
        self.form = form
        self.position = position
        self.PoS = PoS#Store the part of speech so that original text can be reconstructed, with Variation. But
                        #don't use it for anything else.

    def get_position(self):
        return self.position
    def get_form(self):
        return self.form

    def __getitem__(self, key):
        return [self.form,self.position][key]

    def __repr__(self):
        return "{}: {} {}".format(self.__class__.__name__, self.form, self.position)

class Container:
    def __init__(self, *args):
        self.contents = []
        if len(args) > 0 and type(args[0]) == list:
            for i in args[0]:
                self.contents.append(i)
        else:
            if len(args) > 0:
                self.contents.extend(args)

    def add(self, *args):
        if type(args[0]) == list:
            for i in args[0]:
                self.contents.append(i)
        else:
            self.contents.extend(args)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self,key):
        return self.contents[key]

    def __setitem__(self, key, value):
        self.contents[key] = value

    def __repr__(self):
        if len(self.contents) == 0:
            return 'An empty container'
        else:
            return "{}: holding {} items of class {}".format(self.__class__.__name__, len(self.contents), self.contents[0].__class__.__name__)


class Variant(Container):
    def __init__(self, *args):
        super().__init__(*args)
        if len(self) > 0:
            self.name = self.contents[0].form


    def add(self, *args):
        if type(args[0]) == list:
            for i in args[0]:
                self.contents.append(i)
        else:
            self.contents.extend(args)
        if len(self) > 0:
            self.name = self.contents[0].form

    def __repr__(self):
        if len(self) == 0:
            return "An empty Variant"
        else:
            return "A {} with name {} containing {} Instances".format(self.__class__.__name__, self.name, len(self))

    def split(self, known_variants, ID):
        if len(self.contents) == 1:
            return self

        N = [i for i in known_variants[self.name]]
        print('N initially is', N)
        N.append(self.name) #The N list
        print('N is now', N)
        print(self.name)

        V = [i for i in self.contents] #The V list. Named here for reminder.

        min_cuts = 1
        print('Mincuts:', min_cuts)

        print('N is', len(N))
        print('V is', len(V))

        if len(V) >= len(N):
            if len(N) == 2:
                max_cuts = 1
                print('V >= N and N == 2, so max_cuts is', len(N))
            else:
                max_cuts = len(N) - 1#Take away 1 because cuts != labels/chunks
                print('V longer than or equal to N but N > 2. Max is set to', max_cuts)
        if len(V) < len(N):
            if len(V) == 2:
                max_cuts = 1
            else:
                max_cuts = len(V) - 2#Take away 2, not 1, because the first 0 and len are not valid cuts. DUH.
        print('V is shorter than N, so max_cuts is', max_cuts)

        print('Maxcuts:', max_cuts)

        if max_cuts == 1:
            num_cuts = 1
            print('min and max cuts were 1, so setting numcuts to 1, too')
        else:
            num_cuts = random.randint(min_cuts, max_cuts)
            print('Numcuts not equal to 1:', num_cuts)

        if num_cuts == 1 and len(V) == 2:
            cut_pos = [1]#Bloody edge cases...
            print('cutpos is [1] because numcuts is 1 and V')
        else:
            cut_pos = random.sample(range(1,len(V)), num_cuts)#YOU NEED TO SORT THIS NUMERICALLY LOL!!!!
            print('Doing sample because numcuts', num_cuts, 'with range from 1 to', len(V)-1)

        cut_pos = sorted(cut_pos)
        print('Sorted cutpos list')
        cut_pos.insert(0,0)
        print('Inserted leading 0 to cutpos')
        cut_pos.append(len(V))
        print('Appended final', len(V), 'to cutpos')
        print('Final utpos:', cut_pos)

        new_chunks = []

        for i, c in enumerate(cut_pos[0:-1]):
            new_chunks.append(V[c:cut_pos[i+1]])

        #print(N)
        selected_new_names = random.sample(N, len(new_chunks))
        #print(selected_new_names)


        for pair in zip(selected_new_names, new_chunks):
            for inst in pair[1]:
                inst.form = pair[0]

        store = []
        for x in new_chunks:
            store.append(Variant(x))
        for var in store:
            var.ownerID = ID

        return store

class Type(Container):
    #Just a Container but with
    def __init__(self, *args):
        super().__init__(*args)
        self.ID = "".join([random.choice(ascii_letters+digits) for x in range(15)])
        self.branches, self.leaves = self.get_foliage()

        for variant in self.contents:
            variant.ownerID = self.ID

    def get_foliage(self):
        return len(self), sum([len(i) for i in self.contents])

    def induce_split(self, known_variants):
        x = self.contents[0].split(known_variants, ID=self.ID)
        self.add(x)
        self.contents = self.contents[1:]

    def report_variation(self):
        return 1.0/(float(len(self)))

    def __repr__(self):
            return "{} {} ID {} branches {} leaves".format(self.__class__.__name__, self.ID, self.get_foliage()[0], self.get_foliage()[1])



class Text:
    def __init__(self, *source, known_variants='data/knownvariants2.pickle', verbose=True):
        self.known_variants = dict(pickle.load(open(known_variants,'rb')))#Don't use defaultdict, just in case...
        self.added_variation = False
        if source:

            print('Extracting Instances from source')
            self.contents = [Instance(f[0], p, f[1]) for i in source for p,f in enumerate(i)]

            print('Folding Instances into Variants')
            self.variants = defaultdict(Variant)
            for inst in self.contents:
                self.variants[inst.form].add(inst)

            self.variants = dict(self.variants)

            print('Constructing vocabulary of forms with [a-z] as first character')
            self.vocabulary = set([i.form for i in self.contents if i.form[0] in ascii_lowercase])

            print('Generating Types from Instances: splitting into in-vocab Types and junk Types')
            self.junk_types = set()
            self.alpha_types = defaultdict(list)
            for k,v in self.variants.items():
                if v.name in self.vocabulary:
                    self.alpha_types[k[0]].append(Type(v))
                else:
                    self.junk_types.add(Type(v))

            self.alpha_types = dict(self.alpha_types)

            print('Merging junk Types and in-vocab types as all_types')
            alphaset = set()
            for i in self.alpha_types.values():
                for x in i:
                    alphaset.add(x)
            self.all_types = self.junk_types.union(alphaset)



            '''
            I am only using instances whose forms start with [a-z].
            In terms of later calculating the variation stats...will this be problematic?
            I can assume that, pre and post-induction, non-[a-z] items will have a personal
            score of 1, because they have no variants added to their type and only start with
            one anyway.
            How do I measure this and should I include it in the final stats? Probably use
            the contents list.
            '''


            # self.variants = []
            # for item in self.vocabulary:
            #     temp = list(set([i for i in self.target_instances if i.form == item]))
            #     self.variants.append(Variant(temp))

            # self.types = [Type(i) for i in self.variants if i.name in self.known_variants.keys()]

        else:#Just be empty.
            self.contents = []#The Instances
            self.target_instances = []#Instances with [a-z] initia character
            self.vocabulary = []#The set of forms of Instances. The target of variation.
            self.variants = []#The Variants
            self.types = []#The Types

        self.initial_variation = self.calculate_variation()

    def __len__(self):
        return len(self.contents)

    def calculate_variation(self, alpha=True):
        #Can call this on __init__ to get a pre-induction snapshot, for comparison?
        if alpha == True:
            var_counts = [x.report_variation() for t in self.alpha_types.values() for x in t]
        else:
            var_counts = [t.report_variation() for t in self.all_types]
        return pd.DataFrame(var_counts)



    def cause_variation(self, known_variants, target):
        if self.added_variation == False:

            self.variation_info = {}

            all_targets = self.alpha_types[target]
            valid_targets = [i for i in self.alpha_types[target] if i.leaves > 1 and i.contents[0].name in known_variants]
            original_k = len(all_targets)
            targeted = len(valid_targets)

            for t in valid_targets:
                t.induce_split(known_variants)
            self.added_variation = False

            self.variation_info['original_k'] = original_k
            self.variation_info['targeted'] = targeted
            self.variation_info['Target letter'] = target

            new_k = 0
            for t in all_targets:
                new_k += t.branches

            self.variation_info['new_k'] = new_k

        else:
            print('Variation already done. Reinitialise the Text object')

    def find(self,target):
        store = []
        for t in self.all_types:
            #if t.contents[0].name.startswith(target):
            if t.contents[0].name == target:
                store.append(t)
        return store










































































