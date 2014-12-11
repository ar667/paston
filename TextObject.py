from collections import defaultdict
from string import ascii_letters, ascii_lowercase, digits
from random import choice

class Word:
    #A word is an item in a text
    #Its position in the text is unique.
    #The orthographic form of a word is its name
    #A word is a single item, distinguished by name and position
    def __init__(self, form, position):
        self.form = form
        self.position = position

    def get_position(self):
        return self.position
    def get_word_form(self):
        return self.form
    def get_data(self):
        return (self.form, self.position)

    def __repr__(self):
        return "Label = {}, Position = {}".format(self.form, self.position)

class Variant:
    #A variant is a collection of words
    #It encodes the single shared form of those words
    # and their position in the text
    #Variants have names instead of forms, because they are
    # an abstraction over words. It makes no sense to say a
    # variant has a form, because you cannot point to a concrete
    # example of a variant. Variants exist purely as a representation
    # of a collection of words.
    # Since that collection of words is a collection based on form,
    # we can name a variant using the shared form of its constituent words.
    def __init__(self):
        self.instances = []

    def get_all_positions(self):
        return [i.get_position() for i in self.instances]
    def get_all_forms(self):
        return [i.get_word_form() for i in self.instances]
    def get_variant_form(self):
        return self.name
    def get_count(self):
        return len(self.instances)

    def add_word_to_variant(self, word_object):
        self.instances.append(word_object)

    def __iter__(self):
        for i in self.instances:
            yield i

    def __repr__(self):
        return "{}, Label = {}, Count = {}".format(type(self), self.name, self.get_count())

class Type:
    #A type is a collection of variants.
    #It encodes all the different forms of the variants
    #In modern texts, a type has only one variant
    #But in historical texts, a type may have more than one variant.
    #Recognition of types made of variants made of words allows us to
    # conceptualise written texts as mappings from T to V to W
    # and shows that modern texts are T = V and all W ∈ V
    # whilse historical texts are V ∈ T, W ∈ V
    # To avoid confusion, a word is called by its form,
    # a variant is called by its name
    # a type is called by its ID
    # ID is non-linguistic, serving only to uniquely identify collections
    # of variants
    def __init__(self):
        self.contains_variants = []

    def add_variant_to_type(self, variant_object):
        self.contains_variants.append(variant_object)

    def get_ID(self):
        return self.ID

    def total_variants_for_type(self):
        #This should be the number of variants in the type
        return len(self.contains_variants)

    def total_words_for_type(self):
        counter = 0
        for variant in self.contains_variants:
            counter += variant.get_count()
        return counter

    def get_variant_names(self):
        store = []
        for i in self.contains_variants:
            store.append([x for x in set(i.get_all_forms())])
        return store

    def __iter__(self):
        for i in self.contains_variants:
            yield i

    def __repr__(self):
        return "{} with {} variants, over {} instances".format(type(self), len(self.contains_variants), self.total_variants_for_type())


class Text:
    #A text is a collection of words
    #This object contains no information itself other than
    # an index of the words, the variants those words belong to
    # and the types those variants belong to.
    #Instantiating the Text_object from a source generates these
    # indices from that source
    def __init__(self, source):
        self.text = [i[0] for i in source]
        self.vocabulary = list(set([i for i in self.text if i[0] in ascii_letters]))

        self.words = []
        for i,x in enumerate(self.text):
            self.words.append(Word(position=i,form=x))

        self.variants = defaultdict(Variant)
        for w in self.words:
            self.variants[w.form].add_word_to_variant(w)
            self.variants[w.form].name = w.form

        self.variants = dict(self.variants)

        self.types = defaultdict(Type)
        for k,v in self.variants.items():
            temp_name = "".join([choice(ascii_letters+digits) for x in range(15)])
            self.types[temp_name].add_variant_to_type(v)
            self.types[temp_name].ID = temp_name
        self.types = dict(self.types)

        def find_type_by_variant_name(self, search_item):
            results = []
            for typeID, type_object in self.types.items():
                for variant_object in type_object:
                    if variant_object.name.startswith(search_item) == True:
                        results.append({typeID:type_object})
            return results

        def compile_stats():
            stats = {}
            for letter in ascii_lowercase:
                stats[letter] = self.find_type_by_variant_name(letter)
            return stats

        self.variants_alphabetical = compile_stats()

        def calculate_k_and_ksize():
            #This returns a thing that says what K should be, based on type count.
            #And how big each cluster should be, in terms of members
            #K size should equal V-count and T-count, in modern text.
            stats = defaultdict(list)
            ksize = defaultdict(list)

            for key,value in self.variants_alphabetical.items():
                for type_object_dict in value:
                    for type_ID, type_object in type_object_dict.items():
                        WHAT AM I DOING???????????????
                        ksize[key].append([type_object.total_variants_for_type(), type_object.total_words_for_type()])

            for key,value in ksize.items():
                stats[key].append(('Number of variants across all types', value[0]))

            for key,value in ksize.items():
                stats[key].append(('Number of words across all types', value[1]))

            #for key,value in ksize.items():
                #stats[key].append(('Average number of variants per type, equal to average cluster size', sum(value)/len(value)))

            ksize = defaultdict(list)

            for key,value in self.variants_alphabetical.items():
                for type_object_dict in value:
                    for type_ID, type_object in type_object_dict.items():
                        ksize[key].append(type_object.total_words_for_type())

            for key,value in ksize.items():
                stats[key].append(('blaaaah', sum(value)))

            return dict(stats)

        self.type_sizes_list = calculate_k_and_ksize()


    def get_text(self):
        return self.text
    def get_words(self):
        return self.words
    def get_vocab(self):
        return self.vocabulary
    def get_variants(self):
        return [i for i in self.variants.keys()]

    def find_type_by_variant_name(self, search_item):
        results = []
        for typeID, type_object in self.types.items():
            for variant_object in type_object:
                if variant_object.name.startswith(search_item) == True:
                    results.append({typeID:type_object})
        return results






    def __repr__(self):
        return "{},\nwith\t{} tokens,\n\t{} words,\n\t{} variants,\n\t{} types.".format(type(self), len(self.text), len(self.words), len(self.variants), len(self.types))


