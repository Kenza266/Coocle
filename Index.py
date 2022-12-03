import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm

import numpy as np

class Index():
    def __init__(self, docs):
        self.docs = docs
        self.wnl = WordNetLemmatizer()
        self.port = PorterStemmer()
    
    def __init__(self, index, inverted):
        self.wnl = WordNetLemmatizer()
        self.port = PorterStemmer()
        self.index = index
        self.inverted = inverted

    def filter(self, token):
        t = self.port.stem(token)
        return self.wnl.lemmatize(t)

    def tokenize(self, regex='(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*'):
        regex = nltk.RegexpTokenizer(regex) 
        self.tokens_lists = [regex.tokenize(txt) for txt in self.docs]
        self.tokens_lists = [[self.filter(t) for t in tokens_list] for tokens_list in tqdm(self.tokens_lists)] 
        empty_words = nltk.corpus.stopwords.words('english')
        print('Removing empty words...')
        self.tokens_lists = [[token.lower() for token in tokens if token not in empty_words] for tokens in tqdm(self.tokens_lists)]
    
    def get_freq(self):
        self.frequencies = []
        for tokens in tqdm(self.tokens_lists):
            dict = {}
            for token in tokens:
                dict[token] = (dict[token] if token in dict.keys() else 0) + 1
            self.frequencies.append(dict) 

    def get_weights(self):
        max = [np.max(list(d.values())) for d in self.frequencies]
        self.weights = []
        for i in tqdm(range(len(max))):
            d = {}
            for k, v in self.frequencies[i].items():
                d[k] = round((v/max[i]) * np.log10(len(max)/len(self.get_docs_per_token(k))+1), 2)
            self.weights.append(d)
        
    def combine(self, origin):
        out = {}
        sets = set()
        for o in origin:
            sets = sets | set(o)
        frequencies = [{k: origin[i].get(k) if origin[i].get(k) else  0 for k in sets} for i in range(len(origin))]
        for freq, d in zip(frequencies, range(len(frequencies))):
            for k, v in freq.items():
                out[(k, d)] = v
        return out

    def process(self):
        print('Tokenizing...')
        self.tokenize()
        print('Getting frequencies...')
        self.get_freq()
        self.all_frequencies = self.combine(self.frequencies) 
        print('Getting weights...')
        self.get_weights()
        self.all_weights = self.combine(self.weights) 
        
    def get_index(self):
        index = {}
        for doc, (w, f) in enumerate(zip(self.weights, self.frequencies)):
            d = []
            for token in list(w.keys()):
                d.append([token, f[token], w[token]])
            index[doc] = d
        self.index = index

    def get_inverted(self):
        inverted = {}
        for doc, (w, f) in enumerate(zip(self.weights, self.frequencies)):
            for token in list(w.keys()):
                if token in inverted.keys():
                    inverted[token].append([doc, f[token], w[token]]) 
                else:
                    inverted[token] = [[doc, f[token], w[token]]]
        self.inverted = inverted
        
    def get_docs_per_token(self, token):
        f = {d: v for (t, d), v in self.all_frequencies.items() if token == t} 
        f = [i for i in list(f.values()) if i != 0]
        return f
    
    def get_docs(self, token):
        token = self.filter(token.lower())
        return self.inverted[token]

    '''def get_freq_tokens(self, query):
        tokens = [token for token in query.split()]
        all = {}
        details = {}
        for token in tokens:
            docs = self.get_docs(token)
            all.update({d[0] : for d in docs})
            sum = {k: sum[k] + v for k, v in self.get_freq_token(token).items()}
            details[token] = {d[0]: [d[1], d[2]] for d in docs}
        sum = {k: v for k, v in sorted(sum.items(), key=lambda item: item[1])[::-1]}
        return details, all'''
        
    '''def get_freq_per_token(self, token):
        token = self.lan.stem(token.lower())
        return [l[0] for l in self.inverted[token]]
        #return {d: v for (t, d), v in self.all_frequencies.items() if token == t}

    def get_weight_per_token(self, token):
        token = self.lan.stem(token.lower())
        return {d: v for (t, d), v in self.all_weights.items() if token == t}

    def get_docs(self, token):
        token = self.lan.stem(token.lower())
        return 

    def get_tokens(self, doc):
        return {t: v for (t, d), v in self.all_frequencies.items() if doc == d}'''