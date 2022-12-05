import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm

import numpy as np

class Index():
    def __init__(self, docs, preprocessed=False):
        if preprocessed:
            index, inverted, queries, ground_truth = docs
            self.index = index
            self.inverted = inverted 
            self.queries = queries
            self.ground_truth = ground_truth
        else:
            self.docs = docs
        self.wnl = WordNetLemmatizer()
        self.port = PorterStemmer()        

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
                d[k] = round((v/max[i]) * np.log10(len(max)/len(self.get_docs(k, preprocessed=True))+1), 2)
            self.weights.append(d)
        
    def combine(self, origin):
        out = {}
        sets = set()
        for o in origin:
            sets = sets | set(o)
        frequencies = [{k: origin[i].get(k) if origin[i].get(k) else  0 for k in sets} for i in range(len(origin))]
        for freq, d in tqdm(zip(frequencies, range(len(frequencies)))):
            for k, v in freq.items():
                out[(k, d)] = v
        return out

    def process(self):
        print('Tokenizing...')
        self.tokenize()
        print('Getting frequencies...')
        self.get_freq()
        print('Combining...')
        self.all_frequencies = self.combine(self.frequencies) 
        self.get_inverted_f()
        print('Getting weights...')
        self.get_weights()
        print('Combining...')
        self.all_weights = self.combine(self.weights) 
        self.get_index()
        self.get_inverted()
        
    def get_index(self):
        index = {}
        for doc, (w, f) in enumerate(zip(self.weights, self.frequencies)):
            d = []
            for token in list(w.keys()):
                d.append([token, f[token], w[token]])
            index[doc] = d
        self.index = index

    def get_inverted_f(self):
        inverted = {}
        for doc, f in enumerate(self.frequencies):
            for token in list(f.keys()):
                if token in inverted.keys():
                    inverted[token].append([doc, f[token]]) 
                else:
                    inverted[token] = [[doc, f[token]]]
        self.inverted = inverted

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
    
    def get_docs(self, token, preprocessed=False):
        if not preprocessed:
            token = self.filter(token.lower())
        return self.inverted[token]

    def get_docs_query(self, query):
        tokens = [token for token in query.split()]
        all = {}
        details = {}
        for token in tokens:
            docs = self.get_docs(token)
            for d in docs:
                if d[0] not in all.keys():
                    all[d[0]] = [[d[1]], [d[2]]]
                else:
                    all[d[0]] = [[all[d[0]][0][0] + d[1]], [round(all[d[0]][1][0] + d[2], 2)]]
            details[token] = {d[0]: [d[1], d[2]] for d in docs}
        return details, all

    def scalar_prod(self, doc, query):
        result = 0
        for token in doc:
            if token in query:
                result += np.sum([l[2] for l in self.get_docs(token, preprocessed=True)])
        return result

    def cosine_measure(self, doc, query):
        w = [self.get_docs(token, preprocessed=True)[0][2] for token in doc]
        result = np.sqrt(len(query)) * np.sqrt(np.dot(w, w))
        return self.scalar_prod(doc, query) / result

    def jaccard_measure(self, doc, query):
        w = [self.get_docs(token, preprocessed=True)[0][2] for token in doc]
        result = len(query) + np.dot(w, w) - self.scalar_prod(doc, query)
        return self.scalar_prod(doc, query) / result

    def vector_search(self, max_docs=50, metric='scalar'):
        queries = np.unique(self.ground_truth['Query'])
        relevent_docs = [list(self.ground_truth[self.ground_truth['Query'] == q]['Relevent document']) for q in queries]
        predicted = {}
        if metric == 'scalar':
            metric = self.scalar_prod
        elif metric == 'cosine':
            metric = self.cosine_measure
        elif metric == 'jaccard':
            metric = self.jaccard_measure
        #max_docs = max([len(l) for l in relevent_docs])
        for q in tqdm(queries):
            pred = []
            query = self.queries[str(q)]
            for doc, tokens in self.index.items():
                pred.append([q, doc, metric([t[0] for t in tokens], query)])
            pred = sorted(pred, key=lambda x: x[2], reverse=True)
            predicted[q] = [p[1] for p in pred[:max_docs]]
        return predicted.values(), relevent_docs

    def PR(self, pred, relevent):
        precisions = []
        recalls = []
        for p, r in zip(pred, relevent):
            TP = len(set(p) & set(r))
            precisions.append(TP/len(p))
            recalls.append(TP/len(r))
        return np.mean(precisions), np.mean(recalls)