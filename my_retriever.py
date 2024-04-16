import math

class Retrieve:
    def __init__(self,index,term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
        # Only gets dictionary of idfs if option is chosen
        if self.term_weighting == 'tfidf':
            self.idfs = self.get_idfs()

    def compute_number_of_documents(self): 
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # calculates idf for each term in the index
    def get_idfs(self):
        idfs = dict()
        for term in self.index:
            df = len(self.index[term])
            idfs[term] = math.log10(self.num_docs/df)
        return idfs

    def for_query(self, query):
        '''
        Method performing retrieval for a single query

        Returns a list of docids
        '''
        similarities = dict() # {doc_id : cosine similarity score}

        # Get new dict which contains doc_ids containing terms from query {doc_id : {term : count}}
        # terms = self.reduce_index(query) # Fuclk this line

        # Creates vsm from 'terms' dict
        vsm = self.create_vsm(query, self.index)
        q = vsm['Q']

        # for every doc create vectors then get similarity scores
        # ignores the query vector in the vsm
        for doc in vsm:
            if doc != 'Q':
                # Adds similarity scores to dict
                similarities[doc] = self.cosine_similarity(vsm[doc], q)

 
        # Sorts similarities and returns the top 10 results as doc_ids
        sorted_sim = self.sort_similarites(similarities)
        results = [tup[0] for tup in sorted_sim]
        return results
    
    # sorts the similarities by second value in the tuple in descending order
    def sort_similarites(self, similarities):
        array = list(similarities.items())
        array.sort(key=lambda x: x[1], reverse=True)
        return array

    def reduce_index(self, query):
        '''
        Creates dictionary only containing docids where a query term is present
        Calculated using a set AND/intersection operation 

        Returns terms {term : {docid : term_freq}
        '''
        terms = dict()

        terms_set = set(self.index.keys())
        query_set = set(query)
        relevant_terms = terms_set & query_set

        for t in relevant_terms:
            terms[t] = self.index[t]

        return terms
    
    # Calcuates cosine similarity for document vector (dv) and query vector (qv)
    def cosine_similarity(self, dv, qv):
        num = 0
        den = 0
        for t in dv:
            if t in qv:
                num += (dv[t] * qv[t])
            den += (dv[t]**2)

        # catches dividing by 0
        if den > 0:
            return num/math.sqrt(den)
        return 0

    def create_vsm(self, query, terms):
        '''
        Creates VSM from the 'terms' dictionary my rearranging it

        returns dict vsm {docid : {term1 : count}, {term2 : count}, ... 'Q' : {term_n : count}}
        '''
        vsm = dict() 

        # creates vsm from the reduced inverted index (terms) depending on term weight option
        for t in terms:
            for docid in terms[t]:
                term_freq = terms[t][docid]

                # Term freq is either 1 or 0
                if self.term_weighting == 'binary':
                    term_freq = 1

                # Term freq is the current term freq times terms idf
                if self.term_weighting == 'tfidf':
                    if not t in self.idfs:
                        term_freq = 0
                    else:
                        term_freq = (term_freq * self.idfs[t]) # tf.idf (tf * idf)
                
                # if tf is selected as term weight, no changes are needed for the term frequency
                # Creates entry in vsm for the docid and its terms
                if not docid in vsm:
                    vsm[docid] = {t : term_freq}
                else:
                    vsm[docid].update({t : term_freq})

        vsm['Q'] = self.get_queryVector(query)

        return vsm

    def get_queryVector(self, query):
       # calculate query vector
        queryVector = dict()
        for term in query:
            # initialse key in dict
            if not term in queryVector:
                queryVector[term] = 0

        # Change term frequencies for term weighting selected
            if self.term_weighting == 'binary':
                queryVector[term] = 1
            else:
                queryVector[term] += 1
            
        if self.term_weighting == 'tfidf':
            queryVector = self.get_tfidf(queryVector)

        return queryVector

    # calculates tfidf values for terms in a document and makes dictionary containing score for each term
    def get_tfidf(self, doc):
        tfidf = dict() # {term : tf.idf}
        for term in doc:
            # if term doesn't exist in any document tf.idf = 0
            if not term in self.idfs:
                tfidf[term] = 0
            else:
                tfidf[term] = (doc[term] * self.idfs[term])

        return tfidf
