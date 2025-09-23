# Information-Retrieval
University assignment where given a query and a dataset of documents we should return the most relevant 
Relevancy of the documents was calculated using tfidf values and cosine similarity 

To Run: 
```
python IR_engine.py (options)
```
OPTIONS: 
    -h : print this help message 
    -s : use "with stoplist" configuration (default: without) 
    -p : use "with stemming" configuration (default: without) 
    -w LABEL : use weighting scheme "LABEL" (LABEL in {binary, tf, tfidf}, default: binary) 
    -o FILE : output results to file FILE 
