import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus or not query:
        raise ValueError("Corpus and query cannot be empty")

    doc_lengths = [len(doc) for doc in corpus]
    avg_doc_length = np.mean(doc_lengths)
    doc_term_counts = [Counter(doc) for doc in corpus]
    
    print(f"Document term counts: {doc_term_counts}")
    print(f"Document lengths: {doc_lengths}")
    print(f"Average document length: {avg_doc_length:.2f}")
    
    doc_freqs = Counter()
    for doc in corpus:
        # Update the document frequency counter with the unique terms in the document
        doc_freqs.update(set(doc))
    print(f"Document frequency counts: {doc_freqs}")

    scores = np.zeros(len(corpus))
    print(f"Scores: {scores}")

    N = len(corpus)
    print(f"N: {N}")

    for term in query:
        
        # "Laplace smoothing" or "add-one smoothing" 
        # Avoid division by zero in case the term doesn't appear in the corpus
        # Avoid getting zero or negative IDF values
        df = doc_freqs.get(term, 0) + 1
        
        # add logarithm for reducing the size of the value to a suitable range
        idf = np.log((N + 1) / df)
        
        print(f"IDF for term '{term}': {idf:.2f}")
        for idx, term_counts in enumerate(doc_term_counts):
            if term not in term_counts:
                continue

            tf = term_counts[term]
            print(f"TF for term '{term}': {tf}")
            # Normalize the document length Eq. 1
            doc_len_norm = 1 - b + b * (doc_lengths[idx] / avg_doc_length)
            # Calculate the BM25 score for the term Eq. 2
            term_score = (tf * (k1 + 1)) / (tf + k1 * doc_len_norm)
            # Add the term score to the total score for the document Eq. 3
            scores[idx] += idf * term_score

    return np.round(scores, 3)

corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']]
query = ['the', 'cat']

print(calculate_bm25_scores(corpus, query))

# Output:
# [0.693, 0., 0. ]

# explanation:
# The BM25 score for each document is calculated as follows:
# 
# For document 0:
# - IDF for 'the' = log((3 + 1) / (1 + 1)) = log(4 / 2) = 0.693
# - IDF for 'cat' = log((3 + 1) / (1 + 1)) = log(4 / 2) = 0.693
# - TF for 'the' in doc 0 = 1

# Eq. 1: doc_len_norm = 1 - b + b * (doc_lengths[idx] / avg_doc_length)
# Eq. 2: term_score = (tf * (k1 + 1)) / (tf + k1 * doc_len_norm)
# Eq. 3: scores[idx] += idf * term_score

# For document 1:
# - IDF for 'the' = log((3 + 1) / (1 + 1)) = log(4 / 2) = 0.693
# - IDF for 'dog' = log((3 + 1) / (1 + 1)) = log(4 / 2) = 0.693