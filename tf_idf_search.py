# tf-idf: relevance-ranked search 
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def load_documents(filename):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    absolute_path = os.path.join(__location__, filename)
    with open(absolute_path) as f:
        full_text = f.read()
        documents = full_text.split('</article>')
        documents = [' '.join(doc.strip().split("\n")[1:]) for doc in documents]
        return documents

def first_words_in_doc(file):
    words = " ".join(file.split()[:50])
    return words

def tf_idf_search(query, file):
    # Vectorize query string
    tfv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
    tf_matrix = tfv.fit_transform(file).T.tocsr()
    query_vec = tfv.transform([query]).tocsc()

    # Cosine similarity
    hits = np.dot(query_vec, tf_matrix) 

    # Rank hits
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]),
               reverse=True)

    # Output results
    print("Your query '{:s}' matches the following documents:".format(query))
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids[:6]):
        print("Doc #{:d} (score: {:.4f})".format(i, score))
        print(first_words_in_doc(file[doc_idx]), "\n")

def main():
    file = "enwiki-20181001-corpus.1000-articles.txt"
    file = load_documents(file)
    
    query = input("Please enter an query:")
    
    tf_idf_search(query, file)

main()