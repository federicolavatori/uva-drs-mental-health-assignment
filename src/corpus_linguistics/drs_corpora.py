"""

drs_corpora

Digital Research Seminar - Group-making & digital culture
Author: Bharath Ganesh, b.ganesh@uva.nl, University of Amsterdam
Week 3: Corpus Linguistics

This is a package of helper functions used in the Week 3 Corpus Linguistics Notebook

"""

# imports

import pandas as pd
import spacy
from spacy.tokens import DocBin
from collections import Counter
import glob
import math
from scipy.stats import chi2_contingency
import re
from random import sample
from IPython.core.display import display, HTML

# spaCy preprocessing functions

DOCBIN_SIZE = None

def chunker(seq, size): 
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def doc_clean(doc, remove_stopwords=True):
    cleaned_doc = list()
    for i, token in enumerate(doc):
        if token.is_punct:
            continue
        elif remove_stopwords == True and token.is_stop:
            continue
        elif token.pos_ == "SPACE":
            continue
        else:
            cleaned_doc.append(token)
    return cleaned_doc

# frequency computation functions

def docbin_counter(docbin, nlp):
    docbin_subtotal = Counter()
    db = DocBin().from_disk(docbin)
    docs = list(db.get_docs(nlp.vocab))
    counters = [Counter([(token.lemma_, token.pos_) for token in doc_clean(doc, remove_stopwords=True)]) for doc in docs]
    for counter in counters:
        docbin_subtotal.update(counter)
    return docbin_subtotal

def fdist2table(fdist, savename):
    
    """ 
    fdist must be a dict with (word, POS) tuples as keys and int frequencies as values
    this will convert the fdist to a pandas table and save as an XLSX
    """
    
    keys = list(fdist.keys())
    records = [(key[0], key[1], fdist[key]) for key in keys]
    tb = pd.DataFrame.from_records(records, columns = ["word", "label", "count"])
    tb = tb.loc[(tb["label"] != "SPACE") & (tb["label"] != "PUNCT")] # taking out some unnecessary parts of speech
    tb.to_excel(savename, index = True, engine = 'xlsxwriter')
    print("Frequency distribution saved!")
    return tb

# collocation computation functions

def compute_log_likelihood(w1_ct, w2_ct, colloc_ct, nwords):
    
    """
    Contingency table:
    
              | w1 | ~w1 |
        | w2  | a  |  c  |
        | ~w2 | b  |  d  |
        
        where a = colloc_ct
              b = w1_ct - colloc_ct
              c = w2_ct - colloc_ct
              d = nwords - (a + b + c)
              
        See Xiao, R. (2015). Collocation. in Biber, D. and Reppen, R., Cambridge Handbook of English Corpus Linguistics.
        formula from p. 111
        
    """
    a = colloc_ct
    b = (w1_ct - a)
    c = (w2_ct - a)
    d = (nwords - (a + b + c))
    
    try:
        LL = 2 * (
            (a * math.log(a)) + 
            (b * math.log(b)) +
            (c * math.log(c)) + 
            (d * math.log(d)) -
            ((a + b) * math.log((a + b))) -
            ((a + c) * math.log((a + c))) -
            ((b + d) * math.log((b + d))) -
            ((c + d) * math.log((c + d))) +
            ((a + b + c + d) * math.log(a + b + c + d))
        )
    except ValueError:
        print("Error encountered, printing contingency table values...")
        print(" A:", a, " B:", b, " C:", c, " D:", d)
        LL = "math error"
    return LL

def docbin_collocation_collector(docbin, q, nlp, window_size = 3, remove_stopwords = True):
    
    # q must be a tuple of lemma and part of speech
    
    db = DocBin().from_disk(docbin)
    docs = list(db.get_docs(nlp.vocab))
    
    bin_collocs = Counter()
    
    for doc in docs:
        doc = [(token.lemma_, token.pos_) for token in doc_clean(doc, remove_stopwords = remove_stopwords)]
        if (q[0], q[1]) in doc:
            for i, token in enumerate(doc):
                if token == (q[0], q[1]):
                    start = i - window_size
                    if start < 0:
                        start = 0
                    end = i + window_size + 1
                    if end > len(doc) - 1:
                        end = len(doc) - 1
                    left_span_words = [cl for cl in doc[start:i]]
                    right_span_words = [cl for cl in doc[i+1:end]]
                    bin_collocs.update([(left, q) for left in left_span_words])
                    bin_collocs.update([(q, right) for right in right_span_words])
    
    return bin_collocs

def collocator_main(q, DOCBINS, nlp, total, window_size = 3, remove_stopwords = True):
    """
    NOTE: query MUST be a tuple of (lemma, part_of_speech)
    Docbins folder MUST be a valid path (as string)
	Total must be a master fdist, so make sure that frequency has already been run first
    """

    n_words = sum(total.values())

    colloc_subtotals = [docbin_collocation_collector(docbin, q, nlp, window_size = window_size, remove_stopwords = remove_stopwords) for docbin in DOCBINS]
    all_collocs = Counter()
    for subtotal in colloc_subtotals:
        all_collocs.update(subtotal)
    all_collocs = {k: v for k, v in all_collocs.items() if v >= 3} ## this is currently a hard coded frequency cutoff!!
    scored_collocs = list()
    for cl, cl_count in all_collocs.items():
        ll = compute_log_likelihood(total[cl[0]], total[cl[1]], cl_count, n_words)
        left_word = cl[0][0]
        left_pos = cl[0][1]
        right_word = cl[1][0]
        right_pos = cl[1][1]
        frequency = all_collocs[cl]
        out_data = (left_word, left_pos, right_word, right_pos, ll, frequency)
        scored_collocs.append(out_data)
    colloc_df = pd.DataFrame.from_records(scored_collocs, columns = ["left_word", "left_pos", "right_word", "right_pos", "log_likelihood", "frequency"])
    return colloc_df
    
# keyness computation functions

# first, create an index for posts from an account/author

def sliced_docbin_word_counter(docbins, df, nlp, slice_value, slice_variable, remove_stopwords = True, docbin_size = DOCBIN_SIZE):
    idx2var = [(idx, var) for idx, var in enumerate(list(df[slice_variable]))]
    fdist = Counter()
    print("Counting words in docbins...")
    for docbin_n, docbin in enumerate(docbins):
        print(docbin)
        docbin_subtotal = Counter()
        db = DocBin().from_disk(docbin)
        docs = list(db.get_docs(nlp.vocab))
        document_index = docbin_n * docbin_size
        for i, doc in enumerate(docs):
            idx = document_index + i
            if (idx, slice_value) in idx2var:
                ctr = Counter([(token.lemma_, token.pos_) for token in doc_clean(doc, remove_stopwords = remove_stopwords)])
                docbin_subtotal.update(ctr)
            else:
                continue
        fdist.update(docbin_subtotal)
    return fdist

def keyness_chi_sq(study_corpus_fdist, reference_corpus_fdist, savename):
    
    study_corpus_total_words = sum(study_corpus_fdist.values())
    reference_corpus_total_words = sum(reference_corpus_fdist.values())
    
    study_words = list(study_corpus_fdist.keys())
    
    keyness_records = list()
    
    for word_key in study_words:
        fsc = [study_corpus_fdist[word_key], study_corpus_total_words - study_corpus_fdist[word_key]]
        frc = [reference_corpus_fdist[word_key], reference_corpus_total_words - reference_corpus_fdist[word_key]]
        keyness_stats = chi2_contingency([fsc, frc])
        keyness_records.append((word_key[0], word_key[1], study_corpus_fdist[word_key], keyness_stats[0], keyness_stats[1]))
        
    outdf = pd.DataFrame.from_records(keyness_records, columns = ["word", "label", "count", "chi_sq", "p-value"])
    outdf.to_excel(savename, index = True, engine = 'xlsxwriter')
    
    return outdf
    
def keyness_pdiff(study_corpus_fdist, reference_corpus_fdist, savename):
    
    study_corpus_total_words = sum(study_corpus_fdist.values())
    reference_corpus_total_words = sum(reference_corpus_fdist.values())
    
    study_words = list(study_corpus_fdist.keys())
    
    keyness_records = list()
    
    for word_key in study_words:
        
        nf_sc = study_corpus_fdist[word_key]/study_corpus_total_words * 1000000
        nf_rc = reference_corpus_fdist[word_key]/reference_corpus_total_words * 1000000
        
        pdiff = ((nf_sc - nf_rc) * 100)/nf_rc
        
        keyness_records.append((word_key[0], word_key[1], study_corpus_fdist[word_key], pdiff))
        
    outdf = pd.DataFrame.from_records(keyness_records, columns = ["word", "label", "count", "pdiff"])
    outdf.to_excel(savename, index = True, engine = 'xlsxwriter')
    
    return outdf

# hashtag functions

def hashtag_search(text):
    pattern = "(#+[A-Za-zÀ-ÖØ-öø-ÿ0-9_(_)]{1,})"
    hits = re.findall(pattern, text)
    hits = Counter([hit[1:] for hit in hits])
    return hits

def hashtag_counter(documents, savename = "../../output_data/hashtags.xlsx"):
    ht_fdist = Counter()
    for document in documents:
        hits = hashtag_search(document)
        if len(hits) > 0:
            ht_fdist.update(hits)
    
    records = [(key, ht_fdist[key]) for key in ht_fdist.keys()]
    tb = pd.DataFrame.from_records(records, columns = ["hashtag", "count"])
    tb.to_excel(savename, index = True, engine = 'xlsxwriter')
    print("Hashtag frequency distribution saved!")
    return ht_fdist

# concordancing functions

def compute_concordance(docbin, query, window, nlp):
    db = DocBin().from_disk(docbin)
    docs = list(db.get_docs(nlp.vocab))
    hits = list()
    for doci, doc in enumerate(docs):
        for i, token in enumerate(doc):
            if (token.lemma_, token.pos_) == query:
                left_i = None
                right_i = None
                if i <= window:
                    left_i = 0
                else:
                    left_i = i - window
                if i >= len(doc) - window:
                    right_i = -1
                else:
                    right_i = i + window + 1
                left_span = " ".join([token.text for token in doc[left_i:i]])
                right_span = " ".join([token.text for token in doc[i+1:right_i]])
                hits.append((left_span, token.text, right_span, " ".join([docbin, str(doci)])))
    return hits

def generate_html_hits(hits, sample_size = 100, label = "query"):
    html_rows = list()
    outfile = "../../output_data/" + label + "_concordance.html"
    if sample_size > len(hits):
        sample_size = len(hits)
    hits = sample(hits, sample_size)
    for hit in hits:
        ls = "<td class = 'data'><p align = right>" + hit[0] + "</p></td>"
        q = "<td align = center class = 'query'><p><b>" + hit[1] + "</b></p></td>"
        rs = "<td align = left class = 'data'><p>" + hit[2] + "</p></td>"
        row = "<tr>" + ls + q + rs + "</tr>"
        html_rows.append(row)
    out_html = "<table cellspacing = 0>" + "\n".join(html_rows) + "</table>"
    meta = "<!DOCTYPE html><html lang='en'> <style> p {font-family: 'Arial'; margin: 0em; font-size: 10pt;} .label {color: black; padding: 1px; background-color:  #eaecee;} .data {padding:  1px;} .query {padding-top: 1px; padding-bottom:  1px; padding-right:  10px; padding-left:  10px} </style>"
    with open(outfile, "w") as of:
        of.write(meta)
        of.write(out_html)
    helperText = "<h4>Concordance generated for " + label + ". Click <a href = '" + outfile + "' target = '_blank'>here</a> to view.</h4>"
    display(HTML(helperText))

def flatten(t):
    
    return [item for sublist in t for item in sublist]

def concordancer(docbins, query, window, nlp, sample_size = 100, label = "myLabel"):
    
    ###########################################
    # query must be a tuple of ("word","POS") #
    ###########################################
    
    hits = [compute_concordance(docbin, query, window, nlp) for docbin in docbins]
    all_hits = flatten(hits)
    generate_html_hits(all_hits, sample_size = sample_size, label = label)
    return all_hits