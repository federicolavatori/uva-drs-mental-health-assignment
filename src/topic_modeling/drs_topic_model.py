"""

drs_topic_model

Digital Research Seminar - Group-making & digital culture (2023-2024)
Author: Bharath Ganesh, b.ganesh@uva.nl, University of Amsterdam
Week 5: Topic Modelling


"""


import spacy
from spacy.tokens import DocBin
import glob
import os
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile
import re
import pyLDAvis.gensim_models

def cleaning(doc, remove_stopwords = False):
    cleaned_line = list()
    for i, token in enumerate(doc):
        if token.ent_iob_ == "O": # first, check if token is part of an entity
            if token.is_punct: # remove punctuation
                continue
            elif remove_stopwords == True and token.is_stop: # remove stopwords, based on parameters
                continue
            elif token.pos_ == "SPACE":
                continue
            else:
                token = "|".join([token.lemma_, token.pos_]) # eg. dog|NOUN --> looks goofy but lets us put pos into w2v
                cleaned_line.append(token)
        elif token.ent_iob_ == "B": # is the token the beginning of an entity?
            entity_token = list() # collect entity component words
            entity_token.append(token.text)
            # print(i, token.text, len(doc))
            while i < len(doc) - 1 and doc[i+1].ent_iob_ == "I":
                if doc[i + 1].is_punct:
                    i += 1
                else:
                    entity_token.append(doc[i + 1].text)
                    i += 1
            entity_token_text = "_".join(entity_token) + "|" + token.ent_type_ # create a token
            cleaned_line.append(entity_token_text) # add the token which is a compound word of the entity
        elif token.ent_iob_ == "I": # here we skip over any words that are inside an entity, because we would have already covered them in the previous conditional
            continue
    return cleaned_line

def create_topic_model(corpus, dictionary, ks, project_directory = "project_topic_models"):
    lda = gensim.models.ldamulticore.LdaMulticore
    eval_data = list()
    
    for k in ks:
        
        print("Computing model, k =", k)
        
        mdl = lda(
            corpus,
            num_topics = k,
            id2word = dictionary,
            chunksize = 4000,
            workers = os.cpu_count() - 1,
            passes = 10
        )
        
        print("     Computing evaluation data...")
        
        pplx = mdl.log_perplexity(corpus)
        coherence_model = gensim.models.coherencemodel.CoherenceModel(model=mdl, corpus=corpus, dictionary=dictionary, coherence="u_mass")
        chnc = coherence_model.get_coherence()
        d = (k, pplx, chnc)
        eval_data.append(d)
        
        print("     Saving Model...")
              
        topic_dir = "./" + project_directory + "/topic_model_k_" + str(k) + "/"
        if os.path.exists(topic_dir):
            mdl_title = topic_dir + "topic_model_k_" + str(k) + ".lda_mdl"
            mdl.save(mdl_title)
        else:
            os.mkdir(topic_dir)
            mdl_title = topic_dir + "topic_model_k_" + str(k) + ".lda_mdl"
            mdl.save(mdl_title)
            
        print("     Generating topic/term display...")
        top_words_per_topic = []
        for topic_id in range(k):
            topic_words = mdl.get_topic_terms(topic_id, topn=10)
            words = [dictionary[word_id] for word_id, _ in topic_words]
            top_words_per_topic.append(words)

        tw = pd.DataFrame(top_words_per_topic).T
        tw.columns = [f"Topic_{i}" for i in range(k)]
        html_table = tw.to_html(index=False)
        outfile = topic_dir + "top_10_terms_by_topic.html"
        with open(outfile, "w") as html_file:
            html_file.write(html_table)
        
        print("     Generating interactive topic visualisation...")
        vis_of = topic_dir + "/topic_model_k_" + str(k) + "_visualization.html"
        vis = pyLDAvis.gensim_models.prepare(mdl, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, vis_of)
        
    eval_of = project_directory + "/evaluation_data.csv"
    eval_df = pd.DataFrame.from_records(eval_data, columns = ["k", "log_perplexity", "coherence_u_mass"]).to_csv(eval_of, index=False)
    
def docbin_generator(docbin, nlp, remove_stopwords = True):
    with open(docbin, "rb") as f:
        doc_bin = spacy.tokens.DocBin().from_bytes(f.read())
        for doc in doc_bin.get_docs(nlp.vocab):
            cleaned_doc = cleaning(doc, remove_stopwords = remove_stopwords)
            yield cleaned_doc
        
def create_gensim_dictionary(docbins, spacy_model = "en_core_web_sm", remove_stopwords = True, filename = "myname"):
    dictionary = Dictionary()
    filename = filename + ".lda_dict"
    nlp = spacy.load(spacy_model)
    
    for docbin in docbins:
        print("     Processing docbin: ", docbin)
        db = docbin_generator(docbin, nlp, remove_stopwords = remove_stopwords)
        dictionary.add_documents(db)
    
    dictionary.save(filename)
    
    return dictionary

def doc2bow_docbins_streamer(docbins, nlp, remove_stopwords = True):
    for docbin in docbins:
        print("     Processing docbin:", docbin)
        with open(docbin, "rb") as f:
            doc_bin = spacy.tokens.DocBin().from_bytes(f.read())
            for doc in doc_bin.get_docs(nlp.vocab):
                cleaned_doc = cleaning(doc, remove_stopwords = remove_stopwords)
                yield cleaned_doc

def generate_mmcorpus(docbins, dictionary, spacy_model = "en_core_web_sm", remove_stopwords = True, filename = "myname"):
    filename = filename + ".mm"
    nlp = spacy.load(spacy_model)
    corpus_stream = (dictionary.doc2bow(doc) for doc in doc2bow_docbins_streamer(docbins, nlp, remove_stopwords = remove_stopwords))
    MmCorpus.serialize(filename, corpus_stream)
    return MmCorpus(filename)

def generate_docbin_paths(docbin_folder, project_save_name):
    docbins = list()
    #DOCBIN_PATH_PATTERN = docbin_folder + "/" + docbin_folder[0:-1] + "_"
    DOCBIN_PATH_PATTERN = docbin_folder + "/" + project_save_name + "_docbin_"
    for i in range(0, len(list(glob.iglob("./" + docbin_folder + "/*.db"))), 1):
        db_path = DOCBIN_PATH_PATTERN + str(i) + ".db"
        docbins.append(db_path)
    return docbins