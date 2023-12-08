"""

drs_word_embedding

Digital Research Seminar - Group-making & digital culture (2023-2024)
Author: Bharath Ganesh, b.ganesh@uva.nl, University of Amsterdam
Week 5: Word Embedding


"""
import spacy
import gensim
import gensim.downloader as api
import matplotlib.pyplot as plt
import statistics
from gensim.corpora import Dictionary
import os
import pandas as pd
import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec

def stream_mm_corpus(corpus, dictionary):
    for doc in corpus:
        yield [dictionary[idx] for idx, count in doc]
        
def corpus2embedding(project_save_name = "heads_of_state", vector_size = 100, window = 5, min_count = 1):
    
    cores = os.cpu_count() - 1
    
    print("Loading corpus...")
    
    corpus_path = project_save_name + ".mm"
    dictionary_path = project_save_name + ".lda_dict"
    corpus = MmCorpus(corpus_path)
    id2word = Dictionary.load(dictionary_path)
    
    print("Training model...")
    
    model = Word2Vec(
        sentences = list(stream_mm_corpus(corpus, id2word)), 
        vector_size = vector_size, 
        window = window, 
        min_count = min_count, 
        workers = cores
    )
    
    w2v_directory = "./" + project_save_name + "_word2vec"
    w2v_model_save = w2v_directory + "/" + project_save_name + "_model.w2v"
    if os.path.exists(w2v_directory):
        model.save(w2v_model_save)
        print("Model Saved. You can load it below, and you don't have to generate it again. It loads based on the project_save_name you declared earlier.")
    else:
        os.mkdir(w2v_directory)
        model.save(w2v_model_save)
        print("Model Saved. You can load it below, and you don't have to generate it again. It loads based on the project_save_name you declared earlier.")

# naive projection

def naive_projection(x_axis, y_axis, test_words, model, plot_size=8):

  ## check if you have the right input data
  if len(x_axis) != 2:
    print("You must only have two antonyms in your x-axis")
  elif len(y_axis) !=2:
    print("You must only have two antonyms in your y-axis")
  else:

    ## lets do the projection

    x = list() # store x values for each test word
    y = list()

    ## now, we calculate the x and y coordinates for each of our test words
    for word in test_words:
      x_val = model.distance(x_axis[0], word) - model.distance(x_axis[1], word) # x_axis[0] is the first word of your antonym pair, x_axis[1] is the second word
      y_val = model.distance(y_axis[0], word) - model.distance(y_axis[1], word) # same thing, 0,1 here to select your antonyms
      x.append(x_val) # save x/y in our lists above
      y.append(y_val)
    
    ## now we plot the x/y values we just calculated

    fig, ax = plt.subplots(figsize=(plot_size,plot_size))

    for i in range(len(x)):
      ax.scatter(x[i], y[i])
      ax.annotate(test_words[i], (x[i], y[i]))
    
    xlab = x_axis[0] + " --- " + x_axis[1]
    ylab = y_axis[0] + " --- " + y_axis[1]
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.show()

# advanced projection

def advanced_projection(x_dimensions, y_dimensions, test_words, model, plot_size=6, xlab="label", ylab = "label"):
  x = list()
  y = list()

  ## measure each word against our composite axes, eg. "engineer"
  for word in test_words:

    ## set up some lists to hold our calculations
    x_vals = list()
    y_vals = list()

    ## this is a loop that calculates the x values for each individual dimension, eg man/woman, he/she
    for dim in x_dimensions:
      xval = model.distance(dim[0], word) - model.distance(dim[1], word)
      x_vals.append(xval)
    ## repeat for y values
    for dim in y_dimensions:
      yval = model.distance(dim[0], word) - model.distance(dim[1], word)
      y_vals.append(yval)

    ## ok now we need to take the average of all the x_vals and y_vals we collected for this word
    xavg = statistics.mean(x_vals)
    yavg = statistics.mean(y_vals)

    ## now lets save this to our x and y lists that we set up above (outside the for word in test_words loop) so that we can plot the word
    x.append(xavg)
    y.append(yavg)
  
  ## OK time to plot!
  fig, ax = plt.subplots(figsize=(plot_size,plot_size))

  for i in range(len(x)):
    ax.scatter(x[i], y[i])
    ax.annotate(test_words[i], (x[i], y[i]))

  plt.xlabel(xlab)
  plt.ylabel(ylab)

  plt.show()