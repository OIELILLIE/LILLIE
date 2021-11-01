#!/usr/bin/env python
# coding: utf-8

# End-to-end pipeline for Open Information triple extraction
# ======

# 
# ## Step 1 : In-place Coreference Resolution
# Note: We are using the Neuracoref(HuggingFace) implementation 

# Dependency loading 

import sys
from termcolor import colored, cprint

print(colored('Initializing Parallel Triple Extraction. Loading dependencies and dataset...', 'green'))


import pandas as pd 
import numpy as np
from tqdm import tqdm
from transformers import *

import spacy
import neuralcoref
nlp = spacy.load('en_core_web_md')
neuralcoref.add_to_pipe(nlp)

from pyclausie import ClausIE
from pyopenie import OpenIE5


from itertools import islice
import re
from allennlp.predictors.predictor import Predictor
import nltk


import argparse

#Create the parser
parser = argparse.ArgumentParser(description='Input parser')

#add the argument
parser.add_argument('-i', '--input', dest='infile', required=True,
                metavar='INPUT_FILE', help='The input file to the script.')

#parse and assign to the variable
args = parser.parse_args()
infile=args.infile

 
data = pd.read_json(infile)
data = data.transpose()
data.index.name = 'id'

print(colored('Done', 'green'))


# Merge the inner level JSON values of the abstract column in a single string
data['abstracts'] = data.apply(lambda row: (' '.join(row['abstract'].values())), axis=1)


# Joining abstract title with the body, because title usually contains useful info for triple extraction

data["corpus"] = data["title"] + ". " + data["abstracts"]

print(colored('Coreference resolution in progress...', 'green'))

# Coreference resolution 
corpus=[]

#for i in tqdm(range(120,121)):
for i in tqdm(range(len(data))):
    try:
        doc = nlp(data['corpus'].iloc[i])
        corpus.append(doc._.coref_resolved)

    except:    
        corpus.append(data['corpus'].iloc[i])
        print("Coref failed!")
        continue
    
data['coref_corpus'] = corpus
#print(corpus)
data.to_csv(r"coref_text.csv", sep= '\t')

print(colored('Done', 'green'))


# ## Step 2: Triple Extraction with 3 engines (MPI Clausie, OpenIE and AllenNLP)
# Note : The polarity detection in this demo is based on a (GloVe-based) neural model.
# 
# Don't forget to *activate the OpenIE server* at: infili@lab:~/openie/OpenIE-standalone$ java -jar openie-assembly-5.0-SNAPSHOT.jar --httpPort 9000
# 

print(colored('Triple extraction in progress...', 'green'))

data = pd.read_csv('coref_text.csv',  sep= '\t') 

cl = ClausIE.get_instance()

extractor = OpenIE5('http://localhost:9000')

#allennlp openIE predictor
predictor1 = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
#allennlp sentiment classifier
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/sst-2-basic-classifier-glove-2019.06.27.tar.gz")



subject =[]
predicate = []
object = []
paper_titles = []
paper_ids = []
sentiment = []
sent_token = 0
triple_token = 0
triple_num = []
sent_num = []
sent_rawtext = []
extractor_eng =[]


for index, row in tqdm(data.iterrows()):

    text = row['coref_corpus']
    #print(text)
    sent_text = nltk.sent_tokenize(text) 
    for sentence in sent_text:
        #print(sentence)
        #print('====')
        sent_binary = predictor.predict(sentence)['label']
        if sent_binary=='1':
            sentim='positive'
        elif sent_binary=='0':
            sentim='negative'
            
        # Clausie triple extraction
        try:                
            triples = cl.extract_triples([sentence])
            #print(triples)
            #print("=======================")
            triple_token = 0
            for triple in triples:                                        
                #print('|-', triple)
                subject.append(triple[1])
                predicate.append(triple[2])
                object.append(triple[3])
                paper_titles.append(row['title'])
                paper_ids.append(row['id'])
                sentiment.append(sentim)
                triple_num.append(triple_token)
                sent_num.append(sent_token)
                sent_rawtext.append(sentence)
                extractor_eng.append('1')

                triple_token+=1
        except:
            #print("ClausIE failed at extracting a triple from paper", row['id'])
            pass

        
        # OpenIE triple extraction
        try:
            extractions = extractor.extract(sentence)
            for extraction in extractions:
                subject.append(extraction['extraction']['arg1']['text'])
                predicate.append(extraction['extraction']['rel']['text'])
                obj_args =[]
                for j in range(len(extraction['extraction']['arg2s'])):
                    obj_args.append(extraction['extraction']['arg2s'][j]['text'])
                object.append(' '.join(obj_args))
                paper_titles.append(row['title'])
                paper_ids.append(row['id'])
                sentiment.append(sentim)
                triple_num.append(triple_token)
                sent_num.append(sent_token)
                sent_rawtext.append(sentence)
                #print(extraction)
                extractor_eng.append('2')

                triple_token+=1

        except:
            #print("OpenIE failed at extracting a triple from paper", row['id'])
            pass
        
        # AllenNLP triple extraction
        try:                
            extracted = predictor1.predict(sentence)
            for i in range(len(extracted)):
                result = (extracted['verbs'][i]['description'])
                subject.append(' '.join(re.findall(r"\[ARG0: (.*?)\]", result)))
                predicate.append(' '.join(re.findall(r"\[V: (.*?)\]", result)))
                object.append(' '.join(re.findall(r"\[ARG1:(.*?)\]", result)))
                paper_titles.append(row['title'])
                paper_ids.append(row['id'])
                sentiment.append(sentim)
                triple_num.append(triple_token)
                sent_num.append(sent_token)
                sent_rawtext.append(sentence)
                extractor_eng.append('3')

                triple_token+=1
        except:
            #print("AllenNLP failed at extracting a triple from paper", row['id'])
            pass

        sent_token+=1


extracted_triples = pd.DataFrame(
    {'title': paper_titles,
     'id': paper_ids,
     'subject': subject,
     'predicate': predicate,
     'object': object,
     'sentiment': sentiment,
     'sent_num':sent_num,
     'sent_rawtext':sent_rawtext,
     'triple_num':triple_num, 
     'engine':extractor_eng
    })


extracted_triples.to_csv('extracted_triples_learning.csv')
print(colored('Done', 'green'))



