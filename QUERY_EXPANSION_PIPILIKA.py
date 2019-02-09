import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')


import stop_words
import nltk
import six
from nltk.corpus import mac_morpho
from bm25 import BM25
import requests
import gzip
import gensim
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re

query=input('ENTER SEARCH KEY:')




############################################################################Removing_Query_Word######################################################################

def Removing_Word(not_similar,adding_final_document):

    
    split_final_document=adding_final_document.split() 

    split_not_similar=not_similar.split()
    stop=''
    length_split_final_document=len(split_final_document)
    length_split_not_similar=len(split_not_similar)

    for i in range(0,length_split_final_document):
      for j in range(0,length_split_not_similar):
          if(split_not_similar[j] in split_final_document[i]):
              stop+=split_final_document[i]
              stop+=' '
              break
    
    not_similar_final_document_=""
    not_similar_final_document=' '.join([i for i in split_final_document if i not in stop.split()])
    return not_similar_final_document










############################################################################Removing_Similar_Word######################################################################




def unique_list(text):
    ulist = []
    [ulist.append(x) for x in text if x not in ulist]
    search_key=""
    Retrieved_Stopword=Stopword_Retrieve()
    for q in ulist:
          for r in q.split():
             if r not in Retrieved_Stopword: 
                search_key+=r+" "

    
    search_key=search_key.rstrip()
    return search_key






####################################################Retrieving Similar Word for a word Using Word2Vec############################################################

model = Word2Vec.load('D:/200D_model_cbow/word2vec_model.model')

def Removing_Stemming_word(Final):
    import difflib
    Filter=Final.copy()
    Last=[]
    for i in Final:
        if i in Filter:
         Last.append(i)
         Filter.remove(i)
         for j in Filter:
            value=difflib.SequenceMatcher(None, i, j).ratio()
            if value>0.51 :
                    Filter.remove(j);

    return Last

def Retrieve_Similar_Word(word,model):
    import difflib
    word_vectors = model.wv
    not_similar_unique=""
    FINAL_OUTPUT=""
    if word in word_vectors.vocab:
        word2vec=model.wv.most_similar(word, topn=50)
        iword=''.join(str(e) for e in word2vec)                   #string covert
        clean_word2vec=''
        for i in iword:
              line = re.sub('[^ \nঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃ৅েৈ৉োৌ্ৎৗ৘ড়ঢ়য়০১২৩৪৫৬৭৮৯]', '', i[0])
              clean_word2vec+=line


        not_similar=''.join([i for i in word])+' '      
        not_similar=Removing_Word(word,clean_word2vec)
        not_similar_unique=unique_list(not_similar.split())

        
     # Removing spelling error word
        score = difflib.SequenceMatcher(None, word,not_similar_unique.split()).ratio()
        
        FINAL_OUTPUT=''.join([i for i in word])+' '
        #print(not_similar_unique)
 
        for i in not_similar_unique.split():
            value=difflib.SequenceMatcher(None, word, i).ratio()
            #print(i,'-->>>',value)
            if(value<0.51):
                FINAL_OUTPUT+=i+" "
                
    FINAL_OUTPUT=Removing_Stemming_word(FINAL_OUTPUT.split())
    FINAL_OUTPUT=' '.join(str(e) for e in FINAL_OUTPUT)
    return FINAL_OUTPUT






#######################################################################Import Stopwords From Stopword_txt_file#######################################################

def Stopword_Retrieve():
    from_filename="STOP_WORD.txt"
    from_file_read = open(from_filename, 'r', encoding="UTF-8")
    contents_read = from_file_read.read()


    stop_words = ""
    for i in contents_read.split("\n"):
        stop_words+=i+" "
    stop_words=re.sub('[^ ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃ৅েৈ৉োৌ্ৎৗ৘ড়ঢ়য়০১২৩৪৫৬৭৮৯]',"",stop_words)
    stop_words_split=stop_words.split()
    ulist = []
    [ulist.append(x) for x in stop_words_split if x not in ulist]
    return ulist




#######################################################################Removing Stopwords from Search key#######################################################


search_key=""
query_len=len(query.split())
Retrieved_Stopword=Stopword_Retrieve()
Stopword_List=""
for i in Retrieved_Stopword:
    Stopword_List+=i+", "
#print("Stopword_List:\n",Stopword_List.rstrip())
if query_len>1:
 for i in query.split():
     if i not in Retrieved_Stopword: 
        search_key+=i+" "
else:
    search_key=query

INPUT=search_key



#INPUT=search_key.rstrip()

print("SEARCH KEY AFTER REMOVING STOP WORD-->>",INPUT)

#################################################################RETRIEVING_DOCUMENT_FROM_SERVER_AND_FILTERING################################################################

def RETRIEVING_DOCUMENT(input):
    url=('http://10.100.222.160:9930/solr/n_gram/select?fl=data&q=/%s.*/&rows=100&start=0' % query)
    response = requests.get(url)
    response =response.text
    response = re.sub('[^ ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃ৅েৈ৉োৌ্ৎৗ৘ড়ঢ়য়০১২৩৪৫৬৭৮৯]',"",response)
    Retrieved_Document=response.split('              ')
    Retrieved_Document=list(filter(bool, Retrieved_Document))
    FINAL=[]
    for i in Retrieved_Document:
        temp=i.lstrip()
        temp=temp.rstrip()
        FINAL.append(temp)
    return FINAL



Retrieved_Document=RETRIEVING_DOCUMENT(INPUT)
if len(INPUT.split())>1:
 for i in INPUT.split():
      Retrieved_Document+=RETRIEVING_DOCUMENT(i)


############################################################################Removing_StopWord_From_DOCUMENT################################################################


Final_Document=[]

for q in Retrieved_Document:

          search_key=""
          for i in q.split():
             if i not in Retrieved_Stopword: 
                search_key+=i+" "
                

          search_key=search_key.rstrip()                                          #Removing last space
          search_key_length=len(search_key.split())
          if search_key_length>1 and search_key not in Final_Document: 
           Final_Document.append(search_key)



temp=""
adding_final_document=""
bm25 = BM25(Final_Document[:1000])

split_final_document=""

for position, index in enumerate(bm25.ranked(query, 25)):
      temp+=Final_Document[index]
      temp=''.join(temp+" ")
      adding_final_document+=temp        # Adding sentence






not_similar_final_document=Removing_Word(INPUT,adding_final_document)
not_similar_final_term=unique_list(not_similar_final_document.split())
not_similar_final_term=re.sub('[^ \nঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃ৅েৈ৉োৌ্ৎৗ৘ড়ঢ়য়০১২৩৪৫৬৭৮৯]', '', not_similar_final_term)


################################################################################QUERY_EXPANSION#######################################################################

def resize_string(retrieve_query):
    retrieve_query_split=retrieve_query.split()
    len_retrieve_query_split=len(retrieve_query_split)
    extract_word=[]
    if len_retrieve_query_split>5:
        for i in range(0,5):
                    extract_word.append(retrieve_query_split[i])
    else:
        for i in range(0,len_retrieve_query_split):
                    extract_word.append(retrieve_query_split[i])
                
    return extract_word


expansion_query=""

expand_sentence=[]


for i in INPUT.split():
    retrieve_query=Retrieve_Similar_Word(i,model)
    retrieve_query_resize=resize_string(retrieve_query)
    expand_sentence.append(retrieve_query_resize)
    
    adding_retrieve_query=" ( "+re.sub(' ',' OR ',retrieve_query.rstrip())+" ) "
    
    if expansion_query.split():
        expansion_query+=" AND "+adding_retrieve_query
    else:
        expansion_query=adding_retrieve_query

index=0

for i in not_similar_final_term.split():

    retrieve_query=Retrieve_Similar_Word(i,model)
    if index<=1:
        retrieve_query_resize=resize_string(retrieve_query)
        expand_sentence.append(retrieve_query_resize)
        index=index+1


    
    adding_retrieve_query=" ( "+re.sub(' ',' OR ',retrieve_query.rstrip())+" ) "

    if expansion_query.split():
        expansion_query+=" OR "+adding_retrieve_query


    else:
        expansion_query+=adding_retrieve_query

        
print("FINAL_QUERY_EXPANSION---->>",expansion_query)

x=""


print("EXPANDED SENTENCE:\n")
for i in expand_sentence[0]:
    
    for j in expand_sentence[1]:
        
        for k in expand_sentence[2]:
            if len(expand_sentence)>3:
               
                 for l in expand_sentence[3]:
                     if len(expand_sentence)>4:

                         
                        for m in expand_sentence[4]:
                            if len(expand_sentence)>5:

                                
                                   for n in expand_sentence[5]:
                                        if len(expand_sentence)>6:

                                           
                                               for o in expand_sentence[5]:
                                                   if len(expand_sentence)>7:

                                                       
                                                          for p in expand_sentence[5]:
                                                              if len(expand_sentence)>8:

                                                                  
                                                                      for q in expand_sentence[5]:
                                                                          if len(expand_sentence)>9:

                                                                              
                                                                                for r in expand_sentence[5]:
                                                                                    if len(expand_sentence)>10:

                                                                              
                                                                                           for s in expand_sentence[5]:
                                                                                               x=i+" "+j+" "+k +" "+l+" "+m+" "+n+" "+o+" "+p+" "+q+" "+r+" "+s
                                                                                               print(x)

                                                                                    else:
                                                                                        x=i+" "+j+" "+k +" "+l+" "+m+" "+n+" "+o+" "+p+" "+q+" "+r
                                                                                        print(x)


                                                                          else:
                                                                             x=i+" "+j+" "+k +" "+l+" "+m+" "+n+" "+o+" "+p+" "+q
                                                                             print(x) 


                                                              else:
                                                                  x=i+" "+j+" "+k +" "+l+" "+m+" "+n+" "+o+" "+p
                                                                  print(x)

                                                   else:
                                                        x=i+" "+j+" "+k +" "+l+" "+m+" "+n+" "+o
                                                        print(x)



                                        else:
                                            x=i+" "+j+" "+k +" "+l+" "+m+" "+n
                                            print(x)


                                       

                            else:           
                                x=i+" "+j+" "+k +" "+l+" "+m
                                print(x)

                     else:
                         x=i+" "+j+" "+k +" "+l
                         print(x)
            else:
                x=i+" "+j+" "+k
                print(x)


