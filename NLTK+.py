
# coding: utf-8

# ### Install NLTK

# In[1]:


# pip install

get_ipython().system('pip install nltk')


# In[2]:


# import and download

import nltk
nltk.download()


# ### Tokenization
# Chopping off a given document into small pieces is known as tokenization.
# 
# These small pieces of text are called as tokens. 
# 
# Sentence tokenization chops a document or article into sentences.
# 
# word tokenization chops a document or article down to words.

# In[3]:


# importing tokenizers

from nltk import sent_tokenize, word_tokenize


# In[4]:


# using sentence tokenizer

example_text = "This is python class. I like python.  I am a student. This is last line. "
sent = sent_tokenize(example_text)
print(sent)


# In[5]:


# word tokenizer

words = word_tokenize(example_text)
print(words)


# In[6]:


#using .split() function

example_words = example_text.split() 
print (example_words)


# ## Text Preprocessing
# 
# Text is the most unstructured form of all the available data, various types of noise are present in it and the data is not readily analyzable without any pre-processing.
# 
# There are three major ways of text preprocessing. 
# 1. Noise reduction
# 2. Lexicon normalization
# 3. Object standarization 
# 
# #### Noise reduction:
# 
# Any piece of text which is not relevant to the context of the data and the end-output can be specified as the noise.
# 
# Ex: 'is' , 'or' , 'and' , 'the'. 
# 
# These words can also be called as stopwords

# ##### Method 1 :  stopwords removal

# In[2]:


#importing stopwords from nltk.corpus

from nltk.corpus import stopwords


# In[4]:


#making english stopwords into a set

stop_words = set(stopwords.words("English"))

print(stop_words)


# In[11]:


# defining a function to remove stopwords

def noiseless_text(input_text):
    
    words = word_tokenize(input_text)                     #splitting sentence into words
    noiseless_text =[]                                    # making a empty list
    
    for w in words:                                       # FOR loop to remove words from above list
         if w not in stop_words:
                noiseless_text.append(w)                  # appending remaining words into list
    return print(noiseless_text)


# In[12]:


noiseless_text("India fought during second world war. India sent soliders and supplies into war")


# #### Method 2 : Noise removal

# In[7]:


#making a list of meaningless words

noise_text = ["is","a","for", "that", "this" , "it" , "of" , "to"]

# creating a function to remove noise

def remove_noise(input_text):
    
    words = input_text.split()                                                #splitting the sentence into words
    noise_free_words = [w for w in words if w not in noise_text]              # FOR loop to remove words in above list
    noise_free_text = " ".join(noise_free_words)                              # joining those words
    return noise_free_text


# In[8]:


remove_noise(" this is a cricket bat. give this bat to Virat Kohli")


# ## Lexicon normalization
# 
# Another type of textual noise is about the multiple representations exhibited by single word.
# 
# Example :  write, wrote, writing, writer, written 
# 
# The most common lexicon normalization practices are :
# 
# #### Stemming:  
# 
# Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
# 

# In[13]:


# lower casing the words

example = "Automation automatic automated automotive"

example_lower = example.lower().split()
print(example_lower)



# In[14]:


#stemming
#import stemmer

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# In[15]:


#FOR loop for using Porter Stemmer

for word in example_lower:
    stemmed_word = ps.stem(word)
    print(stemmed_word)


# 
# #### Lemmatization:
# 
# Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words)

# In[16]:


#import lemmatizer 

from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()


# In[17]:


example2 = ["maker", "called" , "ears", "loving"]


# In[19]:


#FOR loop for using lemmatizer

for word in example2:
    lemmatized_word =  lem.lemmatize(word)
    print(lemmatized_word)


# In[22]:


#lemmatizer with POS as 'adjective' example

ex_3 =  ["fast", "faster","fastest"]
for word in ex_3:
    lemmatized_word =  lem.lemmatize(word, pos = 'a')
    print(lemmatized_word)


# ### Object Standardization:
# 
# Text data often contains words or phrases which are not present in any standard dictionaries
# 
# Ex: 
#     'awsm' means awesome.
#     'omg' means 'oh my god'
#     

# In[1]:


# making a replacement dictionary

replace_dict = {'awsm':'awesome', 'luv': 'love', 'thnq':'Thank you'}


# In[2]:


# writing a function to replace words with new words

def replace_words(input_text):
    words = input_text.split()                  #splitting the text into words
    new_words =[]                               #making empty list for new words
    for w in words:                             # FOR loop for looking words in replace dict
        if w.lower() in replace_dict:   
            w = replace_dict[w.lower()]         #replacing them with new words
        new_words.append(w)                     # appending words into new words list
        new_text = ' '.join(new_words)          # joining words into a sentence again
    return new_text


# In[3]:


#example for object standarization:

replace_words("This is awsm. I luv it . Thnq .")


# ## Feature Engineering on text data
# 
# ### Syntactic Parsing
# 
# Syntactical parsing invol ves the analysis of words in the sentence for grammar and their arrangement in a manner that shows the relationships among the words.
# 
# 1. Dependency trees
# 2. Parts of speech tagging
# 
# #### Parts of speech tagging (POS tags):
# 
# Every word in a sentence is associated with a part of speech.
# 
# The pos tags defines the usage and function of a word in the sentence.
# 
# https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/

# In[8]:


#importing POS tags:

from nltk import pos_tag, word_tokenize


# In[9]:


# example for POS tags

pos_example = "Cheetah is fastest animal on earth"
pos_words = word_tokenize(pos_example)

print (pos_tag(pos_words))


# 
# 
# 
# ### Entity Extraction/ Entity parsing
# 
# Entities are defined as the most important chunks of a sentence – noun phrases, verb phrases or both.
# 
# Three Methods: 
# 
# A. Named Entity recognition
# B. Topic modelling
# C. N-grams as features
# 
# #### Named entity recognition
# 
# The process of detecting the named entities such as person names, location names, company names etc from the text is called as Named Entity Recognition.
# 
# ex: Jack works as manager in Apple.inc in New York
# 
# 
# name: Jack ,  org: Apple , place: New York
# 
# 
# #### Topic modeling 
# 
# Topic modeling is a process of automatically identifying the topics present in a text 
# 
# Topics are defined as “a repeating pattern of co-occurring terms in a text
# 
# Latent Dirichlet Allocation (LDA) is the most popular topic modelling technique
# 
# #### N- grams as feature 
# 
# A combination of N words together are called N-Grams.
