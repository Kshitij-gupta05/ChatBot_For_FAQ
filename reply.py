import numpy as np
import nltk
import string
import random

f=open("C:\\Users\\HP\\Desktop\\chatbot1.txt","r",errors='Ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens=nltk.sent_tokenize(raw_doc)
word_tokens=nltk.word_tokenize(raw_doc)

lemmer=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dist=dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dist)))

GREET_INPUTS=('hello','hi','hey','greetings','namaste',)
GREET_RESPONSE=["hi","hey","nods","hello","namaste",]
def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSE)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#bot response
def response(user_response):
    robo1_response=''
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalize, stop_words={'english'})
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        robo1_response=robo1_response+"I am sorry! I don't understand you"
        return robo1_response
    else:
        robo1_response=robo1_response+sent_tokens[idx]
        return robo1_response

flag=True
print("BOT:My name is Bot. Lets Have a conversation.want to exit ,just say bye")
while(flag==True):
     print("You:",end="")
     user_response=input()
     user_response=user_response.lower()
     if(user_response!='bye'):
         if(user_response=='thanks' or user_response=='thank you'):
             flag=False
             print("BOT: You are welcome")
         else:
            if(greet(user_response)!=None):
                  print("BOT: ",greet(user_response))
            else:
                  sent_tokens.append(user_response)
                  word_tokens=word_tokens+nltk.word_tokenize(user_response)
                  final_words=list(set(word_tokens))
                  print("BOT:",end="")
                  print(response(user_response))
                  sent_tokens.remove(user_response)
else:
    flag=False
    print("BOT:GOODbye")




