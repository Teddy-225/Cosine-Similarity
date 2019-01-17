import nltk
from nltk.corpus import stopwords
from nltk.corpus import inaugural
from nltk.stem.snowball import SnowballStemmer
import numpy as np

def pre(sam):
    tokens=nltk.word_tokenize(sam)
    words=[w.lower() for w in tokens if w.isalpha()]
    stop_words=set(stopwords.words('english'))
    fill=[w for w in words if not w in stop_words]
    sb=SnowballStemmer('english')
    snowball=[sb.stem(data) for data in fill]
    count=nltk.defaultdict(int)
    for word in snowball:
        count[word]+=1
    return(count)
def cosine(x,y):
    product=np.dot(x,y)
    mag1=np.linalg.norm(x)
    mag2=np.linalg.norm(y)
    return(product/(mag1*mag2))
def similarity(s1,s2):
    words=[]
    for key in s1:
        words.append(key)
    for key in s2:
        words.append(key)
    v1=np.zeros(len(words),dtype=int)
    v2=np.zeros(len(words),dtype=int)
    i=0
    for (key) in words:
        v1[i]=s1.get(key,0)
        v2[i]=s2.get(key,0)
        i=i+1
    return(cosine(v1,v2))

def main():
    s1=pre(inaugural.raw('2009-Obama.txt'))
    sx=inaugural.fileids()
    for file in sx:
        s2=pre(inaugural.raw(file))
        #inter=set(s1) & set(s2)
        similarity1=similarity(s1,s2)
        print(similarity1,file)
    



if __name__=='__main__':
    main()
