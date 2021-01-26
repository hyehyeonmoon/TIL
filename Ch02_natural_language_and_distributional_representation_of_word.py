#!/usr/bin/env python
# coding: utf-8

# ## 말뭉치(문장) 전처리

# In[20]:


import sys
sys.path.append('...')
import numpy as np
from common.util import preprocess
import matplotlib.pyplot as plt


# In[2]:


def preprocess(text):
    text=text.lower()
    text=text.replace('.', ' .')
    words=text.split(' ')
    
    word_to_id={}
    id_to_word={}
    for word in words:
        if word not in word_to_id:
            new_id=len(word_to_id)
            word_to_id[word]=new_id
            id_to_word[new_id]=word
            
    corpus=np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


# In[3]:


text='You say goodbye and I say hello'
corpus, word_to_id, id_to_word=preprocess(text)


# In[4]:


corpus


# ## 말뭉치로부터 동시발생 행렬을 만들어주는 함수

# In[5]:


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size=len(corpus)
    co_matrix=np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx=idx-i
            right_idx=idx+i
            
            if left_idx>=0:
                left_word_id=corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id=corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
        
    return co_matrix


# ## 단어 벡터의 유사도 : 코사인 유사도

# In[6]:


def cos_similarity(x,y, eps=1e-8):
    nx=x/(np.sqrt(np.sum(x**2))+eps) #x의 정규화
    ny=y/(np.sqrt(np.sum(y**2))+eps) #y의 정규화
    return np.dot(nx, ny)


# ## 유사 단어의 랭킹 표시

# In[7]:


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    #검색어를 꺼낸다
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.' % query)
        return
    
    print('\n[query]'+query)
    query_id=word_to_id[query]
    query_vec=word_matrix[query_id]
    
    #코사인 유사도 계산
    vocab_size=len(id_to_word)
    similarity=np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i]=cos_similarity(word_matrix[i], query_vec)
        
    #코사인 유사도를 기준으로 내림차순 출력
    count=0
    for i in (-1*similarity).argsort(): #argsort는 원소들을 오름차순으로 정렬한 index를 반환, 우리는 내림차순을 원하므로 -1을 유사도에 곱해줌
        if id_to_word[i]==query:
            continue
        print('%s: %s' % (id_to_word[i], similarity[i]))
    
        count += 1
        if count>= top:
            return


# In[8]:


#검색어 you와 유사한 단어를 상위 5개만 출력
text='You say goodbye and I say hello.'
corpus, word_to_id, id_to_word=preprocess(text)
vocab_size=len(word_to_id)
C=create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)


# In[9]:


C


# ## 동시발생행렬을 PPMI(Positive Pointwise Mutual Information) 변환하는 함수

# In[13]:


def ppmi(C, verbose=False, eps=1e-8):
    M=np.zeros_like(C, dtype=np.float32)
    N=np.sum(C)
    S=np.sum(C, axis=0)
    total=C.shape[0]*C.shape[1]
    cnt=0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi=np.log2(C[i,j]*N/(S[j]*S[i])+eps)
            M[i,j]=max(0,pmi)
            
            if verbose:
                cnt+=1
                if cnt % (total//100) ==0:
                    print('%.1f%% 완료' % (100*cnt/total))
    return M


# In[14]:


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)  # 유효 자릿수를 세 자리로 표시
print('동시발생 행렬')
print(C)
print('-'*50)
print('PPMI')
print(W)


# ## SVD

# In[15]:


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)


# In[16]:


np.set_printoptions(precision=3)  # 유효 자릿수를 세 자리로 표시
print(C[0])
print(W[0])
print(U[0])


# In[21]:


# 플롯
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()


# ## PTB

# In[22]:


from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')

print('말뭉치 크기:', len(corpus))
print('corpus[:30]:', corpus[:30])
print()
print('id_to_word[0]:', id_to_word[0])
print('id_to_word[1]:', id_to_word[1])
print('id_to_word[2]:', id_to_word[2])
print()
print("word_to_id['car']:", word_to_id['car'])
print("word_to_id['happy']:", word_to_id['happy'])
print("word_to_id['lexus']:", word_to_id['lexus'])


# ## PTB dataset에 통계기반기법 적용(고속 SVD)

# In[23]:



window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('동시발생 수 계산 ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('PPMI 계산 ...')
W = ppmi(C, verbose=True)

print('calculating SVD ...')
try:
    # truncated SVD (빠르다!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (느리다)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

