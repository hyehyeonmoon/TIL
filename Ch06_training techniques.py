#!/usr/bin/env python
# coding: utf-8

# ## Optimizer

# ### SGD
# sgd의 단점 : 비등방성 함수(방향에 따라 성질이 달라지는 함수)에서는 탐색 경로가 비효율적

# In[1]:


class SGD:
    def __init__(self, lr=0.01):
        self.lr=lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[keys] -= self.lr*grads[key]


# ### Momentum
# 가속도를 의미, 이전 iteration의 미분값*alpha 를 SGD에 더해주어 마치 가속도를 주는 듯한 효과

# In[2]:


class Momentum:
    def __init__(self, lr=0.01, momentum=0.6):
        self.lr=lr
        self.momentum=momentum
        self.v=None
    
    def update(self, params, grads):
        if self.v is None:
            self.v={}
            for key, val in params.items():
                self.v[key]=np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


# ### Adagrad
# 학습률을 조정한다는 개념으로, 매개변수의 원소 중에서 많이 움직인(크게 갱신된) 원소는 학습률을 낮게 만들어, 학습률 감소가 매개변수의 원소마다 다르게 적용시킴.  
# 제곱합을 학습률에 나누기 때문에 무한히 계속 학습하면 학습률이 0이 되버리기 때문에 **Rmsprop**라는 개선된 방식이 있음.
# 이는 지수이동평균을 이용해 과거의 모든 기울기를 균일하게 더해가는 것이 아니라 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영.

# In[ ]:


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr=lr
        self.h=None
        
    def update(self, params, grads):
        if self.h is None:
            self.h={}       
            for key, val in params.items():
                self.h[keys]=np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key]*grads[key]
            params[key] -= self.lr*grads[key] / (np.sqrt(self.h[key])+1e-7)
        


# ### Adam
# Rmsprop와 Momentum을 합친 optimizer로 현재 가장 많이 범용적으로 쓰임

# ## Overfitting과 Dropout

# In[4]:


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None
    
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask=np.random.rand(*x.shape) > self.dropout_ratio
            return x*self.mask
        else:
            return x*(1.0-self.dropout_ratio)
        
    def backward(self, dout):
        return dout*self.mask


# In[ ]:




