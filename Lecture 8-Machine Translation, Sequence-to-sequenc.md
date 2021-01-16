# Lecture 8-Machine Translation, Sequence-to-sequence and Attention

Lecture: Lecture 8

## [Machine Translation, Sequence-to-sequence and Attention]

## 4.1 Pre-Neural Machine Translation

Machine Translation : the task of translating a sentence x from one language(the source language) to a sentence y in another language(the target language).

1950s : rule based

1990-2010s : Statistical Machine Translation(SMT)

## 4.2 SMT

### #4.2.1 SMT 개념

아이디어 : Learn a probabilistic model(분포 P(y|x)를 학습하는 모델) from data

가정 : 프랑스 문장 x가 주어졌고, 가장 적합한 영어 문장 y로 번역하고 싶을 경우,

과정 : 베이즈룰(Bayes rule)을 사용해서 두 요소 Translation Model, Language Model 로 분해한 뒤, 두 요소 각각을 학습해서 결과를 도출함

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled.png)

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%201.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%201.png)

Translation Model, P(x|y) : 작은 단어 또는 구의 지엽적인 번역을 하는 역할, Parallel data(같은 내용 다른 언어 사전)로부터 학습

①Large amount of parallel data

②잠재 변수인 Alignment를 도입

Language Model, P(y) : 영어를 더 유창하게 쓸 수 있도록 하는 역할, Monolingual data(일반문서)로부터 학습

문제점

- 확률적 모델이므로 학습 자체가 모든 단어들을 돌아야 해서 계산 비용이 크다.

대안 : 강력한 독립가정을 내세워 viterbi algorithm과 같은 동적 프로그래밍을 사용하자!

- extremely complex,
- separately-designed subcomponents,
- lots of feature engineering,
- require compiling and maintaining extra resources,
- lots of human effort

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%202.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%202.png)

### #4.2.2 Alignment

Alignment : correspondence between particular words in the translated sentence pair(한 문장과 번역된 문장 간의 특정한 단어들 사이의 연결)

문제점

- 다르게 생긴 언어일수록 alignment가 복잡해질 수 있음

Many to one, one to many(fertile), many to many(ppt 10p 참고)

- 어떤 단어들은 대응하는 것이 없음

SMT에 적용하는 방식

- 많은 요소들의 결합으로 P(x,a | y)를 학습시킴
- Alignment는 잠재변수여서 EM알고리즘과 같은 특별한 학습알고리즘을 사용해야 함

## #4.3 Neural Machine Translation(NMT)

### #4.3.1 NMT 개념

Machine Translation에 Neural network 모델의 Sequence to sequne(two rnns) 구조를 사용함

ó Conditional language model

디코더, 인코더 부분으로 나뉘며 디코더의 output이 encoder의 hidden state로 들어가게 됨

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%203.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%203.png)

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%204.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%204.png)

### #4.3.2 학습과정(forward+backward)

Backpropagation 할 때, 시스템 전체의 모수가 동일하 기준으로 업데이트 됨.

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%205.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%205.png)

### #4.3.3 Beam search Algorithm

① Greedy decoding

각각의 단계에서 가장 확률이 높은 단어를 선택하는 것, 문제는 한 번 결정하고 나면 결정을 번복할 수 없음

따라서 NMT에서 Greedy decoding을 사용했을 때, 문제가 발생함

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%206.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%206.png)

② Exhaustive Search decoding(완전탐색 알고리즘)

말 그대로 step t에서 완성문장의 모든 확률을 고려해서 선택하는 것=>계산비용이 매우 큼!

③ Beam search Decoding(ppt 31p 참고)

Beam search Decoding : On each step of decoder, keep track of the k(beam size) most probable partial translations(which we call hypotheses)

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%207.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%207.png)

확률이어서 점수는 모두 음수이지만, 더 높은 점수일수록 더 좋은 문장

최적의 방안을 보장하지는 못해도 완전탐색 알고리즘보다는 매우 효율적임

<END> token 다루는 방법 : 다른 hypotheses는 다른 timestep에서 <END>token을 만들 수 있으므로 when a hypothesis produces <END> that hypothesis is complete. Place it aside and continue exploring other hypotheses.

작동을 멈추는 기준 : reach timestep T or at least n completed hypotheses

문제점 : 더 긴 hypotheses일수록 더 낮은 점수를 가지게 됨(누적합이므로)

- >길이로 정규화를 시켜줌

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%208.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%208.png)

### #4.3.4 NMT의 장점 및 단점

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%209.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%209.png)

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2010.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2010.png)

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2011.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2011.png)

- using common sense is still hard
- Uninterpretable systems do strange things
- NMT picks up biases in training data

### #4.4.4 BLEU(Bilingual Evaluation Understudy)

Machine Translation을 평가하는 방식 중 하나

기계가 번역본과 인간이 쓴 번역본을 n-gram precision을 사용하여 similarity score을 계산하는 것

너무 짧은 문장에 대해서는 brevity penalty를 부여

1. Attention(14주차)

## #5.1 Bottleneck problem

Bottleneck problem : single vector 안에 모든 정보가 담겨야 하는 과한 압박을 주는 문제

(Bottleneck : 시스템 전체 성능이나 용량이 하나 혹은 소수 개의 구성 요소나 자원에 의해 제한 받는 현상)

## #5.2 Attention(ppt 63p 참고)

아이디어 : on each step of the decoder, use direct connection to the encoder to focus on a particular part of the source sequence.

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2012.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2012.png)

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2013.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2013.png)

## #5.3 Attention의 장점

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2014.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2014.png)

## #5.4 Attention의 일반화

Given a set of vector values, and a vector query, attention is a technique to compute a weighted sum of the values, dependent on the query.

óQuery attends to the values

ódecoder hidden state(query)가 encoder hidden state(value)를 이끈다.

ó가중합=선택적 요약

óvalues들이 얼마든 attention은 summary로서 한 개의 output만을 제공한다.

## #5.5 Attention을 구하는 다양한 방식

Attention score을 다양한 방식으로 구할 수 있음

![Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2015.png](Lecture%208-Machine%20Translation,%20Sequence-to-sequenc%20aae237f2ebff458ca76574f3c5339bb6/Untitled%2015.png)