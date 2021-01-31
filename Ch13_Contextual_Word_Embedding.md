## Question & Answer

1. Pre-trained의 의미가 무엇인가?

Glove, word2vec과 같이 대형 corpusd에서 이미 훈련을 시켜서 정해져 있는 word embedding을 가져다 쓰는 것을 보통 말함

이 lecture에서는 ELMO에서 LM을 통해 만들어진 word representation을 본모델에 사용하는 것을 pre-trained word-vector라고 표현함

2. Unsupervised word representation

Word2vec, Glove는 unsupervised word representation이라는 것을 상기!

3. Transformer model에서 encoder 부분의 역할은 무엇인가? 마치 ELMO에서 word vector from LM과 같은 pre-trained word vector를 나타내는 것인가?

(미정)

4.BERT에서 transformer model을 사용할 때?

transformer model에서 encoder 부분을 사용함

5.ULMfit은?

첫번째는 대형 corpus에서 광범위하게 tuning(일반언어모델학습), 두번째는 과제 맞춤형 언어 모델 튜닝, 세번째는 더 국지적으로 가서 과제 분류기 튜닝이다.

## Summary

**ELMO**

- word vector from language model(bi-lstm)
- 두 개의 bi-lstm을 사용하여 양방향(문맥)을 고려했다는 것

**ULMfit**

- 특정 task를 수행하기 전에 미리 model을 pre-training 함으로써 유용하게 어느 상황에서든 사용할 수 있게 만드는 방법

**Transformer**

- Attention mechanism을 이용
- encoder : pre-trained word vector, decoder : LM 역할
- 병렬연산
- Positional encoding+Masked LM

**BERT**

- 양방향을 고려(bi-lstm이 아닌 bi-directional LM)
- k%의 단어들을 mask out, and then predict the masked words
- Next Sentence Prediction(NSP)
- Input representation : token embedding+segment embedding+position embedding
- Transformer model의 encoder로 구성되어서 NSP와 Mask LM을 예측⇒pre-training
- pre-training된 BERT를 이용해서 특정 task에 맞게 Fine-Tuning
