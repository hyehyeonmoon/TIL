# Lecture 7-Vanishing Gradients and Fancy Rnns

Lecture: Lecture 7

## [Vanishing Gradients and Fancy RNNs]

### Index

- Vanilla RNN의 문제점 : Vanishing gradient
- Exploding gradient solution : Gradient clipping
- Vanishing gradient solution : connection
- LSTMs are powerful but GRUs are faster
- Use bidirectionality RNN when possible
- Multi-layer RNN's are powerful, but you might need skip/dense-connections if it's deep

### Vanilla RNN의 문제점 : Vanishing gradient

- 원인

![Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled.png](Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled.png)

1. If w_h is small, then this term gets vanishingly small as i and j get further apar

![Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%201.png](Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%201.png)

2. If the largest eigenvalue of w_h is less than 1, then the gradient(left) will shrink exponentially, 

- 문제점
1. model weights are only updated only with respect to near effects, not long-term effects.
2. 두 가지 상황 중 어떤 것이 맞는지 확인이 불가능한데, 진짜 data에서 시점 t와 시점 t+n이 아무 관계가 없는 경우와 잘못된 모수로 인해 관계가 있는데 없는 것처럼 보이는 경우
3. the model is unable to predict similart long-distance dependencies at test time
4. RNN-LMs are better at learning from sequential recency than syntatic recency

### Solution for Exploding gradient in RNN: Gradient clipping

if the norm of the gradient is graeter than some threshold, scale ift down before applying SGD update 

⇒신경망 파라미터의 L2norm을 구하고, 이 norm의 크기를 제한하는 방법으로 기울기 벡터 gradient vector의 방향은 유지하되, 그 크기는 학습이 망가지지 않을 정도로 줄어들게 하느 ㄴ것

### Solution for Vanishing gradient in RNN : LSTM

![Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%202.png](Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%202.png)

![Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%203.png](Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%203.png)

⇒ The LSTM architecture makes it easier for the RNN to preserve information over many timesteps

### LSTM의 간소화 : GRU

![Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%204.png](Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%204.png)

⇒  **Rule of thumb: start with LSTM, but switch to GRU if you want something more efficient**

## vanishing/exploding gradient problem : other solution

Resnet, DenseNet, HighwayNet

⇒ Solution: lots of new deep feedforward/convolutional architectures that
add more direct connections (thus allowing the gradient to flow)

## Bidirectional RNNs

- motivation

만약 sentiment classification 문제에서 "the movie was terribly exciting" 문장이 주어졌다고 해보자. 이 때, terribly의 경우 앞뿐만 아니라 뒤의 exciting도 문맥상 뜻을 정확하게 하기 위해서 정보를 습득해야 함.

- Architecture

![Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%205.png](Lecture%207-Vanishing%20Gradients%20and%20Fancy%20Rnns%20bcc999345626452aa3e255c871a3a62b/Untitled%205.png)

- Effect

BERT (Bidirectional Encoder Representations from
Transformers) is a powerful pretrained contextual
representation system built on bidirectionality

— bidirectional RNNs are only applicable if you have access to the entire input sequence.

## Multi-layer RNNs

- The lower RNNs should compute lower-level features and the
higher RNNs should compute higher-level features.

- For example: In a 2017 paper, Britz et al find that for Neural Machine Translation, 2 to 4 layers is best for the encoder RNN,
and 4 layers is best for the decoder RNN
• However, skip-connections/dense-connections are needed to train
deeper RNNs (e.g. 8 layers)