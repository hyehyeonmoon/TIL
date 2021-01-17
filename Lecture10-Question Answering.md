# Lecture10-Question Answering

Lecture: lecture 10

## [(Textual)Question Answering]

## Motivation/History

⇒We often want answers to our questions

두 가지 부분으로 나뉘는데

- Finding documents that contain an answer(cs276에서 다룸)
- Finding an answer in a paragraph or a document

두 번째는 Reading Comprehension으로 불리며 이번 강의에서 집중

## The SQuAD dataset

### 특징

- Stanford Question Answering Dataset
- 100k examples
- Answer must be a span in the passage(답이 문단 안에 무조건 들어있어야 하는 extractive QA 문제이다.)

### SQuAD evaluation, v1.1

- Exact match : 3개의 대답 중 정확히 답을 맞출 때 score를 매김
- F1 : 3개의 대답 중 정확히는 아니더라도 bag of words 관점으로 score를 매김

F1 measure is seen as more reliable and taken as primary

Both metrics ignore punctuation and articles(a, an, the only)

### SQuAD 2.0

- 1.1의 문제는 모든 질문에 대한 답이 문단에 있다는 것이여서 2.0에서는 문단에 답이 없는 질문도 넣음. "no answer"
- 어느 정도의 임계값을 넘어선 score를 받은 system은 더 복잡한 일을 수행 할 수 있을 것이라는 접근이였지만, 결과적으로 잘 수행하지 못했고, 이는 인간의 언어를 이해하지 못한다는 한계를 초기에 보임
- 현재는 2.0도 인간수준으로 높지만 기초적인 자연어 이해에 대해 실수를 하는 경우가 있다고 함.

### Limitations

- Only span-based answers
- Questions were constructed looking at the passages
- 동일 지시어(coreference) 문제를 제외하고는 Multi-fact 문제, 문장 추론 문제가 거의 없음

그럼에도 SQuAD is a well-targeted, well-structured, clean dataset

## The Stanford Attentive Reader model

### Stanford Attentive Reader model

![Lecture10-Question%20Answering%20ef25ee99ce75485785b0823df90785a5/Untitled.png](Lecture10-Question%20Answering%20ef25ee99ce75485785b0823df90785a5/Untitled.png)

![Lecture10-Question%20Answering%20ef25ee99ce75485785b0823df90785a5/Untitled%201.png](Lecture10-Question%20Answering%20ef25ee99ce75485785b0823df90785a5/Untitled%201.png)

![Lecture10-Question%20Answering%20ef25ee99ce75485785b0823df90785a5/Untitled%202.png](Lecture10-Question%20Answering%20ef25ee99ce75485785b0823df90785a5/Untitled%202.png)

### Stanford Attentive Reader++(DrQA)

1. 3 layer BiLSTM
2. 2. Bi-lstm의 hidden state를 포지션별로 concate 후, weighted sum을 하여 q 벡터 구성
3. Glove 벡터만 사용한 것이 아닌, 단어의 feature도 같이 넣어줌

## BiDAF

BiDAF(Bi-directional Attention Flow for machine Comprehensive)는 Question에서 paragraph로 한방향으로만 진행되는 Stanford Attentive Reader와 달리, attention이 양방향으로 적용됨

자세한 구조와 과정은 아래 페이지의 pdf 참고

[[Lecture 10] Question Answering](https://www.notion.so/Lecture-10-Question-Answering-af61c16a775749bd85e23ca1fd6b82d0) 

## Recent, more advanced architectures

- Fusion net :

attention function 중 MLP(additive) form과 Bilinear(Product) form 중 Bilinear가 더 적은 양의 비용이 들면서 성능은 데이터마다 다르다고 함

Bilinear form은 행렬변환에 의해 lower rank factorization, symmetric하게 만들기 때문에 더 적은 양의 비용이 듦

- Bert
- Elmo