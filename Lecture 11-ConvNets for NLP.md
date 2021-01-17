## [ConvNets for NLP]

## From RNNs to CNN

### RNN의 단점

"the country of my birth"에서 birth라는 단어를 알기 위해서는 처음부터 birth 앞에까지 모든 단어들이 필요함

often capture too much of last words in final vector

시간이 오래 걸리고 bottleneck problem

### CNN의 장점 및 단점

- 장점

N-gram features를 얻을 수 있음

국소적인 패턴을 얻을 수 있음

RNN보다 빠르며 bottleneck problem을 해결

- 단점

특정 길이의 단어들로 벡터를 계산하는 것이 CNN으로

구가 문법적이든 아니든 언어적이거나 타당하지는 않다

## CNNs

### 1D convolution for text

![Untitled](https://user-images.githubusercontent.com/55529617/104840228-63455d00-5909-11eb-8c0c-0cfc565838fa.png)

- 더 넓은 범위를 한번에 보고 싶다면
1. 더 큰 filter를 사용
2. 더 넓게 dilated convolution을 사용
3. CNN을 더 깊게 만들기

### Other notion

![Untitled 1](https://user-images.githubusercontent.com/55529617/104840220-5fb1d600-5909-11eb-849b-5c69df1fa783.png)

### 용어

Channel, Filter, Kernel, Stride, Padding, Pooling, Feature Map, Activation Map

## Simple CNN for sentence Classification

- 목표 : Sentence Classification
- 성능 : 하나의 CNN layer와 pooling을 사용해서 꽤 높은 성능을 보임
- 구조

Dropout과 Max-norm regularization을 사용

multiple filters(100) with various widths(various n-grams)

Multi-channel input idea(word vector와 word vector 복사본을 만들어 그 중 하나는 고정, 다른 하나는 fine-tuned)

![Untitled 2](https://user-images.githubusercontent.com/55529617/104840222-60e30300-5909-11eb-91d3-d5f47915d15a.png)

## Toolkit & ideas for NLP task

### Model comparison

**Good baseline Model**

Bag of vectors : 문장의 word embedding의 평균을 이용해 text classification

Window Model : good for single word classification for problems that do not need wide context

**Advanced Model**

CNNs : good for representing sentence meaning(sentence classification)

RNNs : good for sequence tagging/classification/language models, but slower than CNNs

### Toolkit

- Gated units used vertically

LSTM과 GRU에서 gate를 연결해주는 개념에서 차용한 Skipping

Residual Net과 Highway Net이 있음

- 1*1 convolutions

먼저 1*1 Convolution을 사용하면 필터의 개수가 몇 개 인지에 따라 output의 dimension은 달라지지만, 원래 가로 세로의 사이즈는 그대로 유지된다.

그래서 filter 의 개수를 원래 input의 dimension 보다 작게 하면, dimension reduction의 효과가 난다.

원래 image 쪽에서 Convolution layer는 "Spatial Relation"을 고려하여 이 image가 어떤 image인지 패턴을 통해 파악하는 용도인데, 1*1 사이즈를 사용한다는 것은 한 픽셀만 고려하기 때문에 패턴 인식보다는 dimesion reduction이라는 전처리 용도로 생각해야 한다.

Dimension reduction을 이용하기 위해 1*1 conv 의 개수를 줄이면, activation의 depth가 작아져서 filter의 총 parameter의 개수가 감소한다.

이 형태를 bottleneck 구조라고 하는데, dimension reduction을 한 뒤 모든 연산을 하고 다시 filter의 갯수를 늘려서 고차원으로 늘리는 방법을 이용하기 때문에 bottleneck이라고 부른다.

출처: [https://yunmap.tistory.com/entry/전산학특강-CS231n-1X1-Convolution-이란](https://yunmap.tistory.com/entry/%EC%A0%84%EC%82%B0%ED%95%99%ED%8A%B9%EA%B0%95-CS231n-1X1-Convolution-%EC%9D%B4%EB%9E%80)

- Batch Normalization

internal covariate shift 문제를 해결하기 위해 배치 단위로 정규화를 시켜주는 것

학습속도가 빨라지고 초기화에 덜 민감해진다

과적합을 예방

복잡도(계산량)을 늘림

### Ideas

![Untitled 3](https://user-images.githubusercontent.com/55529617/104840223-617b9980-5909-11eb-88cb-5cc0ba8e88a5.png)

## Deep CNN for sentence classification

### Structure

- Starting point : sequence models have been very dominant in NLP
but all the models are basically not very deep
- Works from the character-level
- Result is constant size, since text is truncated or padded
- Local max pooling

![Untitled 4](https://user-images.githubusercontent.com/55529617/104840224-617b9980-5909-11eb-92bc-11425164236a.png)

![Untitled 5](https://user-images.githubusercontent.com/55529617/104840225-62143000-5909-11eb-9466-51eba02c32a3.png)

### Result

✓ deeper networks are better
✓ However, if it is too deep, its performance will be degraded
✓ Shrinkage method – “MaxPooling“ is easier

## Quasi-recurrent Neural Networks

![Untitled 6](https://user-images.githubusercontent.com/55529617/104840227-62acc680-5909-11eb-8033-54ba0c59bc27.png)
