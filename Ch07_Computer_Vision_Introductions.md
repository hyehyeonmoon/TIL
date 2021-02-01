## Summary

- VGG 개념을 배웠습니다.
- skip connection의 일종인 ResNet 개념을 배웠습니다.
- Transfer learning(전이 학습)의 개념과 효과를 알아보았습니다.
- Resnet34 model을 이용해 catdog 이미지 분류 실습을 했습니다.

## File

|File |Description|Folder|
|:-- |:-- |:-- |
|train |transfer-learning에서 사용하는 parameter 추가를 제외하면 Ch06와 동일 |. |
|set_dataset |catdog, hymenoptera 사진을 zip 파일로 다운로드 받아서 dataset 폴더에 저장  |. |
|. |강아지와 고양이 사진으로 train, test set으로 나뉘어져 있음(train set만 라벨 존재) |dataset/catdog |
|. |벌과 개미 사진으로 train, test set으로 나뉘어져 있음(라벨 존재)  |dataset/hymenoptera  |
|data_loader |dataset을 가져와서 transform을 이용해 CNN model input에 맞게 processing 그 외 Ch05과 같음 |classification |
|model_loader |transfer learning 구현 model |classification  |
|trainer |Ch05와 동일 |classification  |
|utils|Ch05와 동일 |classification  |


## 새롭게 알게 된 내용

### Vision에서 사용하는 모델

- **Alexnet**

    (2012년)의미있는 성능을 낸 첫번째 CNN 아키텍처이자, AlexNet에 쓰인 드롭아웃 등 기법은 이 분야 표준으로 자리잡을 정도로 선도적인 역할

    - conv layer, max-pooling layer, dropout layer 5개
    - fully connected layer 3개
    - nonlinearity function : ReLU
    - batch stochastic gradient descent

- GoogleNet
    - VGGNet보다 구조가 복잡해 널리 쓰이진 않았지만 아키텍처 면에서 주목을 받았다.
    - Inception module : 한 가지의 conv filter를 적용한 conv layer를 단순히 깊게 쌓는 방법도 있지만, 하나의 layer에서도 다양한 종류의 filter나 pooling을 도입함으로써 개별 layer를 두텁게 확장시킬 수 있다는 것

    ⇒ 1×1 conv filter

- **VGG**
    - Googlenet과 동시에 나온 network 구조로 간단한 구조로 주목을 받았다.
    - Convolution layer는 kernel size 3*3, padding size=1로 정하여 convolution layer에서 size 축소가 일어나지 않고 max pooling을 이용해 이미지 resize
    - 다음과 같이 설정함으로서 학습해야 할 parameter 수가 줄어들게 된다.

- **Resnet**
    - residual block을 이용해 gradient vanishing 문제를 매우 효과적으로 해결한다
    - residual block : 그레디언트가 잘 흐를 수 있도록 해주는 일종의 지름길(skip connection)

- **Densenet**
    - ResNet에서 한발 더 나아가 전체 네트워크의 모든 층과 통하는 지름길을 만들었다
    - conv-ReLU-conv 사이만 뛰어넘는 지름길을 만들었던 ResNet보다 훨씬 과감한 시도이다.

- **Squeezenet**

    (2016년)여러 모델의 weight size를 낮추어주는 모델이다.

    AlexNet 수준의 Accuracy를 달성했지만 AlexNet보다 50배 작은 파라미터 수를 이용한다.

    참고 사이트 : [https://underflow101.tistory.com/27](https://underflow101.tistory.com/27)

### Skip connection

**Residual connection**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec70984a-298c-4683-bc9f-6c05dad10b6e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec70984a-298c-4683-bc9f-6c05dad10b6e/Untitled.png)

**Highway connection**

The Highway Network preserves the shortcuts introduced in the ResNet, but augments them with a learnable parameter to determine to what extent each layer should be a skip connection or a nonlinear connection.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24cd43ed-f2a5-4e11-941a-e8be6e118c3f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/24cd43ed-f2a5-4e11-941a-e8be6e118c3f/Untitled.png)

**Dense connection**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11523f58-d3c3-432a-b07c-4b5149b36972/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/11523f58-d3c3-432a-b07c-4b5149b36972/Untitled.png)

### Transfer-learning

- Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.
- How

(1) set seed weights and train as normal

(2) Fix loaded weights and train unloaded parts

(3) Train with different learning rate on each part(불러온 part는 learning rate를 작게, 안 불러온 part는 learning rate를 크게 해서=discriminate)

### transforms class in PyTorch

```python
def get_loaders(config, input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ]),
    }
```

- transforms.Compose- Compose helps to bind multiple transforms together so we can use more than one transformation.
- transforms.ToTensor — Applies a scaling operation of changing range from 0–255 to 0–1. It converts a PIL Image or numpy ndarray to a tensor (C x H x W) in the range of 0–1.
- transforms.Normalize- This operation normalizes a tensor image with provided mean and standard deviation.
- transforms.RandomHorizontalFlip — Flipping operation helps in changing the orientation of the image. RandomHorizontalFlip changes the orientation horizontally similarly we can use RandomVerticalFlip for changing vertical orientation.
- transforms.CenterCrop- We can do cropping of an image using this transformation. CenterCrop crops the given image at the center as per the size parameter.
- transforms.Resize —To resize image this transformation can be used. It is also very useful incase of images with large dimensions to reduce it to a particular size (parameter for desired output size)
- transforms.RandomResizedCrop—Crop the given image to random size and aspect ratio. It will random scale the image and crop it, and then resize it to the demanded size.

## Reference

[https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32)

[https://ratsgo.github.io/deep learning/2017/10/09/CNNs/](https://ratsgo.github.io/deep%20learning/2017/10/09/CNNs/)

[https://en.wikipedia.org/wiki/Transfer_learning](https://en.wikipedia.org/wiki/Transfer_learning)

[https://medium.com/analytics-vidhya/transforming-data-in-pytorch-741fab9e008c](https://medium.com/analytics-vidhya/transforming-data-in-pytorch-741fab9e008c)

[김기현의 딥러닝을 활용한 자연어처리 입문 올인원 패키지 Online. | 패스트캠퍼스](https://www.fastcampus.co.kr/data_online_dpnlp)
