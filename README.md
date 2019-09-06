창원대학교 정보통신공학과 2019 하계 딥러닝 특강
=============================================
## 목차
1. 개요
2. DNN (Deep Neural Network)
3. CNN (Convolutional Neural Network)
4. GAN (Generative Adversarial Network)
5. RNN (Recurrent Neural Network)
6. Project (GAN, Style Transfer)
<br></br>
* * *
<br></br>

## 개요
* 인공지능과 머신러닝은 지난 수 십년간 대학과 연구소를 중심으로 한 학계가 주도를 해 왔다. 하지만 이러한 패러다임은 최근 들어 완전히 바뀌게 되었으며, 우리가 잘 알고 있는 구글, 페이스북, 마이크로소프트 같은 거대 IT기업이나 첨단 기술을 보유한 소규모 스타트 업 위주의 기업들이 주도하는 양상이다.
* 이번 교육과정의 강의 내용은 기본적인 TensorFlow 내용 및 사용법, 기본 신경망, CNN(Convolutional Neural Networks), RNN(Recurrent Neural Networks), GAN등 널리 알려진 딥러닝 기술들에 대한 내용이었다. 이러한 딥러닝 망들을 활용하여, 대규모 영상 인식, 자연어처리뿐만 아니라, 현재 다양한 딥러닝 응용으로 소개되고 있는 여러 가지 예들을 구현할 수 있다.

<!DOCTYPE html>
<br></br><br></br>

<body>
 <div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/1200px-TensorFlowLogo.svg.png" width="50%"/>
   <br></br>
   TensorFlow Logo
 </div>
</body>

<br></br><br></br>

## DNN
* 
<br></br>

## CAN
* adfasjdflkajsldf
<br></br>

## GAN
* dfawefawegawrgawg
<br></br>

## RNN
* 
<br></br>
## Project
직접 Style Transfer를 실습해 볼 수 있다. 실습을 위한 준비물은 아래 '요구사항' 부분에 기술해놓았다.
프로젝트 실습에 필요한 소스는 ```ProjectDir2/tensorflow-fast~``` 디렉토리에 있으며, 아래 몇가지 요구사항이 충족되어야 한다.
<br></br><br></br>

**1. 요구사항**
* Pycharm
* Python3 (version 3.5 이상)
* 주요 라이브러리
  * tensorflow, numpy, matplotlib, scipy(1.13.x), PIL(or Pillow), os
* Pre-trained VGG19
  * 프로젝트에 첨부했으나, 다운로드가 제대로 되지 않는 경우가 있었다. 정상파일 크기(약 500MB)가 아니거나 Train시 문제가 생기면 아래 다운로드 링크를 통해 다운로드 하면된다. Test시에는 사용되지 않는다.
  * 다운로드 링크 : [Pre-trained VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)
  * 다운로드 후 ```ProjectDir2/tensorflow-fast~```에 ```pre_trained_model``` 디렉토리를 만든 후 그 안에 저장해야 한다.
* MSCOCO train2014 DB
  * 마찬가지로 Train할 경우 필요하다. 용량이 매우 커서(약 13GB) 첨부하지 못했고, 별도로 다운로드 해야한다.
  * 다운로드 링크 : [MSCOCO train2014 DB](http://images.cocodataset.org/zips/train2014.zip)
  * 다운로드 후 ```ProjectDir2/tensorflow-fast~```에 두면 된다.
  * 참고로 다운로드가 ~~더럽게~~ 오래걸린다.
<br></br><br></br>


**<p></p>2-1. Project 수행과정 : Test**
* 기존에 존재하는 여러 Style Transfer Project들을 참조했고, 그 중 본인의 작업환경에 가장 적합한 Project를 찾아냄(아래 링크).
  * Thanks to [hwalsuklee](https://github.com/hwalsuklee) : [TF Fast Style Transfer](https://github.com/hwalsuklee/tensorflow-fast-style-transfer)
* Test과정에서 요구되는 Parameter들을 알아봄.
  * Required
    * ```--content``` : Filename of the content image. Default: content/female_knight.jpg
    * ```--style-model``` : Filename of the style model. Default: models/wave.ckpt
    * ```--output``` : Filename of the output image. Default: result.jpg
  * Optional
    * ```--max_size``` : Maximum width or height of the input images. None do not change image size. Default: None
* 적절한 Parameter값을 넣고 결과물 확인
* 결과물

<!DOCTYPE html>
<div>
 <img src="https://user-images.githubusercontent.com/28914096/64403879-24688b80-d0b5-11e9-8f48-2fdb29b7ee80.jpg" width="30%"  align="left"/>
 <br></br>
 Style Image

 <img src="https://user-images.githubusercontent.com/28914096/64403904-2fbbb700-d0b5-11e9-8383-a3d45882e380.jpg" width="30%"  align="center"/>
 <br></br>
 Content Image

 <img src="https://user-images.githubusercontent.com/28914096/64403919-39451f00-d0b5-11e9-9494-c5e4e7c5ad8a.jpg" width="30%"  algin="right"/>
 <br></br>
 Result Image
</div>
 
<br></br><br></br>

**<p></p>2-2. Project 수행과정 : Train**
* 나만의 Model을 만들기 위해 학습에 필요한 DataSet을 구함(MSCOCO Train2014).
* 마찬가지로, Train과정에서 요구되는 Parameter들을 알아봄.
<br></br><br></br>

<br></br>
<br></br>
