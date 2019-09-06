창원대학교 정보통신공학과 2019 하계 딥러닝 특강
=============================================
## 목차
* 개요
* Project (Style Transfer with TensorFlow)
  * 요구사항
  * 수행과정
    * Test과정
    * Train과정
* Reference
<br></br>

### 개요
* 인공지능과 기계학습은 지난 수 십년간 대학과 연구소를 중심으로 한 학계가 주도를 해 왔다. 하지만 이러한 패러다임은 최근 들어 완전히 바뀌게 되었으며, 우리가 잘 알고 있는 구글, 페이스북, 마이크로소프트 같은 거대 IT기업이나 첨단 기술을 보유한 소규모 스타트 업 위주의 기업들이 주도하는 양상이다.
* 이번 교육과정의 강의 내용은 기본적인 TensorFlow 내용 및 사용법, 기본 신경망, CNN(Convolutional Neural Networks), RNN(Recurrent Neural Networks), GAN등 널리 알려진 딥러닝 기술들에 대한 내용이었다. 이러한 딥러닝 망들을 활용하여, 대규모 영상 인식, 자연어처리뿐만 아니라, 현재 다양한 딥러닝 응용으로 소개되고 있는 여러 가지 예들을 구현할 수 있다.
* 이번강의에서 처음 마주한 신경망은 GAN(Generative Adversarial Network)이었다. 학과 강의중 영상처리라는 과목에서 여러 이미지의 변환, 잡음 처리 등을 해보며 흥미를 느꼈는데, 기계학습을 이런 이미지 변환에 적용시킨다는 것이 흥미롭다 생각하여 프로젝트 주제를 Style Transfer를 선택했다.
* 이번 과정에서 배운 기계학습의 기본개념과 이론, 각 신경망에 대한 강의자료를 ```~/lec_files_DL```에 첨부했다.
<br></br>

### Project
직접 Style Transfer를 실습해 볼 수 있다. 실습을 위한 준비물은 아래 '요구사항' 부분에 기술해놓았다.
프로젝트 실습에 필요한 소스는 ```ProjectDir2/tensorflow-fast~``` 디렉토리에 있으며, 아래 몇가지 요구사항이 충족되어야 한다.
<br></br>

**1. 요구사항**
* Pycharm (or 그 외 Python작업 가능한 환경)
* Python3.5 이상
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
  * [TF Fast Style Transfer](https://github.com/hwalsuklee/tensorflow-fast-style-transfer)
* Test과정에서 요구되는 Parameter들을 알아봄.
  * Required
    * ```--content``` : Content Image의 파일 이름. Default: content/female_knight.jpg
    * ```--style-model``` : Style Model의 파일 이름. Default: models/wave.ckpt
    * ```--output``` : 결과롤 나온 Image의 파일 이름 설정. Default: result.jpg
  * Optional
    * ```--max_size``` : Input Image의 최대 높이/너비 설정, None은 크기 변경x. Default: None
* Pycharm을 이용할 경우 각 Parameter는 ```Run>Edit Configurations```에서 설정
  * 예시
  <img src="https://user-images.githubusercontent.com/28914096/64407900-706cfd80-d0c0-11e9-9c43-48b7b585d0b7.png"><br>
  * script path도 ```~/run_test.py```로 되어있는지 확인해주자. (맞지 않으면 ```run_test.py```가 있는 위치를 적어준다.) 
* 적절한 Parameter값을 넣고 결과물 확인
<br>

(1) Style : La Muse 
<!DOCTYPE html>
<table border="0">
 <tr>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64403879-24688b80-d0b5-11e9-8f48-2fdb29b7ee80.jpg" width="300px" height="300px">
  </td>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64403904-2fbbb700-d0b5-11e9-8383-a3d45882e380.jpg" width="300px" height="300px">
  </td>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64403919-39451f00-d0b5-11e9-9494-c5e4e7c5ad8a.jpg" width="300px" height="300px">
  </td>
 </tr>
 <tr>
  <td>Style Image(La Muse)</td> <td>Content Image</td> <td>Result Image</td>
 </tr>
</table>

(2) Style : Udnie
<!DOCTYPE html>
<table border="0">
 <tr>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64407370-20da0200-d0bf-11e9-9296-553446a8b8aa.jpg" width="300px" height="300px">
  </td>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64407348-19b2f400-d0bf-11e9-8640-bdc1b985b270.jpg" width="300px" height="300px">
  </td>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64407334-115ab900-d0bf-11e9-81d3-3d9336a64902.jpg" width="300px" height="300px">
  </td>
 </tr>
 
 <tr>
  <td>Style Image(Udnie)</td> <td>Content Image</td> <td>Result Image</td>
 </tr>
</table>

<br></br>
**<p></p>2-2. Project 수행과정 : Train**
* 나만의 Model을 만들기 위해 학습에 필요한 DataSet을 구함(MSCOCO Train2014).
* 마찬가지로, Train과정에서 요구되는 Parameter들을 알아봄.
  * Required
    * ```--style```: Style Image의 파일 이름(학습을 통해 만들고 싶은 Style Image). Default: images/wave.jpg
    * ```--output```: 학습을 통해 만든 모델을 저장할 경로. Train-log 역시 같은 곳에 저장된다. Default: models
    * ```--trainDB```: MSCOCO DB의 경로. Default: train2014
    * ```--vgg_model```: Pre-trained model의 경로. Default: pre_trained_model
  * Optional
    * ```--content_weight```: content-loss의 가중치. Default: 7.5e0
    * ```--style_weight```: style-loss의 가중치. Default: 5e2
    * ```--tv_weight```: total-varaince-loss의 가중치. Default: 2e2
    * ```--content_layers```: content손실 계산에 사용되는 공백으로 구분 된 VGG-19 레이어 이름. Default: relu4_2
    * ```--style_layers```: style손실 계산에 사용되는 공백으로 구분 된 VGG-19 레이어 이름. Default: relu1_1 relu2_1 relu3_1 relu4_1 relu5_1
    * ```--content_layer_weights```: content손실에 대한 각 content layer의 공백으로 구분 된 가중치. Default: 1.0
    * ```--style_layer_weights```: 각 style layer의 공백으로 분리된 가중치. Default: 0.2 0.2 0.2 0.2 0.2
    * ```--max_size```: input images의 최대 높이/최대 너비. Default: None
    * ```--num_epochs```: 학습시 epoch 수. Default: 2
    * ```--batch_size```: Batch size. Default: 4
    * ```--learn_rate```: Adam optimizer 학습률. Default: 1e-3
    * ```--checkpoint_every```: checkpoint 저장 빈도. Default: 1000
    * ```--test```: 학습 하는동안의 content image 이름. Default: None
    * ```--max_size```: 학습하는 동안의 input images의 최대 높이/최대 너비, None은 image size 변경x. Default: None
* 각 Parameter 설정은 Test단계와 동일
* Optional Parameters는 모두 Default로 두고 Required Parameters만 적절히 설정함. 결과 확인.
<br>

(1) Style : Life of Farmers
<!DOCTYPE html>
<table border="0">
 <tr>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64409364-b6779080-d0c3-11e9-9772-f8b7befae10d.jpg" width="300px" height="300px">
  </td>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64409396-c3947f80-d0c3-11e9-92a6-633be25c87a8.jpg" width="300px" height="300px">
  </td>
  <td>
   <img src="https://user-images.githubusercontent.com/28914096/64409357-b4153680-d0c3-11e9-93ca-0c34d1c685bf.jpg" width="300px" height="300px">
  </td>
 </tr>
 
 <tr>
  <td>Style Image(Life of Farmers)</td> <td>Content Image</td> <td>Result Image</td>
 </tr>
</table>

(2) Style : __
<!DOCTYPE html>
<table border="0">
 <tr>
  <td>
   <img src="" width="300px" height="300px">
  </td>
  <td>
   <img src="" width="300px" height="300px">
  </td>
  <td>
   <img src="" width="300px" height="300px">
  </td>
 </tr>
 
 <tr>
  <td>Style Image</td> <td>Content Image</td> <td>Result Image</td>
 </tr>
</table>

<br></br>

### Reference
* hwalsuklee's Fast Style Transfer with Tensorflow : [https://github.com/hwalsuklee/tensorflow-fast-style-transfer](https://github.com/hwalsuklee/tensorflow-fast-style-transfer)
