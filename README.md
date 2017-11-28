# Exploring App Store Reiview 분석

Jupyter notebook 링크: [링크](https://nbviewer.jupyter.org/github/simonjisu/app_store_reviews_analysis/blob/master/Exploring_App_Store_Reviews.ipynb)

Slider Share 링크: [링크](https://www.slideshare.net/secret/LaSRTJVgvvtuAx)

개인 프로젝트

기간: 2017.10.16 - 2017.11.27 (6주)

## 목적
App Store 에는 수많은 사람들이 앱에 관련해 리뷰를 달고 해당 앱에대해 평점을 매기고 있다. 많은 개발자들도 종종 리뷰를 보면서 향후 이용자와 소통을 하고 향후 앱 개발의 방향성을 탐색하고 고민하게 된다. 평소에 앱 스토어를 이용하면서 리뷰들이 정확하게 이 앱을 평가하고 있는지 혹은 앱들의 리뷰로 어떤 앱을 별점을 예측 할 수 있을 지 궁금해졌다. 그래서 이 프로젝트를 시작했다.

## 목차
### Part 1. 데이터 획득

### Part 2. 데이터 정제
* 띄어쓰기 문제
* 오타 및 축약어
### Part 3. 데이터 분석
* A. 키워드분석: 문서 전체의 키워드 / 앱별로 키워드 추출
* B. 문서 군집화: 리뷰를 통한 클러스터링
* C. 문서 분류: Navie Bayesian을 통한 평점 분류
### Part 4. 결론 및 향후과제
#### 결론
앱 리뷰의 주요 키워드 추출, 클러스터링과 평점을 통한 분류를 시도해 보았다. 이를 통해 얻은 결론은 아래와 같다. 
1. 앱 스토어 리뷰 데이터를 사용하려면 전처리에서 꽤 많은 처리가 필요하며, 띄어쓰기, 형태소 분석과 단어 후처리에 많은 시간을 쏟아야 한다.
2. 주요 키워드 추출로 앱의 내용을 설명 할 수 있다는 점을 발견 했다. 그러나 가끔 '구매내역 삭제 부탁' 관련 글들이 등장하는 경향을 보였다. 해당 단어들은 앱스토어 평가에 영향을 미칠 수 있기 때문에 제거하지는 않았다.
3. 문서 클러스터링의 경우 리뷰내용에 따라서 클러스터링 하기 어렵다는 것을 알 수 있었다. 이는 각 리뷰마다 앱의 기능이 담긴 내용은  적으며 주로 앱에 대한 평가의 단어가 많기 때문이다. 그러나 그중에서도 게임 카테고리는 비교적 분류가 잘 되는 편이다.
4. "사용자가 평가한 앱에 대한 평점이 리뷰와 정의 상관관계가 있다"라는 것은 부분적으로 맞지만, 완전이 선형적인 관계는 아니다. 또한, 나이브 베이지안 모델의 리뷰 분류는 품사 선택에 영향을 받지 않는다.
#### 향후과제
1. 추가적인 공부를 통해, 단어들을 미리 위키백과 혹은 다른 데이터로 word2vec으로 학습 시킨후 다른 모델로 분류를 다시 진행해본다.
2. 아직은 안되겠지만 공부를 더 해서 카테고리와 키워드를 설정하면 앱스토어 리뷰를 쓰는 챗봇을 만들자!

