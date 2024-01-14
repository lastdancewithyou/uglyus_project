# uglyus_project
어글리어스 기업과 함께하는 프로젝트

회사 홈페이지 : https://uglyus.co.kr/main

## 주제

1. 

2. 어글리어스 리뷰를 활용한 ELECTRA 기반 감성 분석 모델 구축 및 웹페이지 구현
- 파이썬 웹 프레임워크인 Flask를 통해 웹페이지를 구현했다. 모델의 input을 웹페이지 상에서 선택하고, output으로 나온 결과를 보고서 형태로 출력한다.
- 코딩을 모르는 기업의 마케터, 그로스해커들도 간편한 UI로 구성된 웹페이지에 새로운 데이터를 업로드하기만 하면 서비스를 편리하게 활용할 수 있으며, 기존에 방치되어 있던 리뷰들을 적극적으로 활용할 수 있다.
- 감성 점수를 통해 어글리어스 구독자 및 구매자들의 기업에 대한 실시간 민심을 읽을 수 있고, 토픽별 긍부정 현황을 한눈에 파악할 수 있다.


**1. 데이터 업로드와 분석 옵션 선택(Daily, Weekly)이 가능한 기본 페이지**
<img width="60%" alt="스크린샷 2024-01-11 오후 8 59 03" src="https://github.com/lastdancewithyou/uglyus_project/assets/114273570/6d59fcf1-a538-4976-a933-a03f14680088">

**2. 보고서 형태로 분석 결과가 출력되는 report 페이지**
<img width="60%" alt="스크린샷 2024-01-14 오후 7 45 34" src="https://github.com/lastdancewithyou/uglyus_project/assets/114273570/2b332c36-6639-4644-81a6-0db601d36750">


## 주요 라이브러리

python==3.8.17

| library | release |
| ----------- | ---- |
| emoji | 0.6.0 |
| Flask | 3.0.0 |
| matplotlib | 3.7.4 |
| numpy | 1.24.4 |
| pandas | 2.0.3 |
| requests | 2.31.0 |
| scikit-learn | 1.3.2 |
| seaborn  | 0.13.1 |
| torch | 2.1.2 |
| tqdm | 4.66.1 |
| transformers | 4.36.2 |

## 기타
파이토치를 활용한 감성분석 학습 모델은 용량 초과로 인하여 업로드하지 못하였습니다.(electra_best_4.pt)
