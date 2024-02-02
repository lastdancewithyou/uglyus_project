# uglyus_project
어글리어스 기업과 함께하는 프로젝트

기업 컨택을 통해 어글리어스와의 협업이 성사됨

회사 홈페이지 : https://uglyus.co.kr/main

## 프로젝트 소개

프로젝트명 : **어글리us! 스마일Earth! : NLP 기반 프로젝트와 대시보드**

팀원 소개 : 정강민(팀장), 김기수, 김세연, 윤여빈

## 주제

**1. 태블로를 활용한 데이터로 보는 2023 어글리어스 대시보드**
- 기업의 홍보를 위한 대시보드를 제안, 홈페이지에 임베드하는 방식으로 활용할 수 있도록 대시보드를 제작함 

**2. 어글리어스 리뷰를 활용한 KoELECTRA 기반 감성 분석 모델 구축 및 웹페이지 구현**
- 파이썬 웹 프레임워크인 Flask를 통해 웹페이지를 구현함. End-to-End 방식으로 모델의 input을 웹페이지 상에서 선택하고, output으로 나온 결과를 html 상에 보고서 형태로 출력됨
- 직접 제작한 지수(토픽 감성 지수)를 활용하여 리뷰를 통한 기업 평가 지표를 제작함
- LDA 토픽 모델링을 수행, 토픽별로 토픽 감성 지수를 파악할 수 있기에 모든 리뷰를 확인하지 않고 해당 토픽의 세부적인 리뷰만 확인하면 되기에 작업 프로세스 단축 -> Customer Service 팀의 업무의 자동화 실천
- 코딩을 모르는 기업의 마케터, CS팀도 간편한 UI로 구성된 웹페이지에 새로운 데이터를 업로드, 기간을 입력하기만 하면 자연어 기반 AI 분석 서비스를 이용할 수 있음
<br/>

### 1. Workflow

<img width="50%" alt="워크플로우" src="https://github.com/lastdancewithyou/uglyus_project/assets/114273570/f01e0d5e-a688-4c10-893e-b546015422d8">

### 2. 분석 프로세스
<img width="50%" alt="분석 프로세스" src="https://github.com/lastdancewithyou/uglyus_project/assets/114273570/48d6d596-1a87-4b71-856c-a7d589791776">

### 3. 모델 실험 결과

<img width="55%" alt="모델 실험 결과" src="https://github.com/google-research/electra/assets/114273570/9bf06012-cbf9-4472-b5d8-5c16a88fba37">

### 4. 데이터 업로드와 옵션 선택(Daily, Weekly)이 가능한 Main Page

<img width="60%" alt="input" src="https://github.com/lastdancewithyou/uglyus_project/assets/114273570/e4915b93-4cbc-4155-aa25-4607a052e074">

### 5. 보고서 형태로 분석 결과가 출력되는 Report Page

<img width="60%" alt="output" src="https://github.com/lastdancewithyou/uglyus_project/assets/114273570/f58cec0e-93d1-4f6a-9fda-2ce1f21e7efb">

## 대시보드
대시보드는 twbx 파일의 용량 초과로 인하여 태블로 퍼블릭에 업로드하는 것으로 대체하였습니다.

태블로 퍼블릭 링크 : https://public.tableau.com/views/UglyUsDashboard/sheet0?:language=ko-KR&:display_count=n&:origin=viz_share_link

## 시연 영상

- 화면을 클릭하면 해당 영상의 유튜브 페이지로 이동합니다.

**1. 대시보드 시연 영상**

[![Video Label](http://img.youtube.com/vi/MoZzAdNxTRQ/0.jpg)](https://youtu.be/MoZzAdNxTRQ)

**2. 웹페이지 시연 영상**

[![Video Label](http://img.youtube.com/vi/BOcf5czlMIk/0.jpg)](https://youtu.be/BOcf5czlMIk)

## 컨퍼런스 발표 영상

[![Video Label](http://img.youtube.com/vi/HYUYUXvb-LQ/0.jpg)](https://https://youtu.be/HYUYUXvb-LQ)

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
파이토치를 활용한 감성분석 학습 모델(KoELECTRA)은 용량 초과로 인하여 업로드하지 못하였습니다.
