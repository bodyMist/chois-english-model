# chois-english-model
딥러닝 모델을 이용한 영어 문제 생성 웹 사이트 개발 프로젝트
> 본 페이지는 model-server 레포지토리입니다. 웹서버의 레포지토리는 다음의 링크를 참조해 주시길 바랍니다.   
> 링크 : [api-server](https://github.com/bodyMist/chois-english-back)

## 사용 기술
* Flask
* Image Captioning: OFA
* 형태소 분석: NLTK
* 문장 유사도: SBERT

### 기술 채택 이유
* Flask : 쉽고 빠르게 모델 서버를 구축할 수 있다는 장점
* OFA : 높은 평가 지수를 자랑하며 사전 훈련된 모델을 제공함
* SBERT : 사용자의 입력과 답안 사이의 유사도를 최대한 유연하게 비교하고자 사용
* NLTK : tokenize를 이용하기 위해 영어 문제를 생성하고자 한다

## 주요 기능
1. OFA의 Image Captioning과 NLTK 라이브러리를 이용한 영어 문제 생성
2. SBERT와 코사인 유사도를 기반으로 한 문제 정답 채점

## ISSUE
1. Image Captioning 모델의 변경
    * 최초 계획으로 Kakaobrain 사의 PORORO 모델을 채택
    * 성능 테스트 결과 사용 부적합 판정
    * 이후 CIDEr(Image Captioning의 성능 평가 지수)가 약 130점으로 최고 성능을 자랑하던 OFA 모듈을 채택
2. 답안 채점은 어떻게 할 것인가?
    * 최초 계획 : word2Vec과 sentence2Vec로 임베딩한 뒤 코사인 유사도로 검사
    * 문제점) word2Vec과 sentence2Vec은 고정 값으로 임베딩하기 때문에 유연한 벡터화가 불가능하다는 단점이 있다
    * 대안) SBERT는 자연어의 의미와 주변 문장과의 유사성까지 파악하여 훨씬 유연한 답안 채점이 가능하다.
3. 문제 생성 품질 개선
    * NLTK 라이브러리를 이용하여 불용어(분석에 큰 의미가 없는 단어 토큰) 필터링을 거쳐 좀 더 유의미한 문제를 생성할 수 있다  
