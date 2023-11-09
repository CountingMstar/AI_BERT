# AI Question and Answering BERT


## Model


## Dataset


## How to use?


---
# AI Tutor BERT (인공지능 과외선생님 BERT)
이 모델은 인공지능(AI) 관련 용어 및 설명을 파인튜닝(fine-tuning)한 BERT모델입니다.


최근 인공지능에 대한 관심이 높아지면서 많은 사람들이 인공지능관련 수업 및 프로젝트를 진행하고 있습니다. 그러나 인공지능관련 대학원생으로서 이러한 수요에 비해 인공지능 초보자들이 잘 알아들을 수 있는 유용한 자료는 흔치 않습니다. 더불어 각자의 수준과 분야에 개인화된 강의 또한 부족한 상황이어서 많은 사람들이 인공지능 학습을 시작하기를 어려워하고 있습니다. 이러한 문제를 해결하고자, 저희팀은 인공지능 용어 도메인에서 과외선생님역할을 하는 언어모델을 만들었습니다. 모델의 종류, 학습 데이터셋, 사용법 등이 아래에 설명되어 있으니 자세히 읽어보시고, 꼭 사용해보시기 바랍니다.


## Model
https://huggingface.co/bert-base-uncased


모델의 경우 자연어 처리 모델중 가장 유명한 Google에서 개발한 BERT를 사용했습니다. 자세한 설명은 위 사이트를 참고하시기 바랍니다. 질의응답이 주인 과외선생님 답게, BERT 중에서도 질의응답에 특화된 Question and Answering 모델을 사용하였습니다. 불러오는법은 다음과 같습니다.

   from transformers import BertForQuestionAnswering

   model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

## Dataset


## How to use?

