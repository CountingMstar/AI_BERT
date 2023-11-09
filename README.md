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
```
   from transformers import BertForQuestionAnswering
   
   model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
```

## Dataset
### 위키피디아
https://en.wikipedia.org/wiki/Main_Page
### activeloop
https://www.activeloop.ai/resources/glossary/arima-models/
### Adrien Beaulieu
https://product.house/100-ai-glossary-terms-explained-to-the-rest-of-us/


```
Context: 'Feature engineering or feature extraction or feature discovery is the process of extracting features (characteristics, properties, attributes) from raw data. Due to deep learning networks, such as convolutional neural networks, that are able to learn features by themselves, domain-specific-based feature engineering has become obsolete for vision and speech processing. Other examples of features in physics include the construction of dimensionless numbers such as Reynolds number in fluid dynamics; then Nusselt number in heat transfer; Archimedes number in sedimentation; construction of first approximations of the solution such as analytical strength of materials solutions in mechanics, etc..'

Question: 'What is large language model?'

Answer: 'A large language model (LLM) is a type of language model notable for its ability to achieve general-purpose language understanding and generation.'
```


학습 데이터셋은 인공지능관련 문맥, 질문, 그리고 응답 이렇게 3가지로 구성이 되어있습니다. 응답(정답) 데이터는 문맥 데이터안에 포함되어 있고, 문맥 데이터의 문장 순서를 바꿔주어 데이터를 증강 하였습니다. 질문 데이터는 주제가 되는 인공지능 용어로 설정했습니다. 위의 예시를 보시면 이해하시기 편하실 겁니다. 총 데이터수는 3300여개로 data폴더에 pickle파일 형태로 저장이 되었습니다.


## How to use?
https://github.com/CountingMstar/AI_BERT/blob/main/MY_AI_BERT_final.ipynb


자세한 모델 학습 및 사용법은 위의 링크에 설명되어있습니다.
감사합니다.




