# AI Tutor BERT
This model is a BERT model fine-tuned on artificial intelligence (AI) related terms and explanations.

With the increasing interest in artificial intelligence, many people are taking AI-related courses and projects. However, as a graduate student in artificial intelligence, it's not common to find useful resources that are easy for AI beginners to understand. Furthermore, personalized lessons tailored to individual levels and fields are often lacking, making it difficult for many people to start learning about artificial intelligence. To address these challenges, our team has created a language model that plays the role of a tutor in the field of AI terminology. Details about the model type, training dataset, and usage are explained below, so please read them carefully and be sure to try it out.

## Model
https://huggingface.co/bert-base-uncased


For the model, I used BERT, which is one of the most famous natural language processing models developed by Google. For more detailed information, please refer to the website mentioned above. To make the question-answering more like a private tutoring experience, I utilized a specialized Question and Answering model within BERT. Here's how you can load it:


```
   from transformers import BertForQuestionAnswering
   
   model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
```


## Dataset
### Wikipedia
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

The training dataset consists of three components: context, questions, and answers, all related to artificial intelligence. The response (correct answer) data is included within the context data, and the sentence order in the context data has been rearranged to augment the dataset. The question data is focused on artificial intelligence terms as the topic. You can refer to the example above for better understanding. In total, there are over 3,300 data points, stored in pickle files in the 'data' folder. The data has been extracted and processed using HTML from sources such as Wikipedia and other websites. The sources are as mentioned above.


## Training and Result
https://github.com/CountingMstar/AI_BERT/blob/main/MY_AI_BERT_final.ipynb


The training process involves loading data from the 'data' folder and utilizing the BERT Question and Answering model. Detailed instructions for model training and usage can be found in the link provided above.


```
N_EPOCHS = 10
optim = AdamW(model.parameters(), lr=5e-5)
```


I used 10 epochs for training, and I employed the Adam optimizer with a learning rate of 5e-5.


<img src="https://github.com/CountingMstar/AI_BERT/assets/90711707/72142ff8-f5c8-47ea-9f19-1e6abb4072cd" width="500" height="400"/>
<img src="https://github.com/CountingMstar/AI_BERT/assets/90711707/2dd78573-34eb-4ce9-ad4d-2237fc7a5b1e" width="500" height="400"/>


The results, as shown in the graphs above, indicate that, at the last epoch, the loss is 6.917126256477786, and the accuracy is 0.9819078947368421, demonstrating that the model has been trained quite effectively.


## How to use?
https://github.com/CountingMstar/AI_BERT/blob/main/MY_AI_BERT_final.ipynb


You can load the trained model through the training process described above and use it as needed.


Thank you.


---
# AI Tutor BERT (인공지능 과외 선생님 BERT)
이 모델은 인공지능(AI) 관련 용어 및 설명을 파인튜닝(fine-tuning)한 BERT 모델입니다.


최근 인공지능에 관한 관심이 높아지면서 많은 사람이 인공지능 관련 수업 및 프로젝트를 진행하고 있습니다. 그러나 인공지능 관련 대학원생으로서 이러한 수요에 비해 인공지능 초보자들이 잘 알아들을 수 있는 유용한 자료는 흔치 않습니다. 더불어 각자의 수준과 분야에 개인화된 강의 또한 부족한 상황이어서 많은 사람들이 인공지능 학습을 시작하기 어려워하고 있습니다. 이러한 문제를 해결하고자, 저희 팀은 인공지능 용어 도메인에서 과외 선생님 역할을 하는 언어모델을 만들었습니다. 모델의 종류, 학습 데이터셋, 사용법 등이 아래에 설명되어 있으니 자세히 읽어보시고, 꼭 사용해 보시기 바랍니다.


## Model
https://huggingface.co/bert-base-uncased


모델의 경우 자연어 처리 모델 중 가장 유명한 Google에서 개발한 BERT를 사용했습니다. 자세한 설명은 위 사이트를 참고하시기 바랍니다. 질의응답이 주인 과외 선생님답게, BERT 중에서도 질의응답에 특화된 Question and Answering 모델을 사용하였습니다. 불러오는 법은 다음과 같습니다.
```
   from transformers import BertForQuestionAnswering
   
   model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
```

## Dataset
### Wikipedia
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


학습 데이터셋은 인공지능 관련 문맥, 질문, 그리고 응답 이렇게 3가지로 구성이 되어있습니다. 응답(정답) 데이터는 문맥 데이터 안에 포함되어 있고, 문맥 데이터의 문장 순서를 바꿔주어 데이터를 증강하였습니다. 질문 데이터는 주제가 되는 인공지능 용어로 설정했습니다. 위의 예시를 보시면 이해하시기 편하실 겁니다. 총 데이터 수는 3300여 개로 data 폴더에 pickle 파일 형태로 저장되어 있고, 데이터는 Wikipedia 및 다른 사이트들을 에서 html을 이용하여 추출 및 가공하여 제작하였습니다. 해당 출처는 위와 같습니다. 


## Training and Result
https://github.com/CountingMstar/AI_BERT/blob/main/MY_AI_BERT_final.ipynb


학습 방식은 data 폴더의 데이터와 BERT Question and Answering 모델을 불어와 진행됩니다. 자세한 모델 학습 및 사용법은 위의 링크에 설명되어 있습니다.

```
N_EPOCHS = 10
optim = AdamW(model.parameters(), lr=5e-5)
```


에포크(epoch)는 10을 사용했으며, 아담 옵티마이져와 러닝레이트는 5e-5를 사용했습니다.



<img src="https://github.com/CountingMstar/AI_BERT/assets/90711707/72142ff8-f5c8-47ea-9f19-1e6abb4072cd" width="500" height="400"/>
<img src="https://github.com/CountingMstar/AI_BERT/assets/90711707/2dd78573-34eb-4ce9-ad4d-2237fc7a5b1e" width="500" height="400"/>


결과는 위 그래프들과 같이 마지막 에포크 기준 loss = 6.917126256477786, accuracy = 0.9819078947368421로 상당히 학습이 잘 된 모습을 보여줍니다.



## How to use?


```
model = torch.load("./models/AI_BERT_final_10.pth")
```

위 학습 과정을 통해 학습된 모델을 불러와 사용하시면 됩니다.


<img src="https://github.com/CountingMstar/AI_BERT/assets/90711707/45afcd24-7ef9-4149-85d4-2236e23fbf69" width="800" height="400"/>
https://huggingface.co/spaces/pseudolab/AI_Tutor_BERT


또는 위 그림과 같이 인공지능관련 지문(문맥)과 용어 관련 질문을 입력해주고 Submit을 눌러주면, 오른쪽에 해당 용어에 대한 설명 답변이 나옵니다. 


감사합니다.






