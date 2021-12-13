<h3 align="center">
    <p>基于mt5的中文问题生成任务</p>
</h3>
<h4 align="center">
    <p>
        <b>中文说明</b> |
        <a href="https://github.com/algolet/question_generation/blob/main/README_en.md">English</a>
    <p>
</h4>
基于预训练模型mt5精调的问题生成模型

## 在线测试
可以直接在线使用我们的模型 https://www.algolet.com/applications/qg

## 使用说明
我们提供了`question_generation` 和 `question_answering`的`pipeline` API，通过调用对应的pipeline,可以轻松实现相应任务

下面是如何使用问题生成pipepline
``` python
>>> from question_generation import pipeline

# Allocate a pipeline for question-generation
>>> qg = pipeline("question-generation") 
# for single text         
>>> qg("在一个寒冷的冬天，赶集完回家的农夫在路边发现了一条冻僵了的蛇。他很可怜蛇，就把它放在怀里。当他身上的热气把蛇温暖以后，蛇很快苏醒了，露出了残忍的本性，给了农夫致命的伤害——咬了农夫一口。农夫临死之前说：“我竟然救了一条可怜的毒蛇，就应该受到这种报应啊！”")
['在寒冷的冬天,农夫在哪里发现了一条可怜的蛇?', '农夫是如何看待蛇的?', '当农夫遇到蛇时,他做了什么?']

# for batch input
>>> texts = [""在一个寒冷的冬天，赶集完回家的农夫在路边发现了一条冻僵了的蛇。他很可怜蛇，就把它放在怀里。当他身上的热气把蛇温暖以后，蛇很快苏醒了，露出了残忍的本性，给了农夫致命的伤害——咬了农夫一口。农夫临死之前说：“我竟然救了一条可怜的毒蛇，就应该受到这种报应啊！”""]
>>> qg(texts)
[['在寒冷的冬天,农夫在哪里发现了一条可怜的蛇?', '农夫是如何看待蛇的?', '当农夫遇到蛇时,他做了什么?']]
``` 
可以使用你自己训练的模型，或者下载huggingface hub中已经微调好的模型. PyTorch版本的使用方式如下:
``` python   
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> tokenizer = AutoTokenizer.from_pretrained("algolet/mt5-base-chinese-qg")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("algolet/mt5-base-chinese-qg")
>>> pipe = pipeline("question-generation", model=model, tokenizer=tokenizer)
```    
同时，我们也提供的问答pipeline,可以与问题生成模块集成，产生问题自动生成与回答的应用。

下面是如何使用问答pipeline
``` python 
>>> from question_generation import pipeline 

# Allocate a pipeline for question-generation
>>> qa = pipeline("question-answering") 
>>> text = ""在一个寒冷的冬天，赶集完回家的农夫在路边发现了一条冻僵了的蛇。他很可怜蛇，就把它放在怀里。当他身上的热气把蛇温暖以后，蛇很快苏醒了，露出了残忍的本性，给了农夫致命的伤害——咬了农夫一口。农夫临死之前说：“我竟然救了一条可怜的毒蛇，就应该受到这种报应啊！”""
# for single qa input
>>> question_answerer({
...     'question': '在寒冷的冬天,农夫在哪里发现了一条可怜的蛇?',
...     'context': text
... })
{'answer': '路边', 'start': 18, 'end': 20, 'score': 1.0} 

# for batch qa inputs
>>> question_answerer([
...    {
...     'question': '在寒冷的冬天,农夫在哪里发现了一条可怜的蛇?',
...     'context': text
...     },
...    {
...     'question': '农夫是如何看待蛇的?',
...     'context': text
...     },
...    {
...     'question': '当农夫遇到蛇时,他做了什么?',
...     'context': text
...     }])
[{'answer': '路边', 'start': 18, 'end': 20, 'score': 1.0},
 {'answer': '我竟然救了一条可怜的毒蛇，就应该受到这种报应',
  'start': 102,
  'end': 124,
  'score': 0.9996},
 {'answer': '放在怀里', 'start': 40, 'end': 44, 'score': 0.9995}]
```   
qa的返回中，`answer`为答案，`start`和`end`分别为答案在原文中的开始位置和结束位置

## 安装说明
```bash
pip install question_generation
```
            
## 模型训练
你可以训练自己的问题生成模型和问答模型

#### 问题生成模型训练
##### 训练数据格式
``` python 
>>> train.json
{"data": [{"source_text": "对于某些物理情况，不可能将力的形成归因于势的梯度。这通常是由于宏观物理的考虑，屈服力产生于微观状态的宏观统计平均值。例如，摩擦是由原子间大量静电势的梯度引起的，但表现为独立于任何宏观位置矢量的力模型。非保守力除摩擦力外，还包括其他接触力、拉力、压缩力和阻力。然而，对于任何足够详细的描述，所有这些力都是保守力的结果，因为每一个宏观力都是微观势梯度的净结果。",
           "target_text": "拉力、压缩和拉力是什么力?{sep_token}静电梯度电势会产生什么?{sep_token}为什么这些力是无法建模的呢?"}
          {"source_text": "绿宝石失窃案 （法语： Les Bijoux de la Castafiore ；英语： The Castafiore Emerald ）是丁丁历险记的第21部作品。作者是比利时漫画家埃尔热。本作与之前的丁丁历险记有著很大的不同，丁丁首次进行没有离开自己家的冒险，同时故事中没有明显的反派角色，充满了喜剧色彩。丁丁和船长原本在城堡悠闲度假，却因歌后突然造访而弄得鸡飞狗跳；媒体对歌后的行踪极度关注，穷追猛打；歌后一颗珍贵的绿宝石又突然失踪，引起了一波接一波的疑团，究竟谁的嫌疑最大？是船长刚刚收留的一伙吉卜赛人？是偷偷混入记者群中的神秘男子？是歌后的贴身女仆？还是行迹鬼祟的钢琴师？"，
           "target_text": "故事中引起众多谜团的原因是？{sep_token}此部作品与以往不同的地方在于哪里？{sep_token}丁丁和船长的悠闲假期因何被打乱？{sep_token}《绿宝石失窃案》是《丁丁历险记》系列的第几部？{sep_token}《绿宝石失窃案》的作者是谁？"}
          ...
          ]}
``` 
##### 训练配置文件
``` python 
>>> qg_config.json  
{
  "model_name_or_path": "google/mt5-small",
  "tokenizer_name": "google/mt5-small",
  "text_column": "source_text",
  "summary_column": "target_text",
  "train_file": "data/train.json",
  "validation_file": "data/dev.json",
  "output_dir": "data/qg",
  "model_type": "mt5",
  "overwrite_output_dir": true,
  "do_train": true,
  "do_eval": true,
  "source_prefix": "question generation: ",
  "predict_with_generate": true,
  "per_device_train_batch_size": 8,
  "per_device_eval_batch_size": 8,
  "gradient_accumulation_steps": 32,
  "learning_rate": 1e-3,
  "num_train_epochs": 4,
  "max_source_length": 512,
  "max_target_length": 200,
  "logging_steps": 100,
  "seed": 42
}
```   
##### 启动训练
CUDA_VISIBLE_DEVICES=0 python run_qg.py qg_config.json

#### 问答模型训练
##### 训练数据格式
与squad_2的数据格式一致
``` python 
>>> train.json
{'version': 2.0,
 'data': [{'id': 'c398789b7375e0ce7eac86f2b18c3808',
           'question': '隐藏式行车记录仪哪个牌子好',
           'context': '推荐使用360行车记录仪。行车记录仪的好坏，取决于行车记录仪的摄像头配置，配置越高越好，再就是性价比。 行车记录仪配置需要1296p超高清摄像头比较好，这样录制视频清晰度高。再就是价格，性价比高也是可以值得考虑的。 360行车记录仪我使用了一段时间 ，觉得360行车记录仪比较好录得广角比较大，并且便宜实惠 ，价格才299，在360商城可以买到。可以参考对比下。',
           'answers': {'answer_start': [4], 'text': ['360行车记录仪']}}]}
``` 
##### 训练配置文件
``` python 
>>> qg_config.json  
{
  "model_name_or_path": "bert-base-chinese",
  "tokenizer_name": "bert-base-chinese",
  "train_file": "data/train.json",
  "validation_file": "data/dev.json",
  "output_dir": "data/qa",
  "per_device_train_batch_size": 8,
  "per_device_eval_batch_size": 8,
  "gradient_accumulation_steps": 32,
  "overwrite_output_dir": true,
  "do_train": true,
  "do_eval": true,
  "max_answer_length": 200
}
```   
##### 启动训练
CUDA_VISIBLE_DEVICES=0 python run_qa.py qa_config.json



 











