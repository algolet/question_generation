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




