
<h3 align="center">
    <p>Chinese Question Generation and Question Answering Pipeline</p>
</h3>
<h4 align="center">
    <p>
        <a href="https://github.com/algolet/question_generation/blob/main/README.md">中文说明</a> |
        <b>English</b>
    <p>
</h4>

## Online demos
You can test the model directly on https://www.algolet.com/applications/qg

## Ouick tour
To immediately use models on given inputs, we provide `question_generation` and `question_answering` `pipeline` API
Pipelines group together a pretrained model with the preprocessing that was used during that model's training. 

Here is how to quickly use a pipeline to generate questions
``` python
>>> from question_generation import pipeline

# Allocate a pipeline for question-generation
>>> qg = pipeline("question-generation") 
# for single text         
>>> qg("在一个寒冷的冬天，赶集完回家的农夫在路边发现了一条冻僵了的蛇。他很可怜蛇，就把它放在怀里。
       "当他身上的热气把蛇温暖以后，蛇很快苏醒了，露出了残忍的本性，给了农夫致命的伤害——咬了农夫一口。
       "农夫临死之前说：“我竟然救了一条可怜的毒蛇，就应该受到这种报应啊！”")
['在寒冷的冬天,农夫在哪里发现了一条可怜的蛇?', '农夫是如何看待蛇的?', '当农夫遇到蛇时,他做了什么?']

# for batch input
>>> texts = ["在一个寒冷的冬天，赶集完回家的农夫在路边发现了一条冻僵了的蛇。他很可怜蛇，就把它放在怀里。
       "当他身上的热气把蛇温暖以后，蛇很快苏醒了，露出了残忍的本性，给了农夫致命的伤害——咬了农夫一口。
       "农夫临死之前说：“我竟然救了一条可怜的毒蛇，就应该受到这种报应啊！”"]
>>> qg(texts)
[['在寒冷的冬天,农夫在哪里发现了一条可怜的蛇?', '农夫是如何看待蛇的?', '当农夫遇到蛇时,他做了什么?']]
``` 
To use model of your own or any of the fine-tuned model on question-generation. Here is the PyTorch version:
``` python   
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> tokenizer = AutoTokenizer.from_pretrained("algolet/mt5-base-chinese-qg")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("algolet/mt5-base-chinese-qg")
>>> pipe = pipeline("question-generation", model=model, tokenizer=tokenizer)
``` 


 