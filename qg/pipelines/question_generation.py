import logging
from typing import List, Dict
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class QuestionGenerationPipeline:
    def __init__(self,
                 tokenizer,
                 model,
                 max_source_length: int = 512,
                 max_target_length: int = 200):
        self.tokenizer = tokenizer
        self.model = model
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = "question generation: "

    def preprocess(self, examples: List[str]) -> Dict[str, Tensor]:
        added_prefix = ["question generation: " + example for example in examples]
        inputs = self.tokenizer(added_prefix, return_tensors='pt', padding=True, truncation=True,
                                max_length=self.max_source_length)
        return inputs

    def post_processing_function(self, outs, ):
        questions = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
        separated_questions = []
        for each_batch_questions in questions:
            question = [q.strip() for q in each_batch_questions.split("<sep>") if q.strip() != ""]
            if len(each_batch_questions) > 5 and each_batch_questions[-5:] != "<sep>":  # 一个完整的问题需以<sep>结尾
                question = question[:-1]
            separated_questions.append(question)

        return separated_questions

    def __call__(self, paragraphs: List[str], *args, **kwargs):
        logger.info("*** Prediction ***")
        inputs = self.preprocess(paragraphs)
        with torch.no_grad():
            outs = self.model.generate(input_ids=inputs["input_ids"],
                                       attention_mask=inputs["attention_mask"],
                                       max_length=self.max_target_length,
                                       no_repeat_ngram_size=4,
                                       num_beams=4
                                       )
        post_processed = self.post_processing_function(outs)

        return post_processed


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/mnt/nfs/algo/models/mt5-small-zh-qg")
    model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/nfs/algo/models/mt5-small-zh-qg")
    pipe = QuestionGenerationPipeline(tokenizer=tokenizer, model=model)
    texts = ["格里戈里·米哈伊洛维奇·科津采夫（，拉丁化：Grigori Mikhailovich Kozintsev；，苏联电影导演，善用无声、蒙太奇手法，风格偏向表现主义。格里戈里·科津采夫在1905年3月22日生于（俄罗斯）罗曼诺夫王朝基辅，在圣彼得堡的帝国艺术学院就读，1921年开始制作电影。他执导的改编电影，包括威廉·莎士比亚的《李尔王》、《哈姆雷特》、米格尔·德·塞万提斯的《唐吉诃德》，都颇有名声。其中1964年的《哈姆雷特》是他的执导代表作，主演的因诺肯季·斯莫克图诺夫斯基凭此电影声名大噪。1973年5月11日，格里戈里·科津采夫在列宁格勒逝世。"]
    question_res = pipe(texts)
    print(question_res)
