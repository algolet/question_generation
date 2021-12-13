import logging
from typing import Union, List, Dict
from torch import Tensor
import torch

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

logger = logging.getLogger(__name__)


class QuestionGenerationPipeline:
    def __init__(self,
                 tokenizer,
                 model,
                 max_source_length: int = 512,
                 max_target_length: int = 200):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = "question generation: "

    def preprocess(self, examples: List[str]) -> Dict[str, Tensor]:
        added_prefix = ["question generation: " + example for example in examples]
        inputs = self.tokenizer(added_prefix,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
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

    def __call__(self, input: Union[str, List[str]], *args, **kwargs):
        logger.info("*** Prediction ***")
        if isinstance(input, str):
            examples = [input]
        else:
            examples = input
        examples = self.preprocess(examples)
        input_ids = examples["input_ids"].to(self.device)
        input_mask = examples["attention_mask"].to(self.device)

        with torch.no_grad():
            outs = self.model.generate(input_ids=input_ids,
                                       attention_mask=input_mask,
                                       max_length=self.max_target_length,
                                       no_repeat_ngram_size=4,
                                       num_beams=4)

        post_processed = self.post_processing_function(outs)
        if isinstance(input, str):
            post_processed = post_processed[0]
        return post_processed
