import logging
from .question_generation import QuestionGenerationPipeline
from .question_answering import QuestionAnsweringPipeline
from typing import Optional, Union
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = {
    "question-generation": {
        "impl": QuestionGenerationPipeline,
        "pt": AutoModelForSeq2SeqLM,
        "default": {
            "model": "algolet/mt5-base-chinese-qg"
        }
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "pt": AutoModelForQuestionAnswering,
        "default": {
            "model": "luhua/chinese_pretrain_mrc_macbert_large"
        }
    }
}


def pipeline(
        task: str,
        model: Optional = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        device: str = "cpu"
):
    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model"
            "being specified."
            "Please provide a task class or a model"
        )

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model "
            "as the provided tokenizer may not be compatible with the default model. "
            "Please provide a PreTrainedModel class or a path/identifier to a pretrained model when providing tokenizer."
        )

    if task not in SUPPORTED_TASKS:
        raise KeyError(f"Unknown task {task}, available tasks are {list(SUPPORTED_TASKS.keys())}")
    targeted_task = SUPPORTED_TASKS[task]
    pipeline_class = targeted_task["impl"]

    if model is None:
        model = targeted_task["default"]["model"]

    model_name = model if isinstance(model, str) else None
    model_classes = targeted_task["pt"]

    if tokenizer is None:
        if isinstance(model_name, str):
            tokenizer = model_name
        else:
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if isinstance(model, str):
        model = model_classes.from_pretrained(model)
    return pipeline_class(model=model, tokenizer=tokenizer, device=device)
