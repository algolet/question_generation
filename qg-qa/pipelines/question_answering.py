from datasets import Dataset
import logging
import torch
from .utils_qa import postprocess_qa_predictions
from typing import Union, List, Dict

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class QuestionAnsweringPipeline:
    def __init__(self,
                 tokenizer,
                 model,
                 max_seq_length: int = 384,
                 doc_stride: int = 32,
                 version_2_with_negative: bool = True,
                 n_best_size: int = 1,
                 max_answer_length: int = 30,
                 null_score_diff_threshold: int = 0):
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.question_column_name = "question"
        self.context_column_name = "context"
        self.version_2_with_negative = version_2_with_negative
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.null_score_diff_threshold = null_score_diff_threshold
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        # Validation preprocessing

    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name],
            examples[self.context_column_name],
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # correspnding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

        # region Metrics and Post-processing:

    def post_processing_function(self, examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.version_2_with_negative,
            n_best_size=self.n_best_size,
            max_answer_length=self.max_answer_length,
            null_score_diff_threshold=self.null_score_diff_threshold,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction": v} for k, v in predictions.items()]
        return formatted_predictions

    def __call__(self, input: Union[Dict, List[Dict]], *args, **kwargs):
        if isinstance(input, dict):
            questions = [input["question"]]
            contexts = [input["context"]]
        else:
            questions = [item["question"] for item in input]
            contexts = [item["context"] for item in input]
        ids = [i for i in range(len(questions))]
        examples = {"id": ids, "question": questions, "context": contexts}
        processed_datasets = self.prepare_validation_features(examples)
        inputs = {
            "input_ids": torch.tensor(processed_datasets["input_ids"]),
            "attention_mask": torch.tensor(processed_datasets["attention_mask"]),
        }

        input_ids = inputs["input_ids"].to(self.device)
        input_masks = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            eval_predictions = self.model(input_ids=input_ids,
                                          attention_mask=input_masks)
        examples = Dataset.from_dict(examples)
        flat_examples = []
        for idx in range(len(examples["id"])):
            flat_examples.append({"id": examples["id"][idx],
                                  "question": examples["question"][idx],
                                  "context": examples["context"][idx]})

        flat_processed_datasets = []
        for idx in range(len(processed_datasets["input_ids"])):
            flat_processed_datasets.append({"input_ids": processed_datasets["input_ids"][idx],
                                            "token_type_ids": processed_datasets["token_type_ids"][idx],
                                            "attention_mask": processed_datasets["attention_mask"][idx],
                                            "offset_mapping": processed_datasets["offset_mapping"][idx],
                                            "example_id": processed_datasets["example_id"][idx]})
        post_processed_eval = self.post_processing_function(
            flat_examples,
            flat_processed_datasets,
            (eval_predictions.start_logits.cpu().numpy(), eval_predictions.end_logits.cpu().numpy()),
        )
        res = []
        for item in post_processed_eval:
            if not item["prediction"]:
                res.append(dict())
            else:
                if item["prediction"]["start"] == item["prediction"]["end"]:
                    res.append(dict())
                else:
                    res.append({"answer": item["prediction"]["text"],
                                "start": item["prediction"]["start"],
                                "end": item["prediction"]["end"],
                                "score": item["prediction"]["probability"]})
        if isinstance(input, dict):
            return res[0]
        return res
