from datasets import Dataset
import logging
import tensorflow as tf
from qg.pipelines.utils_qa import postprocess_qa_predictions

logger = logging.getLogger(__name__)


class QuestionAnsweringPipeline:
    def __init__(self,
                 tokenizer,
                 model,
                 max_seq_length: int = 384,
                 doc_stride: int = 32,
                 pad_to_max_length: bool = False,
                 version_2_with_negative: bool = True,
                 n_best_size: int = 1,
                 max_answer_length: int = 30,
                 null_score_diff_threshold: int = 0):
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.pad_to_max_length = pad_to_max_length
        self.question_column_name = "question"
        self.context_column_name = "context"
        self.version_2_with_negative = version_2_with_negative
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.null_score_diff_threshold = null_score_diff_threshold
        self.per_device_eval_batch_size = 128
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) == 0:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        elif len(gpus) == 1:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        elif len(gpus) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            raise ValueError("Cannot find the proper strategy, please check your environment properties.")

    def prepare_validation_features(self, examples):
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]
        tokenized_examples = self.tokenizer(
            examples[self.question_column_name],
            examples[self.context_column_name],
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.pad_to_max_length else False
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    def preprocess(self, examples):
        dataset = examples.map(
            self.prepare_validation_features,
            batched=True
        )
        return dataset

    def post_processing_function(self, examples, features, predictions, stage="eval"):
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
        formatted_predictions = [{"id": k, "prediction": v} for k, v in predictions.items()]
        return formatted_predictions

    def __call__(self, examples, *args, **kwargs):
        logger.info("*** Prediction ***")
        processed_datasets = self.prepare_validation_features(examples)
        eval_inputs = {
            "input_ids": tf.ragged.constant(processed_datasets["input_ids"]).to_tensor(),
            "attention_mask": tf.ragged.constant(processed_datasets["attention_mask"]).to_tensor(),
        }
        with self.strategy.scope():
            eval_predictions = self.model.predict(eval_inputs, batch_size=self.per_device_eval_batch_size)
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
            (eval_predictions.start_logits, eval_predictions.end_logits),
        )
        return post_processed_eval
