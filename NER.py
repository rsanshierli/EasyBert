import os
import json
import copy
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from NER.callback.progressbar import ProgressBar
from NER.tools.common import seed_everything
from NER.models.transformers import BertConfig
from NER.models.bert_for_ner import BertCrfForNer
from NER.processors.utils_ner import CNerTokenizer, get_entities
from NER.processors.ner_seq import ner_processors as processors
from NER.processors.ner_seq import collate_fn


class Config(object):
    def __init__(self):
        self.model_type = 'bert'
        self.model_name_or_path = './NER/prev_trained_model/bert-base'
        self.task_name = 'cluener'
        self.do_predict = True
        self.do_lower_case = True
        self.data_dir = './NER/datasets/cluener'
        self.train_max_seq_length = 128
        self.eval_max_seq_length = 512
        self.per_gpu_train_batch_size = 128
        self.per_gpu_eval_batch_size = 128
        self.learning_rate = 3e-5
        self.crf_learning_rate = 1e-3
        self.num_train_epochs = 4.0
        self.save_steps = 448
        self.output_dir = './NER/outputs/cluener_output/bert'
        self.overwrite_output_dir = True
        self.seed = 42
        self.local_rank = -1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))
    return features


def load_and_cache_examples(text, task, tokenizer, data_type='train'):
    processor = processors[task]()
    label_list = processor.get_labels()

    examples = processor.get_predict_text(text)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            label_list=label_list,
                                            max_seq_length=512,
                                            cls_token_at_end=False,
                                            pad_on_left=False,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=0,
                                            sep_token=tokenizer.sep_token,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0,
                                            )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def predict(args, text, id2label, model, tokenizer):

    test_dataset = load_and_cache_examples(text, args.task_name, tokenizer, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "labels": None, 'input_lens': batch[4],
                      "token_type_ids": batch[2]}
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, id2label, 'bios')
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([id2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)

    # test_text = []
    # with open(os.path.join(args.data_dir, "test.json"), 'r', encoding='utf-8') as fr:
    #     for line in fr:
    #         test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(text, results):
        json_d = {}
        json_d['text'] = x
        json_d['label'] = {}
        entities = y['entities']
        words = list(x)
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)

    return test_submit


def pos_predict(text):
    args = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    # label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = BertConfig, BertCrfForNer, CNerTokenizer
    config = config_class.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=None)

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(device)
            result = predict(args, text, id2label, model, tokenizer)
            for i in result:
                print(i)


if __name__ == "__main__":
    text=["四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。",
          "尼日利亚海军发言人当天在阿布贾向尼日利亚通讯社证实了这一消息。",
          "销售冠军：辐射3-Bethesda",
          "所以大多数人都是从巴厘岛南部开始环岛之旅。",
          "备受瞩目的动作及冒险类大作《迷失》在其英文版上市之初就受到了全球玩家的大力追捧。",
          "filippagowski：14岁时我感觉自己像梵高",
          "央视新址文化中心外立面受损严重",
          "单看这张彩票，税前总奖金为5063992元。本张票面缩水后阿森纳的结果全部为0，斯图加特全部为1，",
          "你会和星级厨师一道先从巴塞罗那市中心兰布拉大道的laboqueria市场的开始挑选食材，",
          "波特与凤凰社》的率队下更加红火。乘着7月的上升气流，《发胶》、《辛普森一家》、《谍影憧憧ⅲ》"]
    pos_predict(text)
