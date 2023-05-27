import os
from PIL import Image
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import LayoutXLMTokenizerFast
from transformers import LayoutXLMProcessor

def normalize_box(box, width, height):
    return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]

def encode_example(example, tokenizer, label2idx, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
    words = example['words']
    normalized_word_boxes = example['bbox']
    labels = example['label']

    assert len(words) == len(normalized_word_boxes)

    token_boxes = []
    token_labels = []
    for word, box, label in zip(words, normalized_word_boxes, labels):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
        token_labels.extend([label] * len(word_tokens))

    special_tokens_count = 2 
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
        token_labels = token_labels[: (max_seq_length - special_tokens_count)]

    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    token_labels = [-100] + token_labels + [-100]

    encoding = tokenizer.encode_plus(' '.join(words), padding='max_length', truncation=True, max_length=max_seq_length)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    token_type_ids = encoding["token_type_ids"] if "token_type_ids" in encoding else None

    # Correct padding
    actual_seq_length = len(input_ids) # should be equal to max_seq_length
    padding_length = actual_seq_length - len(token_boxes)
    token_boxes += [pad_token_box] * padding_length
    token_labels += [-100] * padding_length


    if token_type_ids is not None:
        encoding = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'bbox': torch.tensor(token_boxes, dtype=torch.long),
            'labels': torch.tensor(token_labels, dtype=torch.long)
        }
    else:
        encoding = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'bbox': torch.tensor(token_boxes, dtype=torch.long),
            'labels': torch.tensor(token_labels, dtype=torch.long)
        }
    return encoding


class LayoutLMDataset(Dataset):
    def __init__(self, csv_dir, image_dir, tokenizer, label2idx, max_seq_length=512, csv_files=None, is_train=True):
        if csv_files is not None:
            self.csv_files = csv_files
        else:
            self.csv_files = glob.glob(f"{csv_dir}/*.csv")
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_seq_length = max_seq_length
        self.pad_token_box = [0, 0, 0, 0]
        self.is_train = is_train
        
    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        csv_df = pd.read_csv(csv_file, sep="\t")
        
        words = list(csv_df.Object)
        words = [str(w) for w in words]
        
        image_file = os.path.join(self.image_dir, os.path.basename(csv_file).replace(".csv", ".png"))
        image = Image.open(image_file)
        width, height = image.size

        coordinates = csv_df[["xmin", "ymin", "xmax", "ymax"]]
        actual_bboxes = []
        for _, row in coordinates.iterrows():
            xmin, ymin, xmax, ymax = tuple(row)
            actual_bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
            actual_bboxes.append(actual_bbox)

        bboxes = [normalize_box(box, width, height) for box in actual_bboxes]
        labels = [self.label2idx[label] if label in self.label2idx else 0 for label in csv_df['label'].tolist()]
        
        if self.is_train:
            return encode_example({'words': words, 'bbox': bboxes, 'label': labels}, 
                              self.tokenizer, 
                              self.label2idx, 
                              self.max_seq_length, 
                              self.pad_token_box)
        else:
            return encode_example({'words': words, 'bbox': bboxes, 'label': labels}, 
                                self.tokenizer, 
                                self.label2idx, 
                                self.max_seq_length, 
                                self.pad_token_box), image



if __name__ == "__main__":
    from transformers import LayoutLMTokenizer
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    label2idx = {'O': 0, 'SenderVatId': 1, 'ReceiverVatId': 2}
    csv_dir = r"G:\uid\csv"  # replace with your actual path
    image_dir = r"G:\uid\png"  # replace with your actual path    
    dataset = LayoutLMDataset(csv_dir, image_dir, tokenizer, label2idx)
    print(dataset[0])