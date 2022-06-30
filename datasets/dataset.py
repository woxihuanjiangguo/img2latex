import json
import torch
from torch.utils.data import Dataset
import os
from datasets.augmentation import get_image_from_path


def collate_batch(data):
    max_len = max([len(d["label"]) for d in data])
    padded_encoded = [
        d["label"] + (max_len - len(d["label"])) * [99]
        for d in data
    ]
    return {
        "img": torch.stack([d["img"] for d in data], dim=0),
        'label': torch.tensor(padded_encoded)
    }


class AidaDataset(Dataset):
    def __init__(self, mode, config):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.config = config
        filename = mode + '_infolist.txt'
        with open(os.path.join(config['dataset']['root_dir'], 'splits', filename), 'r') as infolist_file:
            self.infolist = infolist_file.readlines()

        with open(os.path.join(config['dataset']['root_dir'], 'token_map.json'), 'r') as token_file:
            self.token2id = json.load(token_file)
        max_id = max(self.token2id.values())
        self.token2id['<START>'] = config['start_id']
        self.token2id['<END>'] = config['end_id']
        self.token2id['<PAD>'] = config['pad_id']
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.num_classes = len(self.token2id)
        assert (
                self.num_classes == config['num_classes'] and
                max_id + 1 == config['start_id'] and
                config['start_id'] + 1 == config['end_id'] and
                config['end_id'] + 1 == config['pad_id']
        )

    def __len__(self):
        return len(self.infolist)

    def __getitem__(self, idx):
        line = self.infolist[idx]
        line_splits = line.strip('\n').split()
        img_path = os.path.join(self.config['dataset']['root_dir'], line_splits.pop(0))
        token_list = line_splits
        img = get_image_from_path(img_path, self.mode, self.config)
        id_list = [self.token2id[x] for x in token_list]
        sample = {'img': img, 'label': [self.token2id['<START>'], *id_list, self.token2id['<END>']]}
        return sample
