import os
import json
import random

vocab_index = 90

'''
change your path to the dataset here
'''
root_dir = 'D:/pyprojects/AidaMaths'

def write_json_item(json_item, file, batch_index, vocab:dict):
    filename = os.path.join('batch_{}'.format(batch_index), 'background_images', json_item['filename'])
    tokens = json_item['image_data']['full_latex_chars']
    for token in tokens:
        if token not in vocab.keys():
            global vocab_index
            vocab_index += 1
            vocab[token] = vocab_index
    line_ar = [filename]
    line_ar.extend(tokens)
    print(' '.join(line_ar), file=file)


if __name__ == '__main__':
    with open(os.path.join(root_dir, 'extras/visible_char_map.json'), 'r') as map_file:
        vocab = json.load(map_file)
    vocab = {k: v - 1 for k, v in vocab.items()}
    vocab_index = max(vocab.values())
    splits_path = os.path.join(root_dir, 'splits')

    train_infolist_file = open(os.path.join(splits_path, 'train_infolist.txt'), 'w')
    val_infolist_file = open(os.path.join(splits_path, 'val_infolist.txt'), 'w')
    test_infolist_file = open(os.path.join(splits_path, 'test_infolist.txt'), 'w')

    for batch_index in range(1, 11):
        print('start batch{}'.format(batch_index))
        batch_path = os.path.join(root_dir, 'batch_{}'.format(batch_index))
        json_path = os.path.join(batch_path, 'JSON', 'kaggle_data_{}.json'.format(batch_index))
        img_dir_path = os.path.join(batch_path, 'background_images')
        with open(json_path, 'r') as file:
            json_list = json.load(file)
            random.shuffle(json_list)
            for json_index in range(len(json_list)):
                json_item = json_list[json_index]
                if json_index < 6400:
                    write_json_item(json_item, train_infolist_file, batch_index, vocab)
                elif json_index < 8000:
                    write_json_item(json_item, val_infolist_file, batch_index, vocab)
                else:
                    write_json_item(json_item, test_infolist_file, batch_index, vocab)
        print('finish batch{}'.format(batch_index))

    # two copies of token map
    with open(os.path.join(root_dir, 'token_map.json'), 'w') as new_file:
        json.dump(vocab, new_file)

    with open('../configs/token_map.json', 'w') as config_file:
        json.dump(vocab, config_file)