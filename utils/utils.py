import yaml
import torch
from datasets.augmentation import crop_contours
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import json
import editdistance


def get_yaml_data(path):
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def get_json_data(path):
    with open(path, 'r') as file:
        return json.load(file)


def sever_tokens(token_list, end_id):
    result = []
    for token in token_list:
        result.append(token)
        if token == end_id:
            break
    return result


def post_process_metrics(loss, outputs, labels, end_id, metrics_dict):
    severed_outputs = []
    severed_labels = []
    outputs_array = outputs.cpu().numpy()
    labels_array = labels.cpu().numpy()
    for output in outputs_array:
        severed_outputs.append(sever_tokens(output, end_id))
    for label in labels_array:
        severed_labels.append(sever_tokens(label, end_id))
    metrics_dict['loss_verbose'] += loss.item()
    metrics_dict['err'].append(word_error_rate(severed_outputs, severed_labels))
    metrics_dict['acc'].append(sentence_acc(severed_outputs, severed_labels))


# predict routines
def post_process_seq(sequences, start_id, end_id, token_map):
    result_strings = []
    for sequence in sequences:
        result_str = ''
        for token in sequence:
            token = token.item()
            if token is start_id:
                continue
            elif token is end_id:
                break
            else:
                result_str += token_map[token]
        result_strings.append(result_str)
    return result_strings


def get_pic_from_path(path, cfg, device):
    imgs = []
    cropped_list, demo_pic = crop_contours(path)
    demo_pic.show()
    for cropped_np_array in cropped_list:
        img = Image.fromarray(cropped_np_array).convert('L')
        img = ImageOps.invert(img.point(lambda x: 255 if x > 130 else 0))
        img = np.asarray(img)
        img = torch.FloatTensor(img)
        img = img[None, :, :]
        img = transforms.Resize(size=(cfg['dataset']['height'], cfg['dataset']['width']))(img)
        img /= 255.0
        imgs.append(np.array(img))
    imgs = np.array(imgs)

    return torch.tensor(imgs, device=device)


# metrics
def word_error_rate(predicted_outputs, ground_truths):
    sum_wer = 0.0
    for output, ground_truth in zip(predicted_outputs, ground_truths):
        distance = editdistance.eval(output, ground_truth)
        length = max(len(output), len(ground_truth))
        sum_wer += (distance / length)
    return sum_wer / len(predicted_outputs)


def sentence_acc(predicted_outputs, ground_truths):
    correct_sentences = 0
    for output, ground_truth in zip(predicted_outputs, ground_truths):
        if np.array_equal(output, ground_truth):
            correct_sentences += 1
    return correct_sentences / len(predicted_outputs)
