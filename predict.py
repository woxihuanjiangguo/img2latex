from model.maths_net import MathsNet
import torch
from utils.utils import get_yaml_data, get_pic_from_path, post_process_seq, get_json_data

use_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

'''
change your path to the test pic here
'''
your_path_to_pic = './pics/1.png'

if __name__ == '__main__':
    config = get_yaml_data('./configs/MathsNet.yaml')
    token_map = get_json_data('./configs/token_map.json')
    token_map = {
        v: k for k, v in token_map.items()
    }

    # greedy
    net = MathsNet(
        config=config,
        training=False,
        device=device,
    )

    if use_gpu:
        net.cuda()

    # predict
    imgs = get_pic_from_path(your_path_to_pic, config, device=device)
    encoder_feats = net.encoder(imgs)
    encoder_feats = encoder_feats.contiguous()
    output, _ = net.decoder.predict(encoder_feats)

    # print result
    for result in post_process_seq(
            output, start_id=config['start_id'], end_id=config['end_id'], token_map=token_map
    ):
        print(result)
