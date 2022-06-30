from model.maths_net import MathsNet
from datasets.dataset import AidaDataset, collate_batch
from torch.utils.data import DataLoader
from utils.logger import Logger
import torch
from utils.utils import get_yaml_data, post_process_metrics

use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

if __name__ == '__main__':
    logger = Logger()
    config = get_yaml_data('./configs/MathsNet.yaml')
    testDataset = AidaDataset('test', config)
    testLoader = DataLoader(
        dataset=testDataset,
        shuffle=True,
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        collate_fn=collate_batch
    )

    net = MathsNet(
        config=config,
        training=False,
        device=device,
    )

    if use_gpu:
        net.cuda(device=device)

    test_metrics = {
        'loss_verbose': 0,
        'err': [],
        'acc': []
    }
    # test
    for sample in testLoader:
        with torch.no_grad():
            sample_gpu = {
                k: v.to(device) for k, v in sample.items()
            }
            ret_dict = net(sample_gpu)
            loss = ret_dict['loss']
            post_process_metrics(loss, ret_dict['output']['pred_rec'], sample['label'],
                                 end_id=config['end_id'],
                                 metrics_dict=test_metrics)

    logger.log_test_metrics(
        loss=test_metrics['loss_verbose'] / len(testLoader),
        err=sum(test_metrics['err']) / len(testLoader),
        acc=sum(test_metrics['acc']) / len(testLoader)
    )
