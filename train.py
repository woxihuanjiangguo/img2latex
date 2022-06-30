import time
from model.maths_net import MathsNet
from datasets.dataset import AidaDataset, collate_batch
from torch.utils.data import DataLoader
from utils.logger import Logger
import torch
import torch.optim as optim
from utils.checkpoint import Checkpoint
from utils.utils import get_yaml_data, post_process_metrics

use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

if __name__ == '__main__':
    logger = Logger()
    config = get_yaml_data('./configs/MathsNet.yaml')
    checkpoint = Checkpoint(checkpoint_period=config['checkpoint_period'], root_dir=config['checkpoint_save_path'])
    trainDataset = AidaDataset('train', config)
    trainLoader = DataLoader(
        dataset=trainDataset,
        shuffle=True,
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        collate_fn=collate_batch
    )

    valDataset = AidaDataset('val', config)
    valLoader = DataLoader(
        dataset=valDataset,
        shuffle=False,
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        collate_fn=collate_batch
    )

    net = MathsNet(
        config=config,
        training=True,
        device=device,
        use_beam=True,
        beam_width=5
    )

    if use_gpu:
        net.cuda(device=device)

    optimizer = optim.Adadelta(
        net.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )

    # loop
    for i in range(config['epoch']):
        train_metrics = {
            'loss_verbose': 0,
            'err': [],
            'acc': []
        }
        val_metrics = {
            'loss_verbose': 0,
            'err': [],
            'acc': []
        }
        start = time.time()

        # train
        for sample in trainLoader:
            sample_gpu = {
                k: v.to(device) for k, v in sample.items()
            }
            net.zero_grad()
            loss = net(sample_gpu)['loss']
            loss.backward()
            optimizer.step()

        # metrics
        # train
        for sample in trainLoader:
            with torch.no_grad():
                sample_gpu = {
                    k: v.to(device) for k, v in sample.items()
                }
                ret_dict = net(sample_gpu)
                loss = ret_dict['loss']
                post_process_metrics(loss, ret_dict['output']['pred_rec'], sample['label'],
                                     end_id=config['end_id'],
                                     metrics_dict=train_metrics)

        # val
        for sample in valLoader:
            with torch.no_grad():
                sample_gpu = {
                    k: v.to(device) for k, v in sample.items()
                }
                ret_dict = net(sample_gpu)
                loss = ret_dict['loss']
                post_process_metrics(loss, ret_dict['output']['pred_rec'], sample['label'],
                                     end_id=config['end_id'],
                                     metrics_dict=val_metrics)

        logger.log_train(
            epoch=i + 1,
            train_loss=train_metrics['loss_verbose'] / len(trainLoader),
            train_acc=sum(train_metrics['acc']) / len(trainLoader),
            train_err=sum(train_metrics['err']) / len(trainLoader),
            val_loss=val_metrics['loss_verbose'] / len(valLoader),
            val_acc=sum(val_metrics['acc']) / len(valLoader),
            val_err=sum(val_metrics['err']) / len(valLoader),
            time=time.time() - start,
        )
        checkpoint.save_model(net.state_dict(), epoch_cnt=i + 1)
