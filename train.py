import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as DocDQA_dataset_module
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')



    # build model architecture, then print to console
    model_class = config.init_obj('arch', module_arch)
    tokenizer = model_class.get_tokenizer()
    model = model_class.get_model()
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # set to cpu mode
    # device = torch.device("cpu")
    # device_ids = [0]

    model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
        
    print('model loaded')


    if config['trainer']['is_testing']:
        print('Do testing')
        test_data_obj = config.init_obj('test_data_loader', DocDQA_dataset_module)
        test_data_obj.set_tokenizer(tokenizer)
        test_data = test_data_obj.get_data_set()

        args = config['trainer']
        trainer = Trainer(args, model, logger, device, device_ids, test_data_obj.get_words_list())

        # now it is for testing
        trainer.train(args, test_data, test_data, tokenizer)

    else:
        print('Do training')


        # setup data_loader instances, including preprocess
        train_data_obj = config.init_obj('train_data_loader', DocDQA_dataset_module)
        train_data_obj.set_tokenizer(tokenizer) # must do this step
        train_data = train_data_obj.get_data_set()

        eval_data_obj = config.init_obj('eval_data_loader', DocDQA_dataset_module)
        eval_data_obj.set_tokenizer(tokenizer)
        eval_data = eval_data_obj.get_data_set()

        args = config['trainer']

        trainer = Trainer(args, model, logger, device, device_ids)
        trainer.train(args, train_data, eval_data, tokenizer)


    
'''
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train(args, train_data, train_tokenizer)
'''

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument("--max_seq_length", default=512, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )



    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
