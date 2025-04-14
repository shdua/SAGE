import os, argparse, logging
import sys
sys.path.append("..")
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.SAGE import FANetModel
from processor.dataset import MMPNERProcessor, MMPNERDataset, PadCollate
from modules.train import NERTrainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {'twitter15': FANetModel, 'twitter17': FANetModel}
TRAINER_CLASSES = {'twitter15': NERTrainer, 'twitter17': NERTrainer}
DATA_PROCESS = {'twitter15': (MMPNERProcessor, MMPNERDataset), 'twitter17': (MMPNERProcessor, MMPNERDataset)}

def set_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    #parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--bert_name', default='xlm-roberta-large', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=18, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--bert_lr', default=5e-5, type=float)
    parser.add_argument('--crf_lr', default=1e-1, type=float)
    parser.add_argument('--other_lr', default=1e-3, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--write_path', default=None, type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--dropout_prob', default=0.5, type=float)
    parser.add_argument('--negative_slope1', default=0.01, type=float)
    parser.add_argument('--negative_slope2', default=0.01, type=float)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--dyn_k', type=int, default=10)
    parser.add_argument('--warmup_epoch', default=20, type=int, help="Number of warmup epochs")
    parser.add_argument('--warmup_power', default=2.0, type=float, help="Warmup power for the learning rate scheduler")
    parser.add_argument('--alpha', default=0.4, type=float, help="Alpha parameter for the training process")

    args = parser.parse_args()
    print("BERT model path:", args.bert_name)

    DATA_PATH = {
        'twitter15': {
            'train': 'data/twitter2015/train.txt',
            'dev': 'data/twitter2015/valid.txt',
            'test': 'data/twitter2015/test.txt'
        },
        'twitter17': {
            'train': 'data/twitter2017/train.txt',
            'dev': 'data/twitter2017/valid.txt',
            'test': 'data/twitter2017/test.txt'
        }
    }

    IMG_PATH = {'twitter15': 'data/twitter2015_images', 'twitter17': 'data/twitter2017_images'}
    data_path, img_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    set_seed(args.seed)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    print(args)
    writer = None

    processor = data_process(data_path, args.bert_name)
    train_dataset = dataset_class(processor, transform, img_path, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True, collate_fn=PadCollate(args=args, processor=processor), drop_last=True)

    dev_dataset = dataset_class(processor, transform, img_path, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                pin_memory=True, collate_fn=PadCollate(args=args, processor=processor))

    test_dataset = dataset_class(processor, transform, img_path, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True, collate_fn=PadCollate(args=args, processor=processor))

    label_mapping = processor.get_label_mapping()
    label_list = list(label_mapping.keys())
    model = FANetModel(label_list, args)

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                      label_map=label_mapping, args=args, logger=logger, writer=writer)

    if args.do_train:
        trainer.train()
        if args.save_path:
            args.load_path = os.path.join(args.save_path, 'best_model.pth')
            trainer.test()

    elif args.only_test:
        trainer.multiModal_before_train()
        if args.save_path:
            args.load_path = os.path.join(args.save_path, 'best_model.pth')
            trainer.test()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
