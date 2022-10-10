import logging
import argparse
import os
import sys
import random
import numpy
import torch
from torch.utils.data import DataLoader
from data_utils import  MyDataset
from tqdm import tqdm
from transformers import  AutoTokenizer
from MyModel import RNT
from sklearn import metrics
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert_name)
        self.tokenizer=tokenizer
        self.testset = MyDataset(self.opt, self.opt.dataset_file['test'], tokenizer)

    def _evaluate(self,model, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_inputs[-1]
                _,t_outputs = model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
            true= t_targets_all.cpu().detach().numpy().tolist()
            pred =torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()

            f = metrics.f1_score(true, pred, average='macro', zero_division=0)
            r = metrics.recall_score(true, pred, average='macro', zero_division=0)
            p = metrics.precision_score(true, pred, average='macro', zero_division=0)
            acc= n_correct/n_total

            error_rate=0

        return  p, r, f, acc, error_rate

    def run(self,):

        path = 'state_dict/{}_nt_{}_{}_{}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample), self.opt.filtering)
        model = RNT(self.opt)
        model.to(self.opt.device)
        model.load_state_dict(torch.load(path))
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size_val, shuffle=False)

        pres, recall, f1_score, acc, error_rate = self._evaluate(model, test_data_loader)
        logger.info(
            '> test_precision: {:.4f},test_recall: {:.4f}, test_f1: {:.4f},  test_acc: {:.4f},  test_err: {:.4f}'.format(
                pres, recall, f1_score, acc, error_rate))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TNEWS', type=str, required=True, choices=  ['TNEWS', 'IFLYTEK', "yelp", 'SST','SST-5', 'CR', 'MR', 'MPQA', "AG", 'DBPedia',"TREC", 'yahoo'])
    parser.add_argument('--batch-size-val', default=128, type=int, help='')
    parser.add_argument('--plm', default='bert', type=str, choices=['bert', 'roberta','sbert', 'chinese'])
    parser.add_argument('--train-sample', default=1000, type=int, required=True,  choices=[0,30,1000,10000])
    parser.add_argument('--save-dir', default='state_dict' , type=str, help='')
    parser.add_argument('--device-group', default='5' , type=str, help='')
    parser.add_argument('--filtering', default='DST' , type=str, choices=['DST', 'PT', 'Nope'])
    parser.add_argument('--device', default='cuda' , type=str, help='')
    opt = parser.parse_args()

    if opt.dataset in ['TNEWS', 'IFLYTEK', 'OCNLI', 'AFQMC']:
        opt.plm='chinese'
        opt.save_model_nt=1

    if opt.plm == 'bert':
        opt.pretrained_bert_name = 'bert-base-uncased'
        # opt.pretrained_bert_name = 'plm/bert/'
    elif opt.plm == 'roberta':
        opt.pretrained_bert_name = 'roberta-large'
    elif opt.plm == 'sbert':
        opt.pretrained_bert_name = 'sentence-transformers/all-mpnet-base-v2'
    elif opt.plm == 'chinese':
        opt.pretrained_bert_name = 'hfl/chinese-roberta-wwm-ext-large'

    label_dims = {'TNEWS':15,'OCNLI':3,'IFLYTEK':119,'yelp': 5,'AFQMC':2, 'IMDB': 2,'semeval': 2, 'semeval16_rest': 2, 'sentihood': 4, 'TREC': 6, 'DBPedia': 14, 'AG': 4,
                  'SUBJ': 2,'ELEC': 2,'SST': 2,'SST-5': 5,'CR': 2,'MR': 2, 'PC': 2, 'yahoo': 10, 'MPQA':2}
    opt.lebel_dim = label_dims[opt.dataset]
    opt.max_seq_len = {'TNEWS':128, 'OCNLI':128, 'IFLYTEK':128, 'AFQMC':128,'yelp': 256, 'TREC': 20, 'yahoo': 256,'ELEC': 256,'MPQA':10 ,'AG': 100,'MR':30,'SST':30,'SST-5':30,'PC':30,'CR':30,'DBPedia': 160, 'IMDB':280,'SUBJ': 30,
                       'semeval': 80}.get(opt.dataset)
    task_list = {'CLUE':['TNEWS', 'IFLYTEK'],"SA": [ "yelp", 'SST','SST-5',  'CR', 'MR', 'MPQA'], "TOPIC": ["AG", 'DBPedia'], "QA": ["TREC", 'yahoo']
                 }

    if opt.dataset  in ['TNEWS', 'IFLYTEK']:
        opt.neg_sample_num = 10
    else:
        opt.neg_sample_num = opt.lebel_dim-1

    for k, v in task_list.items():
        if opt.dataset in v:
            opt.task = k
            break

    opt.seed= random.randint(20,300)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_files = {
        'test': '../datasets/{1}/{0}/test.json'.format(opt.dataset, opt.task)
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_group
    input_colses =  ['input_ids', 'segments_ids', 'input_mask', 'label']

    opt.dataset_file = dataset_files
    opt.inputs_cols = input_colses

    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.save_dir_histo = 'results/{0}/{1}/'.format(opt.task, opt.dataset)
    os.makedirs(opt.save_dir_histo, exist_ok=True)
    log_file = '{}-{}.log'.format(opt.dataset, opt.train_sample)
    logger.addHandler(logging.FileHandler(log_file))

    logger.info('seed {}'.format(opt.seed))
    logger.info('dataset {}'.format(opt.dataset))
    ins = Instructor(opt)
    ins.run()



if __name__ == '__main__':
    main()
