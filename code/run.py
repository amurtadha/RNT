import json
import logging
import argparse
import math
import os
import sys
import random
import numpy
from transformers import AdamW
import  copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,ConcatDataset
from data_utils import  MyDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
from transformers import  AutoTokenizer
from MyModel import RNT
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle as pk
from EvidentialSupport import RelDST, Noisiness
class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = AutoTokenizer.from_pretrained(self.opt.pretrained_bert_name)
        self.tokenizer=tokenizer
        self.trainset = MyDataset(self.opt, self.opt.dataset_file['train'], tokenizer)
        self.testset = MyDataset(self.opt, self.opt.dataset_file['test'], tokenizer)
        self.valset = MyDataset(self.opt, self.opt.dataset_file['dev'], tokenizer)
        self.noiseset = MyDataset(self.opt, self.opt.dataset_file['noise'], tokenizer)
        logger.info('clean {0}, noise {1}, test: {2}, dev {3}'.format(len(self.trainset),len(self.noiseset),len(self.testset),len(self.valset), ))

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x



    def _train_pt(self,model, criterion, optimizer, train_data_loader, val_data_loader,test_data_loader,t_total):
        max_val_acc = 0
        global_step = 0
        path = None

        for epoch in range(self.opt.num_epoch_pt):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            loss_total=[]
            model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,outputs= model(inputs)
                targets= inputs[-1]
                loss = criterion(outputs, targets)
                loss.sum().backward()

                optimizer.step()
                with torch.no_grad():
                    n_total += len(outputs)
                    loss_total.append(loss.sum().detach().item())

            logger.info('epoch : {}'.format(epoch))

            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            pres, recall, f1_score, acc, error_rate = self._evaluate_pt(model,val_data_loader)
            logger.info('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f},  val_err: {:.4f}'.format(pres, recall, f1_score, acc, error_rate))

            if f1_score > max_val_acc:
                max_val_acc = f1_score
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = copy.deepcopy(model.state_dict())

        return path

    def _evaluate_pt(self,model, data_loader):
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
    def _filter_pt_conffident(self, data_loader):
        labels = json.load(open('../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))
        n_correct, n_total = 0, 0
        model = RNT(self.opt)
        # model = nn.DataParallel(model)
        model.to(self.opt.device)
        model_path = 'state_dict/{0}_{1}_{2}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample))
        model.load_state_dict(torch.load(model_path))
        t_targets_all, t_outputs_all = None, None

        # switch model to evaluation mode
        data_to_save=[]
        model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_inputs[-1]
                _,t_outputs = model(t_inputs)
                t_outputs = nn.functional.softmax(t_outputs, dim=1)
                conf, pred= torch.max(t_outputs, dim=1)
                for i in range(len(conf)):
                    if conf[i].item() <.9:continue

                    label = pred[i].item()
                    or_label = t_sample_batched['ori_label'][i]
                    tem = {'text': t_sample_batched['text'][i], 'label': labels[label],
                       'ori_label': labels[or_label]}
                    data_to_save.append(tem)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
                true = t_targets_all.cpu().detach().numpy().tolist()
                pred = torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()

                f = metrics.f1_score(true, pred, average='macro', zero_division=0)
                acc = n_correct / n_total


        logger.info('{} {} pt_ filltering fi {} acc {} filltered {} '.format(self.opt.dataset, self.opt.train_sample,f, acc, len(data_to_save) ))
        # tem={'text':ckpt_noise_text[i], 'label':labels[ckpt_noise_target[i].item()], 'ori_label':labels[ckpt_noise_target[i].item()]}


        path_ = os.path.join(self.opt.save_dir, 'pt_confident_filtered_{0}_{1}_{2}.json'.format(self.opt.dataset, self.opt.plm,
                                                                                        str(self.opt.train_sample)))
        json.dump(data_to_save, open(path_, 'w'), indent=3)
    def _associate_pseudo_labels_label_pt(self, data_loader):
        labels = json.load(open('../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))

        model = RNT(self.opt)
        # model = nn.DataParallel(model)
        model.to(self.opt.device)
        model_path = 'state_dict/{0}_{1}_{2}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample))
        model.load_state_dict(torch.load(model_path))


        # switch model to evaluation mode
        data_to_save=[]
        model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,t_outputs = model(t_inputs)
                t_outputs = nn.functional.softmax(t_outputs, dim=1)
                conf, pred= torch.max(t_outputs, dim=1)
                for i in range(len(conf)):
                    # if conf[i].item() <.9:continue

                    label = pred[i].item()
                    or_label = t_sample_batched['ori_label'][i]
                    tem = {'text': t_sample_batched['text'][i], 'label': labels[label],
                       'ori_label': labels[or_label]}
                    data_to_save.append(tem)
        # tem={'text':ckpt_noise_text[i], 'label':labels[ckpt_noise_target[i].item()], 'ori_label':labels[ckpt_noise_target[i].item()]}


        path_ = os.path.join(self.opt.save_dir, 'pt_confident_filtered_{0}_{1}_{2}.json'.format(self.opt.dataset, self.opt.plm,
                                                                                        str(self.opt.train_sample)))
        json.dump(data_to_save, open(path_, 'w'), indent=3)


    def run_pt(self, ):
        model = RNT(self.opt)
        model = nn.DataParallel(model)
        model.to(self.opt.device)

        trainset = self.trainset
        testset = self.testset
        valset = self.valset

        if len(valset) == 2000:
            labeled_labels = np.array([v['label'] for v in valset])
            subval, _ = train_test_split(list(range(len(labeled_labels))),
                                                     train_size=200,
                                                     stratify=labeled_labels)
            valset.data = [valset[i] for i in subval]
        if self.opt.task in ['IFLYTEK', 'OCNLI', 'TNEWS']:
            testset= valset


        if self.opt.train_sample > 0 and len(trainset) > self.opt.train_sample:

            if self.opt.train_sample in [100, 30]:
                labeled_labels = np.array([v['label'] for v in trainset])
                train_labeled_idxs, _ = train_test_split(list(range(len(labeled_labels))),
                                                                        train_size=self.opt.train_sample*self.opt.lebel_dim,
                                                                        stratify=labeled_labels)
                trainset.data = [trainset[i] for i in train_labeled_idxs]

            else:
                index = random.sample(range(0, len(trainset)), self.opt.train_sample)
                trainset = torch.utils.data.Subset(trainset, index)

        logger.info(' train {}'.format(len(trainset)))
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = self.opt.optimizer(model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        batch_size = self.opt.batch_size
        if  self.opt.train_sample in[100, 30]:
            batch_size=8
        train_data_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.batch_size_val, shuffle=False)
        val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size_val, shuffle=False)
        t_total= int(len(train_data_loader) * self.opt.num_epoch)


        best_model_path = self._train_pt(model, criterion, optimizer, train_data_loader, val_data_loader,test_data_loader, t_total)
        model.load_state_dict(best_model_path)
        model.eval()

        pres, recall, f1_score, acc, error_rate = self._evaluate_pt(model,test_data_loader)
        logger.info(
            '>> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f},test_err: {:.4f}'.format(pres, recall, f1_score, acc, error_rate))
        path = 'state_dict/{0}_{1}_{2}.bm'.format(self.opt.dataset, self.opt.plm,str(self.opt.train_sample))
        if self.opt.save_model_pt:
            torch.save(model.module.state_dict(), path)

        with open('results/results_pt.tx', 'a+') as f:
            f.write('{0} {1} {2}\n'.format(self.opt.dataset,f1_score, acc))
        f.close()
        return  pres, recall, f1_score, acc, error_rate

    def _evaluate_nt(self, model, criterion,val_data_loader):
        with torch.no_grad():
            pred_list, true_all = [], []
            test_loss = test_acc = 0.0
            # logger.info('testing')
            for i, v_sample_batched in enumerate(tqdm(val_data_loader)):
                labels = v_sample_batched['label']

                labels = labels.to(self.opt.device)

                inputs = [v_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,logits = model(inputs)

                loss = criterion(logits, labels)
                test_loss += inputs[0].size(0) * loss.data

                _, pred = torch.max(logits.data, -1)
                acc = float((pred == labels.data).sum())
                test_acc += acc
                pred_list.extend(pred.detach().cpu().tolist())
                true_all.extend(labels.data.detach().cpu().tolist())

            test_loss /= len(val_data_loader.dataset)
            test_acc /= len(val_data_loader.dataset)
            f1_sc = metrics.f1_score(true_all, pred_list, average='macro')
            f1_micro = metrics.f1_score(true_all, pred_list, average='micro')

            return test_loss,f1_sc,f1_micro,test_acc

    def _train_nt(self,model,optimizer,weight,criterion,criterion_nll,criterion_nr, train_data_loader, val_data_loader, test_data_loader, t_total):

        best_acc_test=0
        global_step = 0
        best_f1_micro_test = 0.0
        best_valid_acc = 0.0
        best_f1_test = 0.0
        train_losses = torch.zeros(len(train_data_loader.dataset)) - 1.
        with torch.no_grad():
            train_preds_hist = torch.zeros(len(train_data_loader.dataset), self.opt.num_hist, self.opt.lebel_dim)
            train_preds = torch.zeros(len(train_data_loader.dataset), self.opt.lebel_dim) - 1.
            clean_ids = [d['new_index'] for _, d in enumerate(train_data_loader.dataset) if d['is_evidence'] == 1]
            noisy_ids = [d['new_index'] for _, d in enumerate(train_data_loader.dataset) if d['is_evidence'] == 0]
        for epoch in range(self.opt.num_epoch_negative):
            train_loss = train_loss_neg = train_acc = 0.0
            pl = 0.
            nl = 0.
            model.train()
            if epoch % self.opt.num_hist == 0 and epoch != 0:
                # if epoch in self.opt.epoch_step:

                lr_this_step = self.opt.learning_rate * self.warmup_linear(global_step / t_total,
                                                                           self.opt.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                    self.opt.learning_rate = param_group['lr']

            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):

                model.zero_grad()

                global_step += 1
                optimizer.zero_grad()
                labels = sample_batched['label']
                index = sample_batched['new_index']
                # with torch.no_grad():
                labels_neg = (labels.unsqueeze(-1).repeat(1, self.opt.neg_sample_num) + torch.LongTensor(len(labels),self.opt.neg_sample_num).random_(1, self.opt.lebel_dim)) % self.opt.lebel_dim
                labels = labels.to(self.opt.device)
                labels_neg = labels_neg.to(self.opt.device)

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                _,logits = model(inputs)


                s_neg = torch.log(torch.clamp(1. - F.softmax(logits, -1), min=1e-5, max=1.))
                s_neg *= weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()

                _, pred = torch.max(logits.data, -1)
                acc = float((pred == labels.data).sum())
                train_acc += acc

                train_loss += inputs[0].size(0) * criterion(logits, labels).data
                train_loss_neg += inputs[0].size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data
                train_losses[index] = criterion_nr(logits, labels).cpu().data

                ##

                if epoch >= self.opt.switch_epoch:
                    if epoch == self.opt.switch_epoch and i_batch == 0: logger.info('Switch to SelNT')
                    labels_neg[train_preds_hist.mean(1)[index, labels] < 1 / float(self.opt.lebel_dim)] = -100
                    # labels_neg[train_preds_hist.mean(1)[index, labels] < .8] = -100

                labels = labels * 0 -100
                loss = criterion(logits, labels) * float((labels >= 0).sum())
                loss_neg = criterion_nll(s_neg.repeat(self.opt.neg_sample_num, 1),
                                              labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
                ((loss + loss_neg) / (float((labels >= 0).sum()) + float((labels_neg[:, 0] >= 0).sum()))).backward()

                optimizer.step()
                #
                # with torch.no_grad():
                train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data
                pl += float((labels >= 0).sum())
                nl += float((labels_neg[:, 0] >= 0).sum())

            # with torch.no_grad():
            train_loss /= len(train_data_loader.dataset)
            train_loss_neg /= len(train_data_loader.dataset)
            train_acc /= len(train_data_loader.dataset)
            pl_ratio = pl / float(len(train_data_loader.dataset))
            nl_ratio = nl / float(len(train_data_loader.dataset))
            noise_ratio = 1. - pl_ratio


            noise = len(noisy_ids)
            logger.info(
                '[%6d/%6d] loss: %5f, loss_neg: %5f, acc: %5f, lr: %7f, noise: %d, pl: %5f, nl: %5f, noise_ratio: %5f'
                % (epoch, self.opt.num_epoch_negative, train_loss, train_loss_neg, train_acc, self.opt.learning_rate, noise,
                   pl_ratio, nl_ratio,
                   noise_ratio))


            model.eval()
            with torch.no_grad():
                logger.info('validating')
                val_loss, val_f1_sc,val_f1_micro, val_acc = self._evaluate_nt(model, criterion, val_data_loader)

                best_valid_acc = max(val_acc, best_valid_acc)

                logger.info('\t valid ...loss: %5f, acc: %5f,f1: %5f,f1 micro: %5f, best_acc: %5f' % (
                    val_loss, val_acc, val_f1_sc,val_f1_micro,best_valid_acc))

            is_best = val_acc >= best_valid_acc
            if is_best:

                path = 'state_dict/{}_nt_{}_{}_{}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample), self.opt.filtering)
                if self.opt.save_model_nt:
                    torch.save(model.module.state_dict(), path)
                model.eval()
                with torch.no_grad():
                    logger.info('testing')
                    # test_loss, f1_sc, f1_micro, test_acc
                    test_loss, test_f1_sc, test_f1_micro, test_acc = self._evaluate_nt(model, criterion,
                                                                                       test_data_loader)
                    best_acc_test = max(best_acc_test, test_acc)
                    best_f1_test = max(best_f1_test, test_f1_sc)
                    best_f1_micro_test = max(best_f1_micro_test, test_f1_micro)
                    logger.info(
                        '\t test ...loss: %5f, acc: %5f,f1 macro: %5f , f1 micro: %5f best_acc: %5f best_f1: %5f best_f1 micro: %5f ' % (
                            test_loss, test_acc, test_f1_sc, test_f1_micro, best_acc_test, best_f1_test,
                            best_f1_micro_test))

            model.train()

            assert train_preds[train_preds < 0].nelement() == 0
            train_preds_hist[:, epoch % self.opt.num_hist] = train_preds
            train_preds = train_preds * 0 - 1.
            assert train_losses[train_losses < 0].nelement() == 0
            train_losses = train_losses * 0 - 1.


            if epoch % self.opt.num_hist == 0:

                logger.info('saving separated histogram...')
                plt.hist(train_preds_hist.mean(1)[torch.arange(len(train_data_loader.dataset))[clean_ids]
                                                  , np.array([d['label'] for d in train_data_loader.dataset])[
                    clean_ids]].numpy(), bins=20, range=(0., 1.), edgecolor='black', alpha=0.5, label='clean')
                plt.hist(train_preds_hist.mean(1)[
                             torch.arange(len(train_data_loader.dataset))[noisy_ids]
                             ,  np.array([d['label'] for d in train_data_loader.dataset])[
                                 noisy_ids]].numpy(), bins=20, range=(0., 1.), edgecolor='black', alpha=
                         0.5, label='noisy')
                plt.xlabel('probability');
                plt.ylabel('number of data')
                plt.grid()
                plt.savefig(self.opt.save_dir_histo + '/histogram_sep_pt_conf_epoch%03d.jpg' % ( epoch))
                plt.clf()
        with open('results/main_nt.txt', 'a+') as f :
            f.write('{} {} {} {} f1 micro {} acc {} f1 {}  \n'.format(self.opt.dataset,self.opt.filtering, self.opt.train_sample, self.opt.plm, str(round(best_f1_micro_test, 4)), str(round(best_acc_test, 4)),str(round(best_f1_test, 4))))
        f.close()
    def run_nt(self):

        trainset = self.trainset
        testset = self.testset
        valset = self.valset
        if len(valset) == 2000:
            labeled_labels = np.array([v['label'] for v in valset])
            subval, _ = train_test_split(list(range(len(labeled_labels))),
                                         train_size=1000,
                                         stratify=labeled_labels)
            valset.data = [valset[i] for i in subval]

        if self.opt.task in ['IFLYTEK', 'OCNLI', 'TNEWS']:
            testset= valset

        for i in range(len(trainset)):
                trainset[i]['is_evidence'] = 1
        if self.opt.train_sample > 0 and  len(trainset)>self.opt.train_sample:

            if  self.opt.train_sample in [100, 30]:
                labeled_labels = np.array([v['label'] for v in trainset])
                train_labeled_idxs, _ = train_test_split(list(range(len(labeled_labels))),
                                                         train_size=self.opt.train_sample*self.opt.lebel_dim,
                                                         stratify=labeled_labels)
                trainset.data = [trainset[i] for i in train_labeled_idxs]
            else:
                index = random.sample(range(0, len(trainset)),self.opt.train_sample)
                trainset=torch.utils.data.Subset(trainset, index)

        if self.opt.filtering=='DST':
            path_ = os.path.join(self.opt.save_dir, 'DST_filtered_{0}_{1}_{2}_.json'.format(self.opt.dataset, self.opt.plm,
                                                                                            str(self.opt.train_sample)))
        else:
            path_ = os.path.join(self.opt.save_dir,'pt_confident_filtered_{0}_{1}_{2}.json'.format(self.opt.dataset, self.opt.plm,
                                                                                        str(self.opt.train_sample)))

        trainset_noise = MyDataset(self.opt, path_, self.tokenizer, noisy=True)



        for i in range(len(trainset_noise)):
            trainset_noise[i]['is_evidence'] = 0

        try:
            trainset = ConcatDataset([trainset.data, trainset_noise.data])
        except:
            trainset = ConcatDataset([trainset, trainset_noise])
        logger.info('negative training data')
        logger.info('train {0} , test {1}, dev {2}'.format(len(trainset), len(testset), len(valset)))

        t = 0
        for i in range(len(trainset)):
            trainset[i]['new_index'] = t
            t += 1

        train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=testset, batch_size=self.opt.batch_size_val, shuffle=False)
        val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size_val, shuffle=False)
        t_total = int(len(train_data_loader) * self.opt.num_epoch_negative)


        model = RNT(self.opt)
        model = nn.DataParallel(model)
        model.to(self.opt.device)
        _params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = self.opt.optimizer(model.parameters(), lr=self.opt.learning_rate,
                                            weight_decay=self.opt.l2reg)

        weight = torch.FloatTensor(self.opt.lebel_dim).zero_() + 1.
        for i in range(self.opt.lebel_dim):
            weight[i] = (torch.from_numpy(
                np.array([d['label'] for d in trainset]).astype(int)) == i).sum()
        weight = 1 / (weight / weight.max())

        weight = weight.to(self.opt.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        criterion_nll = nn.NLLLoss()
        criterion_nr = nn.CrossEntropyLoss(reduce=False)

        self._train_nt(model,optimizer,weight,criterion,criterion_nll,criterion_nr,  train_data_loader, val_data_loader, test_data_loader, t_total)

    def extract_DST_feature(self, path_, n_sample, data, data_clean):
        logger.info('extract DST Features')
        ckpt_val_text = data['text']
        if self.opt.task == 'STS':
            ckpt_val_text2 = data['text2']
        ckpt_val_target = data['target']
        ckpt_val_output = data['output']
        ckpt_val_repres = data['repres']
        ckpt_val_output_pred = torch.argmax(ckpt_val_output, dim=-1)

        ckpt_clean_target = data_clean['target']
        ckpt_clean_output = data_clean['output']
        ckpt_clean_repres = data_clean['repres']
        loss = nn.CrossEntropyLoss()

        trainin_data = []
        data_to_save = {}
        for i, trg in enumerate(tqdm(ckpt_clean_target)):
            if i in data_to_save: continue
            n_cand = RelDST()
            n_cand.id = i
            n_cand.pred = torch.argmax(ckpt_clean_output[i]).item()
            n_cand.label = trg.item()
            n_cand.prob = round(torch.max(F.softmax(ckpt_clean_output[i])).item(), 1)
            n_cand.evidence = True

            noise_features = torch.from_numpy(ckpt_clean_repres[i]).to('cuda')
            noise_features = F.normalize(noise_features.unsqueeze(0))
            rel = []
            for j in range(self.opt.lebel_dim):
                ids = (ckpt_clean_target == j).nonzero().squeeze(-1).tolist()
                if not len(ids):
                    ids = random.sample(ckpt_clean_target.tolist(), k=n_sample)
                elif len(ids) > n_sample:
                    ids = random.sample((ckpt_clean_target == j).nonzero().squeeze(-1).tolist(), k=n_sample)

                clean_features = torch.from_numpy(ckpt_clean_repres[ids]).to('cuda')
                clean_features = F.normalize(clean_features)
                cosine = torch.ones(len(ids), self.opt.lebel_dim).to('cuda')
                cosine[:, j] = F.linear(noise_features, clean_features)
                l = loss(cosine, torch.tensor([j] * len(ids)).to('cuda')).item()
                rel.append((j, round(l, 1)))
            n_cand.rel = rel
            data_to_save[i] = n_cand

        for i, trg in enumerate(tqdm(ckpt_val_target)):
            pred = ckpt_val_output_pred[i]

            noise_features = torch.from_numpy(ckpt_val_repres[i]).to('cuda')
            noise_features = F.normalize(noise_features.unsqueeze(0))

            rel = []
            for j in range(self.opt.lebel_dim):
                ids = (ckpt_clean_target == j).nonzero().squeeze(-1).tolist()
                if not len(ids):
                    ids = random.sample(ckpt_clean_target.tolist(), k=n_sample)
                elif len(ids) > n_sample:
                    ids = random.sample((ckpt_clean_target == j).nonzero().squeeze(-1).tolist(), k=n_sample)

                clean_features = torch.from_numpy(ckpt_clean_repres[ids]).to('cuda')
                clean_features = F.normalize(clean_features)
                # cosine = F.linear(noise_features, clean_features).squeeze(0).cpu().tolist()
                # for k, id_ in enumerate(ids):
                #     rel.append((j, id_, round(cosine[k], 1)))

                cosine = torch.ones(len(ids), self.opt.lebel_dim).to('cuda')
                cosine[:, j] = F.linear(noise_features, clean_features)
                l = loss(cosine, torch.tensor([j] * len(ids)).to('cuda')).item()
                rel.append((j, round(l, 1)))

            n_cand = RelDST()
            n_cand.id = len(data_to_save)
            n_cand.text = ckpt_val_text[i]
            if self.opt.task == 'STS':
                n_cand.text2 = ckpt_val_text2[i]
            n_cand.pred = pred.item()
            n_cand.label = trg.item()
            n_cand.prob = round(torch.max(F.softmax(ckpt_val_output[i])).item(), 1)
            n_cand.rel = rel
            n_cand.evidence = False
            data_to_save[len(data_to_save)] = n_cand

  
        pk.dump(data_to_save, open(path_, 'wb'))

    def _generate_features(self, model, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        represe_all = None
        # switch model to evaluation mode
        model.eval()
        texts = []
        texts2 = []
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                # t_inputs = [b.to(self.opt.device) for b in t_sample_batched]
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                # t_inputs = [t_sample_batched[col] for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['ori_label'].to(self.opt.device)
                # t_targets = t_sample_batched['label'].to(self.opt.device)
                vec, t_outputs = model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)
                texts.extend(t_sample_batched['text'])
                if self.opt.task in ['STS']:
                    texts2.extend(t_sample_batched['text2'])
                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                    represe_all = vec.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
                    represe_all = torch.cat((represe_all, vec.detach()), dim=0)
            true = t_targets_all.cpu().detach().numpy().tolist()
            pred = torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()
            represe_all = represe_all.cpu().detach().numpy()
            state = ({
                'text' : texts,
                'text2' 	: texts2,
                'target' 	: t_targets_all,
                'output' 	  : t_outputs_all,
                'repres' 	  : represe_all,

            })

            f = metrics.f1_score(true, pred, average='macro', zero_division=0)
            r = metrics.recall_score(true, pred, average='macro', zero_division=0)
            p = metrics.precision_score(true, pred, average='macro', zero_division=0)


            acc= n_correct /n_total
            error_rate =0

        return  p, r, f, acc, error_rate ,state

    def extract_dnn_vectors(self):

       
        logger.info('extract DNN Features')
        trainset_clean= self.trainset
        if self.opt.train_sample > 0 and len(self.trainset) > self.opt.train_sample:
            if self.opt.train_sample in [100, 30]:

                    labeled_labels = np.array([v['label'] for v in trainset_clean])
                    train_labeled_idxs, _ = train_test_split(list(range(len(labeled_labels))),
                                                             train_size=self.opt.train_sample*self.opt.lebel_dim,
                                                             stratify=labeled_labels)
                    trainset_clean.data = [trainset_clean[i] for i in train_labeled_idxs]
            else:
                index = random.sample(range(0, len(self.trainset)), self.opt.train_sample)
                trainset_clean = torch.utils.data.Subset(self.trainset, index)

        trainset_noise = self.noiseset
        val = self.valset

        clean_data_loader = DataLoader(dataset=trainset_clean, batch_size=self.opt.batch_size_val, shuffle=False)
        noise_data_loader = DataLoader(dataset=trainset_noise, batch_size=self.opt.batch_size_val, shuffle=False)
        val_data_loader = DataLoader(dataset=val, batch_size=self.opt.batch_size_val, shuffle=False)
        path = 'state_dict/{0}_{1}_{2}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample))

        model = RNT(self.opt)
        # self.model = nn.DataParallel(self.model)
        model.to(self.opt.device)
        model.load_state_dict(torch.load(path))
        model.eval()

        pres, recall, f1_score, acc, error_rate , state_clean= self._generate_features(model,clean_data_loader)
        pres, recall, f1_score, acc, error_rate , state_noise= self._generate_features(model,noise_data_loader)
        pres, recall, f1_score, acc, error_rate , state_val= self._generate_features(model,val_data_loader)


        logger.info('>> val_precision: {:.4f},  val_recall: {:.4f},  val_f1: {:.4f},  val_acc: {:.4f}, val_err: {:.4f}'.format(pres, recall, f1_score, acc, error_rate))
        path_clean = os.path.join(self.opt.save_dir, 'clean_{0}_{1}_{2}_exp_rand.tar'.format(self.opt.dataset, self.opt.plm,str(self.opt.train_sample)))
        path_noise = os.path.join(self.opt.save_dir, 'noise_{0}_{1}_{2}_exp_rand.tar'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample)))
        path_valid = os.path.join(self.opt.save_dir, 'valid_{0}_{1}_{2}_exp_rand.tar'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample)))
        # logger.info('saving best model...')
        # shutil.copyfile(fn, fn_best)
        torch.save(state_clean, path_clean)
        torch.save(state_noise, path_noise)
        torch.save(state_val, path_valid)

    def run(self,):


        # train PT classifier to assign pseudo-labels
        model_path = 'state_dict/{0}_{1}_{2}.bm'.format(self.opt.dataset, self.opt.plm, str(self.opt.train_sample))
        if not os.path.exists(model_path):
            self.run_pt()

        noise_data_loader = DataLoader(dataset=self.noiseset, batch_size=self.opt.batch_size_val, shuffle=False)


        if self.opt.filtering=='PT':
            self._filter_pt_conffident(noise_data_loader)
        elif self.opt.filtering == 'DST':

            path_valid = os.path.join(self.opt.save_dir,
                                      'valid_{0}_{1}_{2}_exp_rand.tar'.format(self.opt.dataset, self.opt.plm,
                                                                              str(self.opt.train_sample)))
            if not os.path.exists(path_valid):
                self.extract_dnn_vectors()

            # ckpt_val = torch.load(os.path.join(self.opt.save_dir,'valid_{0}_{1}_{2}_exp_rand.tar'.format(self.opt.dataset, self.opt.plm,str(self.opt.train_sample))))
            ckpt_clean = torch.load(os.path.join(self.opt.save_dir,
                                                 'clean_{0}_{1}_{2}_exp_rand.tar'.format(self.opt.dataset, self.opt.plm,
                                                                                         str(self.opt.train_sample))))
            ckpt_noise = torch.load(os.path.join(self.opt.save_dir,
                                                 'noise_{0}_{1}_{2}_exp_rand.tar'.format(self.opt.dataset, self.opt.plm,
                                                                                         str(self.opt.train_sample))))


            # valid_path = os.path.join(self.opt.save_dir, 'dt_DST_{0}_dev_{1}_{2}_exp_rand.pk'.format(self.opt.dataset, self.opt.plm,str(self.opt.train_sample)))
            noise_path = os.path.join(self.opt.save_dir,
                                          'dt_DST_{0}_noise_{1}_{2}_exp_rand.pk'.format(self.opt.dataset, self.opt.plm,
                                                                                        str(self.opt.train_sample)))
                # if not os.path.exists(valid_path):
            if not os.path.exists(noise_path):
                # self.extract_DST_feature( valid_path, n_sample=5, data=ckpt_val, data_clean=ckpt_clean)
                self.extract_DST_feature(noise_path, n_sample=5, data=ckpt_noise, data_clean=ckpt_clean)

            # dev_data = pk.load(open(valid_path, 'rb'))
            noise_data = pk.load(open(noise_path, 'rb'))

            # noise = Noisiness(self.opt, dev_data)
            noise = Noisiness(self.opt, noise_data)
            noise.getNoisy(save=True, top_m=0.5)
        else:
            self._associate_pseudo_labels_label_pt(noise_data_loader)

        self.run_nt()




def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TNEWS', type=str, required=True, choices=  ['TNEWS', 'IFLYTEK', "yelp", 'SST','SST-5', 'CR', 'MR', 'MPQA', "AG", 'DBPedia',"TREC", 'yahoo'])
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning-rate', default=1e-5, type=float, help='')
    parser.add_argument('--adam-epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--warmup_proportion', default=0.002, type=float)
    parser.add_argument('--num-epoch', default=6, type=int, help='')
    parser.add_argument('--num-epoch-pt', default=6, type=int, help='')
    parser.add_argument('--num-epoch-negative', default=11, type=int, help='')
    parser.add_argument('--switch-epoch', default=5, type=int, help='')
    parser.add_argument('--num-hist', default=2, type=int, help='')
    parser.add_argument('--neg-sample-num', default=10, type=int, help='')
    parser.add_argument('--batch-size', default=64, type=int, help='')
    parser.add_argument('--batch-size-val', default=128, type=int, help='')
    parser.add_argument('--plm', default='bert', type=str, choices=['bert', 'roberta','sbert', 'chinese'])
    parser.add_argument('--train-sample', default=1000, type=int, required=True,  choices=[0,30,1000,10000])
    parser.add_argument('--save-model-pt', default=1, type=int, help='0 false or 1 true')
    parser.add_argument('--save-model-nt', default=1, type=int, help='0 false or 1 true')
    parser.add_argument('--filtering', default='DST' , type=str, choices=['DST', 'PT', 'Nope'])
    parser.add_argument('--save-dir', default='state_dict' , type=str, help='')
    parser.add_argument('--device-group', default='5' , type=str, help='')
    parser.add_argument('--device', default='cuda' , type=str, help='')
    opt = parser.parse_args()

    if opt.dataset in ['TNEWS', 'IFLYTEK', 'OCNLI', 'AFQMC']:
        opt.plm='chinese'
        opt.save_model_nt=1


    if opt.plm == 'bert':opt.pretrained_bert_name = 'bert-base-uncased'
    elif opt.plm == 'roberta':opt.pretrained_bert_name = 'roberta-large'
    elif opt.plm == 'sbert':opt.pretrained_bert_name = 'sentence-transformers/all-mpnet-base-v2'
    elif opt.plm == 'chinese':opt.pretrained_bert_name = 'hfl/chinese-roberta-wwm-ext-large'

    label_dims = {'TNEWS':15,'OCNLI':3,'IFLYTEK':119,'yelp': 5,'AFQMC':2, 'IMDB': 2,'semeval': 2, 'semeval16_rest': 2, 'sentihood': 4, 'TREC': 6, 'DBPedia': 14, 'AG': 4,'SUBJ': 2,'ELEC': 2,'SST': 2,'SST-5': 5,'CR': 2,'MR': 2, 'PC': 2, 'yahoo': 10, 'MPQA':2}
    opt.lebel_dim = label_dims[opt.dataset]
    opt.max_seq_len = {'TNEWS':128, 'OCNLI':128, 'IFLYTEK':128, 'AFQMC':128,'yelp': 256, 'TREC': 20, 'yahoo': 256,'ELEC': 256,'MPQA':10 ,'AG': 100,'MR':30,'SST':30,'SST-5':30,'PC':30,'CR':30,'DBPedia': 160, 'IMDB':280,'SUBJ': 30,'semeval': 80}.get(opt.dataset)
    task_list = {'CLUE':['TNEWS', 'IFLYTEK'],"SA": [ "yelp", 'SST','SST-5',  'CR', 'MR', 'MPQA'], "TOPIC": ["AG", 'DBPedia'], "QA": ["TREC", 'yahoo']}



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
        'train_all': '../datasets/{1}/{0}/train.json'.format(opt.dataset, opt.task),
        'train': '../datasets/{1}/{0}/sub_clean.json'.format(opt.dataset, opt.task),
        'noise': '../datasets/{1}/{0}/sub_noise_new.json'.format(opt.dataset, opt.task),
        'test': '../datasets/{1}/{0}/test.json'.format(opt.dataset, opt.task),
        'dev': '../datasets/{1}/{0}/dev.json'.format(opt.dataset, opt.task),
    }


    if opt.dataset in ['TREC', 'SST', 'SST-5', 'CR', 'MR']:
        opt.batch_size= 16
        opt.num_epoch_pt= 20
        opt.num_epoch_negative= 21
        opt.switch_epoch= 10
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_group
    input_colses =  ['input_ids', 'segments_ids', 'input_mask', 'label']

    opt.dataset_file = dataset_files
    opt.inputs_cols = input_colses
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.optimizer = AdamW
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

