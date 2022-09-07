import pandas as pd

from utils.pyds import MassFunction
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from sklearn import metrics
# import networkx as nx
import random
import pickle as pk
from collections import defaultdict
import json
class Noisiness():
    def __init__(self,opt, data, ):
        self.data = data
        # self.classes_evidence = classes_evidence
        # self.train_loader = train_loader
        self.get_noise_evidence()
        self.opt= opt

    def get_noise_evidence(self):
        features=defaultdict(list)
        features_per_var=defaultdict(list)
        for i in range(len(self.data)):
            current= self.data[i]
            # if current.evidence: continue
            # if current.pred == current.label:continue
            features[('prob',current.pred, current.prob)].append(current.id)
            features_per_var[current.id].append(('prob',current.pred, current.prob))
            for f in  current.rel:
                # features[('cosine', f[0], f[1])].append(f[1])
                features[('cosine', f[0], f[1])].append(current.id)
                features_per_var[current.id].append(('cosine', f[0], f[1]))
        for  k,v in features.items():
            features[k] =list(set(v))

        for  k,v in features_per_var.items():
            features_per_var[k] =list(set(v))

        features_labeled_var= defaultdict(list)
        for k, v in features.items():
            for var in v:
                if not self.data[var].evidence: continue
                features_labeled_var[k].append(var)

        # here compute mass
        dict_feat_stats = {}  # key: feature names  # value: [(n_samples, neg_prob, pos_prob)]
        for feat, label_vars in features.items():

            n_samples = len(label_vars)
            # if feat[0] =='prob':continue
            pos_label_vars =[asp for asp in label_vars if self.data[asp].label == feat[1]]
            neg_label_vars =[asp for asp in label_vars if self.data[asp].label != feat[1]]
            # pos_label_vars = [asp for (asp, rel_type) in label_vars if dict_asp_aspnode[asp].polarity == 'positive']
            # neg_label_vars = [asp for (asp, rel_type) in label_vars if dict_asp_aspnode[asp].polarity == 'negative']
            # {c: v for c, v in dict_feat_label_vars.items() if 'cluster' not in c}
            # if feat== 'dnn_0.9987' :
            #     pass
            if n_samples != 0:
                dict_feat_stats[feat] = (n_samples, len(neg_label_vars) / n_samples, len(pos_label_vars) / n_samples)
            if len(pos_label_vars) + len(neg_label_vars) != n_samples:
                raise ValueError('the sum of probabilities is not equal to 1')

        dict_unlabvar_feature_evis = {}  # key: unlabeled variable
        # value: [(feat_name, n_samples, neg_prob, pos_prob), ...]
        for i in range(len(self.data)):
            current = self.data[i]
            if current.evidence:continue
            feat_distrib_tmp = []
            for feat in features_per_var.get(current.id):
                if dict_feat_stats.get(feat):
                    n_samples, neg_prob, pos_prob = dict_feat_stats[feat]
                    feat_distrib_tmp.append((feat, n_samples, neg_prob, pos_prob))
            if len(feat_distrib_tmp) > 0:
                dict_unlabvar_feature_evis[current.id] = feat_distrib_tmp
        self.dict_unlabvar_feature_evis= dict_unlabvar_feature_evis

    def construct_mass_function_for_propensity(self, uncertain_degree, label_prob, unlabel_prob):
        '''
        # l: support for labeling
        # u: support for noisy
        '''
        return MassFunction({'l': (1 - uncertain_degree) * label_prob,
                             'u': (1 - uncertain_degree) * unlabel_prob,
                             'lu': uncertain_degree})

    def combine_evidences_with_ds(self, mass_functions, normalization):
        # combine evidences from different sources
        if len(mass_functions) < 2:
            combined_mass = mass_functions[0]
        else:
            combined_mass = mass_functions[0].combine_conjunctive(mass_functions[1], normalization)
            if len(mass_functions) > 2:
                for mass_func in mass_functions[2: len(mass_functions)]:
                    combined_mass = combined_mass.combine_conjunctive(mass_func, normalization)
        return combined_mass

    def labeling_propensity_with_ds(self, mass_functions):
        combined_mass = self.combine_evidences_with_ds(mass_functions, normalization=True)
        return combined_mass

    def get_unlabvar_evi_support(self,cosine_evi_uncer_degree=.3):
        dict_unlabvar_propensity_masses = {}  # key: unlabeled variable
        # value: mass functions for different evidences
        for i in range(len(self.data)):
            unlabel_var = self.data[i]
            if unlabel_var.evidence:continue
            mass_functions_tmp = []
            if self.dict_unlabvar_feature_evis.get(unlabel_var.id):
                for (feat_name, n_samples, neg_prob, pos_prob) in self.dict_unlabvar_feature_evis.get(unlabel_var.id):

                    mass_functions_tmp.append(self.construct_mass_function_for_propensity(cosine_evi_uncer_degree,
                                                                                     max(pos_prob, neg_prob),
                                                                                     min(pos_prob, neg_prob)))
            if len(mass_functions_tmp) > 0:
                dict_unlabvar_propensity_masses[unlabel_var] = mass_functions_tmp

        dict_unlabvar_propen_combined_masses = {unlabel_var: self.labeling_propensity_with_ds(mass_funcs)
                                                for unlabel_var, mass_funcs in dict_unlabvar_propensity_masses.items()}
        # key: unlabel variable
        # value: combined mass function ({{'l'}:0.9574468085106382; {'u'}:0.04255319148936169; {'l', 'u'}:0.0})
        return dict_unlabvar_propen_combined_masses


    # def getNoisy(self, limit_per_class=3000):
    def getNoisy_(self, top_m=500, save=False, show_plt=True):
        dict_noisy_evi_support = self.get_unlabvar_evi_support()
        sorted_unlabvar_evi_support = sorted(dict_noisy_evi_support.items(), key=lambda tuple: tuple[1]['l'],
                                             reverse=True)

        from sklearn import metrics
        if show_plt:
            data_to_print = []
            for i in range(len(sorted_unlabvar_evi_support)):
                item, mass = sorted_unlabvar_evi_support[i]
                data_to_print.append([item.label, item.pred, round(mass['l'], 2)])
            dt = pd.DataFrame(data_to_print, columns=['label', 'pred', 'mass'])
            data_to_print = []
            # for grp, data in dt.groupby('mass'):
            true_label = []
            true_pred = []
            prev = 1.0
            for data in dt.sort_values('mass', ascending=False).values:

                true, prd, mass = data
                if mass < prev:
                    acc = metrics.accuracy_score(true_label, true_pred)
                    data_to_print.append([acc, mass])
                    true_label = []
                    true_pred = []
                    prev = mass
                true_label.append(true)
                true_pred.append(prd)
            import matplotlib.pyplot as plt
            dt = pd.DataFrame(data_to_print, columns=['acc', 'mass'])
            plt.plot(dt['mass'], dt['acc'])
            plt.xlabel('clean data probability')
            plt.ylabel('accuracy')
            plt.show()

        top_varibal_m = [var for (var, propens) in sorted_unlabvar_evi_support[:top_m]]

        true_label = []
        predicted = []
        for i in range(len(top_varibal_m)):
            current = top_varibal_m[i]
            true_label.append(current.label)
            predicted.append(current.pred)

        f1 = metrics.f1_score(true_label, predicted, average='macro')
        acc = metrics.accuracy_score(true_label, predicted)
        print('f1 :{0} acc {1}, top_m: {2}'.format(f1, acc, top_m))
        print('----')

        if save:
            print('save less risky data')
            data_to_save=[]
            labels = json.load(open('datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))

            for i in tqdm(range(len(top_varibal_m))):
                current = top_varibal_m[i]
                if self.opt.task == 'STS':
                    tem = {'text': current.text, 'text2':current.text2, 'label': labels[current.pred],
                           'ori_label': labels[current.label]}
                else:
                    tem = {'text': current.text, 'label': labels[current.pred],
                           'ori_label': labels[current.label]}
                # tem={'text':ckpt_noise_text[i], 'label':labels[ckpt_noise_target[i].item()], 'ori_label':labels[ckpt_noise_target[i].item()]}
                data_to_save.append(tem)

            path_ = os.path.join(self.opt.save_dir,
                                     'DST_filtered_{0}_{1}_{2}.json'.format(self.opt.dataset, self.opt.plm,
                                                                                str(self.opt.train_sample)))
            json.dump(data_to_save, open(path_, 'w'), indent=3)


        if False:

            while True:
                top_varibal_m = [var for (var, propens) in sorted_unlabvar_evi_support[:top_m]]

                true_label =[]
                predicted = []
                for i in range(len(top_varibal_m)):
                    current= top_varibal_m[i]
                    true_label.append(current.label)
                    predicted.append(current.pred)
                from sklearn import metrics
                f1= metrics.f1_score(true_label,predicted,average='macro')
                acc= metrics.accuracy_score(true_label,predicted)
                print(metrics.classification_report(true_label,predicted))
                print('f1 :{0} acc {1}, top_m: {2}'.format(f1, acc, top_m))
                print('----')
                top_m+=200
    def getNoisy(self, top_m=None, save=False, show_plt=True, threshold=None):
        dict_noisy_evi_support = self.get_unlabvar_evi_support()
        sorted_unlabvar_evi_support = sorted(dict_noisy_evi_support.items(), key=lambda tuple: tuple[1]['l'],
                                             reverse=True)
        accs= []
        t_m =int(len(sorted_unlabvar_evi_support)*.1)
        data_to_print=[]
        for i in range(0, len(sorted_unlabvar_evi_support), t_m):
                true_pred = [t[0].pred for t in sorted_unlabvar_evi_support[: i+t_m]]
                true_label = [t[0].label for t in sorted_unlabvar_evi_support[:i+t_m]]
                mass = np.mean([t[1]['l'] for t in sorted_unlabvar_evi_support[:i+t_m]])
                f1_m = metrics.f1_score(true_label, true_pred, average='macro')
                f1_weight = metrics.f1_score(true_label, true_pred, average='weighted')
                acc = metrics.accuracy_score(true_label, true_pred)
                accs.append(acc)
                data_to_print.append([acc,mass, f1_m,f1_weight, len(true_pred)/ len(sorted_unlabvar_evi_support)])

        dt = pd.DataFrame(data_to_print, columns=['acc', 'mass','f1 macro','f1 weighted','data'])

        mean_acc= np.max(accs) - np.std(accs)
        if top_m is None:
            top_m = np.max(dt[dt['acc'] > mean_acc]['data'].values)
        true_,pred_=[],[]
        for (var, propens) in sorted_unlabvar_evi_support:
            true_.append(var.label)
            pred_.append(var.pred)
        f1_full=  metrics.f1_score(true_, pred_, average='macro')
        acc_full=  metrics.accuracy_score(true_, pred_)
        top_varibal_m = [var for (var, propens) in sorted_unlabvar_evi_support[:int(len(sorted_unlabvar_evi_support)*top_m)]]

        if save:
            print('save less risky data')
            data_to_save=[]
            labels = json.load(open('../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))
            for i in tqdm(range(len(top_varibal_m))):
                current = top_varibal_m[i]
                if self.opt.task == 'STS':
                    tem = {'text': current.text, 'text2':current.text2, 'label': labels[current.pred],
                           'ori_label': labels[current.label]}
                else:
                    tem = {'text': current.text, 'label': labels[current.pred],
                           'ori_label': labels[current.label]}
                # tem={'text':ckpt_noise_text[i], 'label':labels[ckpt_noise_target[i].item()], 'ori_label':labels[ckpt_noise_target[i].item()]}
                data_to_save.append(tem)

            path_ = os.path.join(self.opt.save_dir,'DST_filtered_{0}_{1}_{2}.json'.format(self.opt.dataset, self.opt.plm,
                                                                                str(self.opt.train_sample)))
            json.dump(data_to_save, open(path_, 'w'), indent=3)


            predicted = [d['label'] for d in data_to_save]
            true_label = [d['ori_label'] for d in data_to_save]
            f1 = metrics.f1_score(true_label, predicted, average='macro')
            acc = metrics.accuracy_score(true_label, predicted)
            print(metrics.classification_report(true_label, predicted))
            print('full dev f1 macro: {4}, acc_full {5}, f1 :{0} acc {1}, top_m: {2} propo :{3}'.format(f1, acc, len(data_to_save), len(data_to_save)/len(sorted_unlabvar_evi_support),f1_full, acc_full ))
            print('----')





class RelDST():
    def __init__(self, ):
        self.id = int,
        self.pred= float,
        self.text= str,
        self.text2= str,
        self.label= int,
        self.prob= float,
        self.rel =[]
        self.evidence =bool

def extract_DST_feature(opt,path_, n_sample, data, data_clean):
    print('extract DST Features')
    ckpt_val_text = data['text']
    ckpt_val_target = data['target']
    ckpt_val_output = data['output']
    ckpt_val_repres = data['repres']
    ckpt_val_output_pred = torch.argmax(ckpt_val_output, dim=-1)

    ckpt_clean_target = data_clean['target']
    ckpt_clean_output = data_clean['output']
    ckpt_clean_repres = data_clean['repres']
    # loss = nn.CrossEntropyLoss()

    trainin_data = []
    data_to_save={}
    for i, trg in enumerate(tqdm(ckpt_clean_target)):
        if i  in data_to_save:continue
        n_cand = RelDST()
        n_cand.id = i
        n_cand.pred = torch.argmax(ckpt_clean_output[i]).item()
        n_cand.label = trg.item()
        n_cand.prob = round(torch.max(F.softmax(ckpt_clean_output[i])).item(), 1)
        n_cand.evidence= True

        data_to_save[i] = n_cand


    for i, trg in enumerate(tqdm(ckpt_val_target)):
        pred = ckpt_val_output_pred[i]
        # if ckpt_noise_output[i]<prob_thre:continue

        noise_features = torch.from_numpy(ckpt_val_repres[i]).to('cuda')
        noise_features = F.normalize(noise_features.unsqueeze(0))


        rel = []
        for j in range(opt.lebel_dim):
            ids = (ckpt_clean_target == j).nonzero().squeeze(-1).tolist()
            if not len(ids):
                ids = random.sample(ckpt_clean_target.tolist(), k=n_sample)
            elif len(ids) > n_sample:
                ids = random.sample((ckpt_clean_target == j).nonzero().squeeze(-1).tolist(), k=n_sample)


            clean_features = torch.from_numpy(ckpt_clean_repres[ids]).to('cuda')
            clean_features = F.normalize(clean_features)
            cosine= F.linear(noise_features, clean_features).squeeze(0).cpu().tolist()

            for k, id_ in enumerate(ids):
                rel.append((j,id_, round(cosine[k], 1)))
            # cosine = torch.ones(len(ids), opt.lebel_dim).to('cuda')
            # cosine[:, j] = F.linear(noise_features, clean_features)
            # l = loss(cosine, torch.tensor([j] * len(ids)).to('cuda')).item()
            # logits.append(l)

        n_cand = RelDST()
        n_cand.id = len(data_to_save)
        n_cand.text = ckpt_val_text[i]
        n_cand.pred = pred.item()
        n_cand.label = trg.item()
        n_cand.prob = round(torch.max(F.softmax(ckpt_val_output[i])).item(), 1)
        n_cand.rel= rel
        n_cand.evidence= False
        data_to_save[len(data_to_save)]= n_cand

        # trainin_data.append([pred.item(), logits, ckpt_val_output[i].tolist(), 0 if trg == pred else 1, trg])
    # trainin_data = pd.DataFrame(trainin_data, columns=['pred', 'cosine', 'logits', 'label', 'ori_label'])

    pk.dump(data_to_save, open(path_, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SST', type=str, help='TREC,, ')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    # parser.add_argument('--learning_rate', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--adam_epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--reg', type=float, default=0.00005, help='regularization constant for weight penalty')
    parser.add_argument('--num_epoch', default=6, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--num_epoch_negative', default=21, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--switch_epoch', default=11, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--num_hist', default=5, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--neg_sample_num', default=10, type=int, help='try larger number for non-BERT models')

    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--batch_size_val', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=35500, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.002, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--lebel_dim', default=2, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--use_noisy', default=0, type=int, help='0 false or 1 true')
    parser.add_argument('--plm', default='bert', type=str, help='0 false or 1 true')
    parser.add_argument('--train_sample', default=0, type=int, help='0 false or 1 true')
    parser.add_argument('--save_model', default=1, type=int, help='0 false or 1 true')
    parser.add_argument('--save_model_nt', default=1, type=int, help='0 false or 1 true')
    parser.add_argument('--use_pt', default=0, type=int, help='0 false or 1 true')

    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    parser.add_argument('--save_dir', default='state_dict', type=str, help='e.g. cuda:0')
    parser.add_argument('--device_group', default='2', type=str, help='e.g. cuda:0')
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)

    parser.add_argument('--seed', default=65, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,
                        help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    # RoBERTa-wwm-ext-large

    if opt.plm == 'bert':
        opt.pretrained_bert_name = 'bert-base-uncased'
    elif opt.plm == 'roberta':
        opt.pretrained_bert_name = 'roberta-large'
    elif opt.plm == 'sbert':
        opt.pretrained_bert_name = 'sentence-transformers/all-mpnet-base-v2'
    elif opt.plm == 'chinese':
        opt.pretrained_bert_name = 'hfl/chinese-roberta-wwm-ext-large'

    label_dims = {'TNEWS': 15, 'OCNLI': 3, 'IFLYTEK': 119, 'yelp': 5, 'IMDB': 2, 'semeval': 2, 'semeval16_rest': 2,
                  'sentihood': 4, 'TREC': 6, 'DBPedia': 14, 'AG': 4,
                  'SUBJ': 2, 'SST': 2, 'CR': 2, 'MR': 2, 'PC': 2, 'yahoo': 10, 'MPQA': 2}
    opt.lebel_dim = label_dims[opt.dataset]
    opt.max_seq_len = {'TNEWS': 128, 'OCNLI': 128, 'IFLYTEK': 128, 'yelp': 256, 'TREC': 20, 'yahoo': 256, 'MPQA': 10,
                       'AG': 100, 'MR': 30, 'SST': 30, 'PC': 30, 'CR': 30, 'DBPedia': 160, 'IMDB': 280, 'SUBJ': 30,
                       'semeval': 80}.get(opt.dataset)
    task_list = {'CLUE': ['TNEWS', 'IFLYTEK'], "SA": ["yelp", 'SST', 'PC', 'CR', 'MR', 'MPQA', 'IMDB'],
                 "TOPIC": ["AG", 'sougou', 'DBPedia'], "QA": ["TREC", 'yahoo'], 'SUBJ': ['SUBJ'],
                 'ACD': ['semeval', 'sentihood'], 'STS': ['OCNLI']}
    # opt.batch_size = {'yelp': 64, 'TREC': 64, 'yahoo': 64, 'AG': 64, 'DBPedia': 64, 'SUBJ': 64, 'semeval': 64}.get(
    #     opt.dataset)
    opt.num_epoch = {'yelp': 6, 'TNEWS': 6, 'IFLYTEK': 10, 'OCNLI': 10, 'MPQA': 8, 'IMDB': 6, 'TREC': 10, 'SST': 6,
                     'PC': 4, 'CR': 10, 'yahoo': 6, 'MR': 10, 'AG': 6, 'DBPedia': 4, 'SUBJ': 3, 'semeval': 6}.get(
        opt.dataset)

    ckpt_val = torch.load(os.path.join(opt.save_dir, 'valid_{0}_{1}_{2}.tar'.format(opt.dataset, opt.plm,
                                                                                    str(opt.train_sample))))
    ckpt_clean = torch.load(os.path.join(opt.save_dir,
                                         'clean_{0}_{1}_{2}.tar'.format(opt.dataset, opt.plm,
                                                                        str(opt.train_sample))))
    ckpt_noise = torch.load(os.path.join(opt.save_dir,
                                         'noise_{0}_{1}_{2}.tar'.format(opt.dataset, opt.plm,
                                                                        str(opt.train_sample))))

    valid_path = os.path.join(opt.save_dir, 'dt_DST_{0}_dev_{1}_{2}.pk'.format(opt.dataset, opt.plm,
                                                                                 str(opt.train_sample)))

    noise_path = os.path.join(opt.save_dir, 'dt_DST_{0}_noise_{1}_{2}.pk'.format(opt.dataset, opt.plm,
                                                                               str(opt.train_sample)))
    if not os.path.exists(valid_path):
        extract_DST_feature(opt,valid_path, n_sample=5, data=ckpt_val, data_clean=ckpt_clean)
        extract_DST_feature(opt,noise_path, n_sample=5, data=ckpt_noise, data_clean=ckpt_clean)


    dev_data = pk.load(open(valid_path, 'rb'))
    noise_data = pk.load(open(noise_path, 'rb'))

    print()

    # noise= Noisiness(noise_distant_evidence, self.train_all_data_loader, top_m_evi_support=10, classes_evidence=self.classes_evidence)
    noise= Noisiness(opt, noise_data)
    noise.getNoisy(top_m= .6)
