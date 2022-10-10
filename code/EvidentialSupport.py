import copy
import logging

import pandas as pd
from utils.pyds import MassFunction
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from collections import defaultdict
import json

class Noisiness():
    def __init__(self,opt, data, ):
        self.data = data
        self.get_noise_evidence()
        self.opt= opt

    def get_noise_evidence(self):
        features=defaultdict(list)
        features_per_var=defaultdict(list)
        for i in range(len(self.data)):
            current= self.data[i]
            # if current.evidence: continue
            # if current.pred == current.label:continue
            if current.evidence:
                features[('prob',current.pred, current.prob)].append(current.id)
            features_per_var[current.id].append(('prob',current.pred, current.prob))
            for f in  current.rel:
                # features[('cosine', f[0], f[1])].append(f[1])
                if current.evidence:
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
                # if feat[0]=='prob':
                # dict_feat_stats[feat] = (n_samples, 1-feat[2], feat[2])
                # else:
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
            if self.dict_unlabvar_feature_evis.get(unlabel_var.id):
                for j in range(self.opt.lebel_dim):
                    unlabel_var_cp = copy.deepcopy(unlabel_var)
                    unlabel_var_cp.class_id= j
                    mass_functions_tmp = []
                    for (feat_name, n_samples, neg_prob, pos_prob) in self.dict_unlabvar_feature_evis.get(unlabel_var.id):
                        if feat_name[1]!=j:continue
                        # mass_functions_tmp.append(self.construct_mass_function_for_propensity(cosine_evi_uncer_degree,max(pos_prob, neg_prob),min(pos_prob, neg_prob)))
                        mass_functions_tmp.append(self.construct_mass_function_for_propensity(cosine_evi_uncer_degree,pos_prob,neg_prob))
                    if len(mass_functions_tmp) > 0:
                        dict_unlabvar_propensity_masses[unlabel_var_cp] = mass_functions_tmp

        dict_unlabvar_propen_combined_masses = {unlabel_var: self.labeling_propensity_with_ds(mass_funcs)
                                                for unlabel_var, mass_funcs in dict_unlabvar_propensity_masses.items()}

        return dict_unlabvar_propen_combined_masses


    def getNoisy(self, top_m=None, save=False):
        dict_noisy_evi_support = self.get_unlabvar_evi_support()
        sorted_unlabvar_evi_support = sorted(dict_noisy_evi_support.items(), key=lambda tuple: tuple[1]['l'],
                                             reverse=True)
        top_varibal_m = [var for (var, propens) in sorted_unlabvar_evi_support[:int((len(sorted_unlabvar_evi_support)*top_m)/self.opt.lebel_dim)]]

        if save:
            logging.info('save less risky data')
            data_to_save=[]
            labels = json.load(open('../../datasets/{0}/{1}/labels.json'.format(self.opt.task, self.opt.dataset)))
            for i in tqdm(range(len(top_varibal_m))):
                current = top_varibal_m[i]
                if self.opt.task == 'STS':
                    tem = {'text': current.text, 'text2':current.text2, 'label': labels[current.pred],
                           'ori_label': labels[current.label]}
                else:
                    tem = {'text': current.text, 'label': labels[current.pred],
                           'ori_label': labels[current.label]}
                data_to_save.append(tem)

            path_ = os.path.join(self.opt.save_dir,'DST_filtered_{0}_{1}_{2}_.json'.format(self.opt.dataset, self.opt.plm,str(self.opt.train_sample)))
            json.dump(data_to_save, open(path_, 'w'), indent=3)


            predicted = [d['label'] for d in data_to_save]
            true_label = [d['ori_label'] for d in data_to_save]
            f1 = metrics.f1_score(true_label, predicted, average='macro')
            acc = metrics.accuracy_score(true_label, predicted)
            logging.info(metrics.classification_report(true_label, predicted))
            logging.info('statistics on the filtered data ')
            logging.info(' Macro-f1 :{} accuracy {}, size of filltered data: {} '.format(f1, acc, len(data_to_save) ))






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




