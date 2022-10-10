import tqdm
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, ttest_ind_from_stats, ttest_ind
from functools import reduce
from scipy.stats import hmean
import torch

def generate_table(outputs_tr, outputs_te, names, selected_m=10, max_m=45, total_inner_rep=100, order=None):
    def get_p_mean_diff(outputs_train, outputs_test):
        pred_test = outputs_test[:,0].detach().cpu().numpy()
        pred_train = outputs_train[:,0].detach().cpu().numpy()
        tval, pval = ttest_ind(pred_test, pred_train, alternative="greater", equal_var=False)
        if pval < 0:
            ipdb.set_trace()
        return pval, (pred_test.mean() - pred_train.mean())

    def get_p_values_mean_diffs(num_ex, train, test, k):
        total = train.shape[0]
        sum_p = 0
        p_values = []
        diffs = []
        for i in range(k):
            positions = torch.randperm(total)[:num_ex]
            p_val,  mean_diff = get_p_mean_diff(train[positions], test[positions])
            p_values.append(p_val)
            diffs.append(mean_diff)
        return p_values, diffs
    
    name2p_val_mean_diff = {}

    n_pbar = tqdm.notebook.tqdm(names, leave=False)
    for name in n_pbar:
        p_values_list = []
        p_list, diffs = get_p_values_mean_diffs(selected_m, outputs_tr[name], outputs_te[name], total_inner_rep)
        try:
            hm = hmean(p_list)
        except:
            hm = 1.0
        diff = np.mean(diffs)
        name2p_val_mean_diff[name] = [hm, diff]
          
    tab = pd.DataFrame(name2p_val_mean_diff, index=["p_value", "mean_diff"]).T
    tab = tab[["mean_diff", "p_value"]]
    if order is None:
        order = ['teacher',
         'distillation',
         'pre-act-18',
         'zero-shot',
         'fine-tune',
         'extract-label',
         'extract-logit',
         'independent']

    tab = tab.loc[order]
    return tab
