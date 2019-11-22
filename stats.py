import matplotlib.pyplot as plt
from collections import defaultdict
from neptune import Session
import numpy as np

session = Session()
project = session.get_projects('csadrian')['csadrian/global-sinkhorn']

def check_crit(params, crit):
  for k, v in crit.items():
    if k not in params.keys():
      return False
    if isinstance(v, list):
      if params[k] not in v:
        return False
    elif params[k] != v:
        return False
  return True


crits={}

for z_test in {'mmd', 'sinkhorn'}:
    for z_test_scope in {'global', 'local'}:
        for wae_lambda in {0.0001, 0.001, 0.01, 0.1, 1, 10}:
            crits['{:}_{:}_{:}'.format(z_test, z_test_scope, wae_lambda)]={'z_test': z_test, 'z_test_scope' : z_test_scope, 'lambda': wae_lambda}

#channels = ['covered_area', 'mmd', 'sinkhorn_ot', 'loss_rec']
channels = ['covered_area']

all_exps = project.get_experiments()

#use interval of experiments -- 120 experiments of checkers
interval = [i for i in range(1703, 1883)]
exps = [exp for exp in all_exps if exp.id[-4]!='-' and int(exp.id[-4:]) in interval]


for exp in exps:
  exp._my_params = exp.get_parameters()

#results={'mmd-local': [], 'mmd-global': [], 'sinkhorn-local': [], 'sinkhorn-global': []}
results=[]
for key, crit in crits.items():
  res = defaultdict(list)
  for exp in exps:
    params = exp._my_params
    if not check_crit(params, crit):
      continue
   # else:
   #   print(exp.id)
    vals = exp.get_logs()
    for channel in channels:
      if channel in vals:
        res[channel].append(float(vals[channel]['y']))


  for channel in channels:
    v = np.array(res[channel])
   # if v.shape[0] > 5:
   #   print("{}: Warning: more than 5 experiments: {} using only 5 (no order assumed)".format(key, v.shape[0]))
   # v = v[:5]
    mean = np.mean(v)
    results.append([crit, mean])
    std = np.std(v)
    cnt = v.shape[0]
   # print("{:} {} mean: {:.2f}, std: {:.2f}, cnt: {}".format(key, channel, mean, std, cnt))

print(results)

results
improved_results = []
for z_test in {'mmd', 'sinkhorn'}:
    for z_test_scope in {'global', 'local'}:
        results_sliced=[result[1] for result in results if result[0]['z_test']==z_test and result[0]['z_test_scope']==z_test_scope]
        results_sliced=np.stack(results_sliced)
        optimal_value=np.max(results_sliced)
        optimal_place=np.argmax(results_sliced)
        optimal_lambda=results[optimal_place][0]['lambda']
        improved_results.append([z_test, z_test_scope, optimal_value, optimal_lambda])

print(improved_results)
