import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from neptune import Session
import numpy as np
import pandas as pd


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
channels = ['covered_area', 'loss_rec']

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

  means=[] 
  for channel in channels:
    v = np.array(res[channel])
   # if v.shape[0] > 5:
   #   print("{}: Warning: more than 5 experiments: {} using only 5 (no order assumed)".format(key, v.shape[0]))
   # v = v[:5]
    mean = np.mean(v)
    means.append(mean)
   # results.append([crit, mean])
    std = np.std(v)
    cnt = v.shape[0]
   # print("{:} {} mean: {:.2f}, std: {:.2f}, cnt: {}".format(key, channel, mean, std, cnt))
  results.append([crit, means[0], means[1]])
#print(results)

scopedict = {}
scopedict[('mmd', 'global')] = 0
scopedict[('mmd', 'local')] = 1
scopedict[('sinkhorn', 'global')] = 2
scopedict[('sinkhorn', 'local')] = 3

tables = []
improved_results = []
for z_test in {'mmd', 'sinkhorn'}:
    for z_test_scope in {'global', 'local'}:
      results_sliced=[[result[1], result[2], scopedict[(z_test, z_test_scope)]] for result in results if result[0]['z_test']==z_test and result[0]['z_test_scope']==z_test_scope]
      results_sliced=np.stack(results_sliced)
      optimal_value_area=np.max(results_sliced[0])
      optimal_value_rec=np.max(results_sliced[1])
      optimal_place_area=np.argmax(results_sliced[0])
      optimal_place_rec=np.argmax(results_sliced[1])
      optimal_lambda_area=results[optimal_place_area][0]['lambda']
      optimal_lambda_rec=results[optimal_place_rec][0]['lambda']        
      improved_results.append([z_test, z_test_scope, optimal_value_area, optimal_value_rec, optimal_lambda_area, optimal_lambda_rec])
      tables.append(results_sliced)

tables = np.stack(tables)
tables = [item for sublist in tables for item in sublist]


#color_dict = {('mmd', 'global'): 'blue', ('mmd', 'local'): 'red', ('sinkhorn', 'global'): 'green',  ('sinkhorn', 'local'): 'yellow'}

fig, ax = plt.subplots()
mmd_global = np.array([[result[1], result[2]] for result in results if result[0]['z_test']=='mmd' and result[0]['z_test_scope']=='global'])
mmd_local = np.array([[result[1], result[2]] for result in results if result[0]['z_test']=='mmd' and result[0]['z_test_scope']=='local'])
sinkhorn_global = np.array([[result[1], result[2]] for result in results if result[0]['z_test']=='sinkhorn' and result[0]['z_test_scope']=='global'])
sinkhorn_local = np.array([[result[1], result[2]] for result in results if result[0]['z_test']=='sinkhorn' and result[0]['z_test_scope']=='local'])

ax.scatter(mmd_global[:, 1], mmd_global[:, 0], c = 'blue')
ax.scatter(mmd_local[:, 1], mmd_local[:, 0], c = 'red')
ax.scatter(sinkhorn_global[:, 1], sinkhorn_global[:, 0], c = 'green')
ax.scatter(sinkhorn_local[:, 1], sinkhorn_local[:, 0], c = 'yellow')

'''
for i in range(len(results)):
  ax.scatter(x = results[i][2], y = results[i][1],
              c = color_dict[(results[i][0]['z_test'], results[i][0]['z_test_scope'])])
'''

plt.legend(['mmd_global', 'mmd_local', 'sinkhorn_global', 'sinkhorn_local'])
plt.savefig('Scatter_plt.png')


'''
best_covered = pd.DataFrame(improved_results, columns = ['z_test', 'z_test_scope', 'covered_area', 'lambda'])
print(best_covered.to_latex())
#print(best_covered)
'''

'''
points = pd.DataFrame(tables, columns = ['area', 'rec', 'param'])
points.plot.scatter(x = 'rec', y = 'area', c = 'param', colormap = 'Set1')
plt.savefig("Scatter2.png")
'''
