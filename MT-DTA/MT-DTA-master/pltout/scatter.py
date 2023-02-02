import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib

txt_real = r'affinities.txt'
txt_pre = r'preaffinities.txt'

with open(txt_real,'r') as fi:
    docs = fi.read().splitlines()
    fi.close
with open(txt_pre,'r') as fo:
    docs_pre = fo.read().splitlines()
    fo.close

dataset_real = ''.join(docs)
dataset_pre = ''.join(docs_pre)
dataset_real.split()
dataset_pre.split()
dataset_compom_real = ','.join(dataset_real.split())
dataset_compom_pre = ','.join(dataset_pre.split())
print(type(dataset_compom_real))
x = dataset_compom_real.split(',')
x_dataset = []
for str in x:
    x_dataset.append(str[:3])
y = dataset_compom_pre.split(',')
y_dataset = []
for str in y:
    y_dataset.append(str[:3])

plt.scatter(x_dataset,y_dataset,s = 20)
plt.axis([4, 11, 4, 11])
plt.xlabel('Value', fontsize = 14)
plt.ylabel('predict Value', fontsize = 14)
plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
plt.show()