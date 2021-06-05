import glob
import pandas as pd
from pdb import set_trace

dataname = # NAME OF DATASET ('train' or 'test')
path = # PATH WITH RESULTS FILES
system_name = # NAME OF TESTED SYSTEM

df = pd.read_csv('Annotation_' + dataname + '.csv', delimiter='\t', header=(0))
print(df.head())

# DROP EMPTY SENTENCES
dfs = []
paths = []
for i in range(int(len(df)/15)):
  dfs.append(df[i*15:(i+1)*15])
  paths.append(df['Filepath'][i*15])
print(len(dfs))
import numpy as np
for df in dfs:
  df.dropna(subset=['Texts'], inplace=True)
df = pd.concat(dfs)
df.to_csv('Noempty_%s.csv' % dataname, sep='\t')

layers = ['0', '1', 'mean']

fileids = []
fileids.append("")
for row in df['Filepath']:
  tempid = row.split('/')[-1]
  if fileids[-1] != tempid:
    fileids.append(tempid)
fileids.pop(0)
print(fileids, len(fileids))

l0 = []
l1 = []
lmean = []

lista = [l0, l1, lmean]

for lindex, layer in enumerate(layers):
  path = path + '/' + dataname + '/Ranks_L' + layer + '_H0/'
  for index, fileid in enumerate(fileids):
    with open(path + fileid, 'r') as f:
      lines = f.readlines()
      lines = lines[4:7]
      summary = [line.split('\t')[1] for line in lines]
      for text in df['Texts'][index*15:(index+1)*15]:
        if text in summary:
          lista[lindex].append(1)
        else:
          lista[lindex].append(0)

for lindex, layer in enumerate(layers):
  df[system_name + '_L' + layer] = lista[lindex]

df.to_csv(system_name + '_' + dataname + '.csv', sep='\t')