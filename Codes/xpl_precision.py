import argparse
import numpy as np
import pandas as pd
from pdb import set_trace

print('Hello World!')
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--filename", default=None, help="name of the .ods file.")
args = parser.parse_args()

systems = # LIST OF NAMES OF THE SYSTEMS TO TEST
layers = ['0', '1', 'mean']

name = str(args.filename) # NAME OF THE DATASET ('train' or 'test')
df = pd.read_csv(name, sep='\t', header=(0))

if name == 'train':
  n_docs = 50
else:
  n_docs = 100

summaries = []
n_sentences = []
counts = []
for i in range(3):
  summaries.append([])
  n_sentences.append([])
  counts.append([])

fileid = []

for doc_index in range(n_docs):
  start = doc_index * 15
  end = (doc_index + 1) * 15
  document = df[start:end]
  fileid.append(df['Filepath'][start].split('/')[-1])
  for i in range(3):
    summary = document.loc[df['>%d' % i] == 1]
    summaries[i].append(summary)
    n_sentences[i].append(len(summaries[i][doc_index]))

for i in range(3):
  for doc_index in range(n_docs):
    for system in systems:
      for layer in layers:
        path = 'Results/' + system + '/' + name + '/Ranks_L' + layer + '_H0/'
        with open(path + fileid[doc_index], 'r') as f:
          lines = f.readlines()
          lines = lines[4:-1] #4+n_sentences[i][doc_index]]
          sentences = [line.split('\t')[1] for line in lines]
          sentences = [sent for sent in sentences if str(sent) != '']
          machine_sum = sentences[:n_sentences[i][doc_index]] 
          df = summaries[i][doc_index]
          df[system + '_L' + layer] = machine_sum

dizlist = []
for i in range(3):
  dizlist.append({})

for i in range(3):
  for system in systems:
    for layer in layers:
      dizlist[i][system + '_L' + layer] = []

for i in range(3):
  for doc_index in range(n_docs):
    human_sum = np.array(summaries[i][doc_index]['Texts'])
    for system in systems:
      for layer in layers:
        machine_sum = summaries[i][doc_index][system + '_L' + layer]
        count = 0
        for sentence in machine_sum:
          if sentence in human_sum:
            count += 1
        dizlist[i][system + '_L' + layer].append(count)

for i in range(3):
  precisions = []
  df = pd.DataFrame()
  for key in dizlist[i]:
    for j in range(n_docs):
      if n_sentences[i][j] != 0:
        precisions.append(dizlist[i][key][j] / n_sentences[i][j])
    precision = np.mean(precisions)
    print(key, precision)
    df[key] = [precision]
  df.to_csv('./' + name + '_summaries/Atleast_' + str(i) + '/precisions_atleast_' + str(i) + name + '.csv', sep='\t', index=False)