import glob
from sklearn.metrics import classification_report

path = # PATH TO RESULTS FILES

labels = []
preds = []
probs = []
filenames = glob.glob(path + '*.txt')
filenames.sort(key=lambda v: v.upper())
probs = []
labels = []
for filename in filenames:
  with open(filename, 'r') as f:
    lines = f.readlines()
    probs.append(float(lines[-38].split('\t')[-1]))
    preds.append(float(lines[-39].split('\t')[-1]))
    labels.append(float(lines[-40].split('_')[-1]))

print(classification_report(labels, preds, digits=4))