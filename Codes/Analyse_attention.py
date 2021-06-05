import os
import glob
import tabulate
import numpy as np
import seaborn as sns
from operator import add
from pdb import set_trace
from matplotlib import pyplot as plt

# ------------------------------ CONSTANTS -------------------------------

#region
layers = 2
n_heads = 1
num_max_sentences = 15
max_sentence_length = 32
dataname = input('Which set (train/test)? ').lower()

path = # PATH TO THE RESULTS
path = path + '/' + dataname' + '/'
filenames = glob.glob(path + '*.txt')
filenames.sort(key=lambda v: v.upper())
CMAP = 'Blues'
sns.set()
#endregion

# ------------------------------ READING FILES -------------------------------

def reading_attention_files():
  all_sentences = []
  all_labels = []
  all_predictions = []
  all_probs = []
  id_list = []
  for k, filename in enumerate(filenames):
    if k % 100 == 0:
      print(k)
    with open(filename, 'r') as f:
      lines = f.readlines()
      sentences = []
      for l, line in enumerate(lines):
        if line == '----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----\n':
          break
        try:
          sent = line.split('\n')[0]
          sentence = sent.split('\t')[1]
          sentences.append(sentence)
        except Exception as e:
          new_sent = sentences[-1] + ' ' + sent
          sentences[-1] = new_sent
      label = int(lines[l+1].split('\n')[0].split('_')[1])
      prediction = int(lines[l+2].split('\n')[0].split('\t')[1])
      probs = lines[l+3].split('\n')[0].split('\t')
      probability = (float(probs[1]), float(probs[-1]))

      all_sentences.append(sentences)
      all_labels.append(label)
      all_predictions.append(prediction)
      all_probs.append(probability)

      offset = l+7
      offset2 = 3
      layer_list = []
      for layer in range(layers):
        head_list = []
        for head in range(n_heads):
          rows = lines[offset:offset+num_max_sentences]
          offset += (num_max_sentences) + offset2
          float_list = []
          for row in rows:
            float_list.append([float(x) for x in row.split('\n')[0].split('  ')])
          head_list.append(float_list)
        layer_list.append(head_list)
      id_list.append(layer_list)

  print(len(id_list))
  return id_list, all_sentences, all_labels, all_predictions, all_probs

# ------------------------------ DATA MANAGEMENT -------------------------------

#region
def column_mean(table, sentences):
  somma = list(np.zeros(num_max_sentences))
  for row in table:
    somma = list(map(add, somma, row))
  media = np.array([elem/num_max_sentences for elem in somma])
  neg_media = -media
  ord_index = np.argsort(neg_media)
  output = []
  for index in ord_index:
    output.append((index, sentences[index], str(media[index])))
  return output, media

def matrix_sum(tables):
  result = np.zeros((num_max_sentences, num_max_sentences))#tables[0]
  for table in tables:
    # iterate through rows
    for i in range(len(tables[0])):
      # iterate through columns
      for j in range(len(tables[0][0])):
          result[i][j] = result[i][j] + table[i][j]
  return result

def matrix_mean(tables):
  matrix = matrix_sum(tables)
  mean = np.array([elem/len(tables) for elem in matrix])
  return mean

def manage_data(id_list, all_sentences):
  final_outputs = []
  final_tables = []
  global LAYER
  global HEAD
  repeat = True
  while repeat:
    layer = input('Which layer (max=%d, all, mean)? ' % (layers-1)).lower()
    head = input('Which head (max=%d, all, mean)? ' % (n_heads-1)).lower()
    for i, sample in enumerate(id_list):
      if i % 100 == 0:
        print(i)
      if layer not in ['all', 'mean'] and head not in ['all', 'mean']:
        layer = int(layer)
        head = int(head)
        table = sample[layer][head]
        output, _ = column_mean(table, all_sentences[i])
        final_outputs.append(output)
        final_tables.append(table)
        repeat = False

      elif layer in ['all', 'mean'] and head not in ['all', 'mean']:
        head = int(head)
        tables = []
        for l in range(len(sample)):
          tables.append(sample[l][head])
        if layer == 'mean':
          matrix = matrix_mean(tables)
          final_tables.append(matrix)
          output, _ = column_mean(matrix, all_sentences[i])
          final_outputs.append(output)
        else:
          outputs = []
          for table in tables:
            output, _ = column_mean(table, all_sentences[i])
            outputs.append(output)
          final_outputs.append(outputs)
        
        repeat = False

      elif layer not in ['all', 'mean'] and head in ['all', 'mean']:
        layer = int(layer)
        tables = []
        for h in range(len(sample[layer])):
          tables.append(sample[layer][h])
        if head == 'mean':
          matrix = matrix_mean(tables)
          final_tables.append(matrix)
          output, _ = column_mean(matrix, all_sentences[i])
          final_outputs.append(output)
        else:
          outputs = []
          for table in tables:
            output, _ = column_mean(table, all_sentences[i])
            outputs.append(output)
          final_outputs.append(outputs)
        repeat = False

      else:
        print('Error. Please, try again.')
  LAYER = layer
  HEAD = head
  return final_outputs
#endregion

# ------------------------------ WRITING FILES -------------------------------

def writing_ranking_files(final_outputs, sep='N', all_labels=None, all_predictions=None, all_probs=None):
  new_path = path + 'Ranks_L' + str(LAYER) + '_H' + str(HEAD) + '/'
  try:
      os.mkdir(new_path)
  except OSError:
      print ("Creation of the directory %s failed" % new_path)
  else:
      print ("Successfully created the directory %s" % new_path)

  if sep == 'Y' or sep == 'y':
    for k, filename in enumerate(filenames):
      if k % 100 == 0:
        print(k)
      fileid = filename.split('/')[-1]
      with open(new_path + fileid, 'w') as f:
        f.write(fileid + '\n')
        if all_labels != None:
          f.write('Label\t' + str(all_labels[k]) + '\n')
        if all_predictions != None:
          f.write('Prediction\t' + str(all_predictions[k]) + '\n')
        if all_probs != None:
          f.write('Probabilities\t' + str(all_probs[k]) + '\n')
        
        sample = final_outputs[k]
        
        for rank in sample:
          f.write(str(rank[0]) + '\t' + rank[1] + '\t' + str(rank[2]) + '\n')
        f.write('--------------------\n')

  else:
    with open(new_path + 'full.txt', 'w') as f:
      for k, filename in enumerate(filenames):
        if k % 100 == 0:
          print(k)
        fileid = filename.split('/')[-1]
        f.write(fileid + '\n')
        if all_labels != None:
          f.write('Label\t' + str(all_labels[k]) + '\n')
        if all_predictions != None:
          f.write('Prediction\t' + str(all_predictions[k]) + '\n')
        if all_probs != None:
          f.write('Probabilities\t' + str(all_probs[k]) + '\n')

        sample = final_outputs[k]
        if len(sample) == num_max_sentences:
          for rank in sample:
            f.write(str(rank[0]) + '\t' + rank[1] + '\t' + str(rank[2]) + '\n')
        else:
          for sample_layer in sample:
            for rank in sample_layer:
              f.write(str(rank[0]) + '\t' + rank[1] + '\t' + str(rank[2]) + '\n')
            f.write('--------------------\n')
        f.write('--------------------\n')

# ------------------------------ RANKING FILES -------------------------------

def ranking_files():
  for k, filename in enumerate(filenames):
    if k % 100 == 0:
      print(k)
    with open(filename, 'r') as f:
      with open(path + '/attention_ranks/' + filename.split(path + '/')[1], 'w') as f2:
        lines = f.readlines()
        sentences = []
        for l, line in enumerate(lines):
          if line == '----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----\n':
            break
          try:
            sent = line.split('\n')[0]
            sentence = sent.split('\t')[1]
            sentences.append(sentence)
          except Exception as e:
            new_sent = sentences[-1] + ' ' + sent
            sentences[-1] = new_sent

        label = int(lines[l+1].split('\n')[0].split('_')[1])
        prediction = int(lines[l+2].split('\n')[0].split('\t')[1])
        probs = lines[l+3].split('\n')[0].split('\t')
        probability = (float(probs[1]), float(probs[-1]))
        f2.write('Label\t'+str(label))
        f2.write('\nPrediction\t'+str(prediction))
        f2.write('\nProbability\t'+str(probability))

        offset = l+7
        offset2 = 3
        means = []
        MATRIX = []
        for layer in range(layers):
          rows = []
          for head in range(n_heads):
            rows = lines[offset:offset+num_max_sentences]
            offset += (num_max_sentences) + offset2
          somma = list(np.zeros(num_max_sentences))
          FL = []
          for row in rows:
            float_list = [float(x) for x in row.split('\n')[0].split('  ')]
            FL.append(float_list)
            somma = list(map(add, somma, float_list))
          media = np.array([elem/num_max_sentences for elem in somma])
          neg_media = -media
          ord_index = np.argsort(neg_media)
          f2.write('\n-----')
          f2.write('\nLayer\t' + str(layer))
          f2.write('\tHead\t' + str(head))
          for index in ord_index:
            f2.write('\n' +str(index) + '\t' + sentences[index] + '\t' + str(media[index]))

          means.append(media)
          MATRIX.append(FL)
        lmedia = list(map(add, means[0], means[1]))
        lmedia = np.array([elem/num_max_sentences for elem in lmedia])
        neg_media = -lmedia
        ord_index = np.argsort(neg_media)

        X = MATRIX[0]
        Y = MATRIX[1]
        result = np.zeros((len(X), len(X)))
        # iterate through rows
        for i in range(len(X)):
          # iterate through columns
          for j in range(len(X[0])):
              result[i][j] = X[i][j] + Y[i][j]
        for row in result:
          float_list = [float(x) for x in row]
          FL.append(float_list)
          somma = list(map(add, somma, float_list))
        media = np.array([elem/num_max_sentences for elem in somma])
        neg_media = -media
        ord_index = np.argsort(neg_media)
        f2.write('\n-----')
        f2.write('\nLayer\tmean\t' + 'Head\t' + str(head))
        for index in ord_index:
          f2.write('\n' +str(index) + '\t' + sentences[index] + '\t' + str(media[index]))

# ------------------------------ PLOTTING -------------------------------

def plt_attentions(sample, ex_id, labels, save_or_show='save', fig_size=(num_max_sentences,num_max_sentences), annot=True, cmap = CMAP):
  '''
  plot the NxN matrix as a heat map
  
  sample: list of square matrices
  labels: labels for xticks and yticks (the sentences in our case)
  '''
  repeat = True
  while repeat:
    layer = input('Which layer (max=%d, mean)? ' % (layers-1)).lower()
    head = input('Which head (max=%d, mean)? ' % (n_heads-1)).lower()
    title = 'Layer ' + layer + ' - Head ' + head
    if layer not in ['mean'] and head not in ['mean']:
      layer = int(layer)
      head = int(head)
      table = sample[layer][head]
      fig, ax = plt.subplots(figsize=fig_size) 
      ax = sns.heatmap(table, annot=annot, yticklabels=labels,xticklabels=labels, cmap=cmap)
      ax.xaxis.set_ticks_position('top')
      ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
      ax.set_title(title)
      repeat = False

    elif layer in ['mean'] and head not in ['mean']:
      set_trace()
      head = int(head)
      tables = []
      for l in range(len(sample)):
        tables.append(sample[l][head])
      table = matrix_mean(tables)
      fig, ax = plt.subplots(figsize=fig_size) 
      ax = sns.heatmap(table, annot=annot, yticklabels=labels,xticklabels=labels, cmap=cmap)
      ax.xaxis.set_ticks_position('top')
      ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
      ax.set_title(title)
      repeat = False

    elif layer not in ['mean'] and head in ['mean']:
      layer = int(layer)
      tables = []
      for h in range(len(sample[layer])):
        tables.append(sample[layer][h])
      table = matrix_mean(tables)
      fig, ax = plt.subplots(figsize=fig_size) 
      ax = sns.heatmap(table, annot=annot, yticklabels=labels,xticklabels=labels, cmap=cmap)
      ax.xaxis.set_ticks_position('top')
      ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
      ax.set_title(title)  
      repeat = False
    
    else:
      print('Error. Please, try again.')
    
  if save_or_show == 'save':
    new_path = path + 'Heatmaps/'
    try:
      os.mkdir(new_path)
    except OSError:
      print ("Creation of the directory %s failed" % new_path)
    else:
      print ("Successfully created the directory %s" % new_path)
    filename = new_path + 'ID' + str(ex_id) + '_' + title + '.png'
    fig.savefig(filename, bbox_inches='tight')
  else:
    fig.show()

# ------------------------------ MAIN -------------------------------

def main():#args):
  id_list, all_sentences, all_labels, all_predictions, all_probs = reading_attention_files()
  final_outputs = manage_data(id_list, all_sentences)
  sep = input('Separate output files (y/n)? ')
  writing_ranking_files(final_outputs, sep, all_labels, all_predictions, all_probs)
  sep = input('Want to plot any attention map (y/n)? ')
  if sep.lower() == 'y':
    sep = 'save'
    annot = input('Want to annot (y/n)? ').lower()
    if annot == 'y':
      annot = True
    else:
      annot = False
    while True:
      try:
        ex_id = int(input('Example id (just the number)? '))
      except Exception as e:
        print(e)
        break
      plt_attentions(id_list[ex_id], ex_id, labels=all_sentences[ex_id], save_or_show=sep, annot=annot)
  print('Bye!')

  
if __name__ == "__main__":
  print('Hello World!')
  main()