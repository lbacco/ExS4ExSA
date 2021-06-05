# ------------------------------ IMPORTS -------------------------------
import re
import os
import sys
import argparse
import tabulate
import large_text_classifier.frasexfrase as frasexfrase
import numpy as np
import pandas as pd
import tensorflow as tf
from pdb import set_trace
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from nltk.tokenize import sent_tokenize
from transformers import TFRobertaModel, RobertaTokenizer
from transformers import TFDistilBertModel, DistilBertTokenizer, DistilBertConfig
from large_text_classifier.processors import SentenceSplitClassificationProcessor, SingleSentenceClassificationProcessor

# ------------------------------ CONSTANTS -------------------------------

#region
data = tfds.load('imdb_reviews')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
re_br = re.compile(r'<br /><br />')
BATCH_SIZE = 8
num_max_sentences = int(input('Max number of sentences? '))
max_sentence_length = 32
num_examples = -1
res_path = # PATH TO THE RESULTS
n_epoch = 'best' # CHANGE THIS IF YOU ARE TESTING THE MODEL WITH ANOTHER EPOCH
try:
  os.mkdir(res_path)
except OSError:
  print ("Creation of the directory %s failed" % res_path)
else:
  print ("Successfully created the directory %s" % res_path)
#endregion

# ------------------------------ DATASET MANAGEMENT -------------------------------

#region
# Formatting text
def format(text, verbose=False):
  ''' Replacing '<br /><br />' with newline. ''' 
  s = re_br.sub('\n', text)
  if verbose:
    print(text) 
    print(s)
  return s

# Preprocessing
def preprocessing(data):
  ''' Replacing '<br /><br />' with newline and sentence splitting.'''
  print('Preprocessing...')
  labels = [x['label'].numpy() for x in data]
  preprocessed_texts = [format(str(x['text'].numpy().decode('utf-8'))) for x in data]
  
  sentences_samples = []
  sentences_labels = []
  numlist = []
  for text, label in zip(preprocessed_texts, labels):
    sentence_list = sent_tokenize(text)
    for i, sentence in enumerate(sentence_list):
      sentences_samples.append(sentence)
      sentences_labels.append(label)
      if i == num_max_sentences-1:
        break
    numlist.append(i+1)
  print('Preprocessed!')
  return sentences_samples, sentences_labels, numlist

# Builiding a dataset
def build_dataset(texts, labels, shuffle=True):
  print('Building dataset...')
  processor = SentenceSplitClassificationProcessor(verbose=False)
  processor.add_examples(texts, labels)
  dataset = processor.get_features(tokenizer,
                                        max_length=max_sentence_length,
                                        return_tensors="tf",
                                        num_max_sentences=1)
  if shuffle:
    dataset = dataset.shuffle(100).batch(BATCH_SIZE)
  print('Dataset built!')
  return dataset

# Building datasets for training and test (and validation)
def build_train_test_datasets(train=True, test=True, val=1, return_processor=False, verbose=True):
  train_dataset = None
  test_dataset = None
  val_dataset = None
  train_processor = None
  test_processor = None
  val_processor = None
  if train:
    print('Preprocessing train texts')
    full_train_preprocessed_texts, full_train_labels = preprocessing(data['train'])
    fullsize = len(list(full_train_labels))
    size = int(val*fullsize)
    train_preprocessed_texts = full_train_preprocessed_texts[:size]
    train_labels = full_train_labels[:size]

    val_preprocessed_texts = full_train_preprocessed_texts[size:2*size]
    val_labels = full_train_labels[size:2*size]

    print('Building train dataset(s)')
    train_dataset, train_processor = build_dataset(train_preprocessed_texts, train_labels, shuffle=True)
    val_dataset, val_processor = build_dataset(val_preprocessed_texts, val_labels, shuffle=True)

    if verbose:
      print('Training set len = ', len(train_labels))
      print('Validation set len = ', len(val_labels))

  if test:
    print('Preprocessing test texts')
    test_preprocessed_texts, test_labels = preprocessing(data['test'])
    print('Building test dataset')
    test_dataset, test_processor = build_dataset(test_preprocessed_texts, test_labels)

    if verbose:
      print('Test set len = ', len(test_labels))
    
  if return_processor:
    return (train_dataset, train_processor), (test_dataset, test_processor), (val_dataset, val_processor)
  return train_dataset, test_dataset, val_dataset
#endregion

# ------------------------------ STATISTICS -------------------------------

#region
# Get the list of sentences of all the documents (where each doc is a list of sentences)
def Doc_tolistof_sents(Docs_in_sents):
  Sentences = []
  for doc in Docs_in_sents:
    for sent in doc:
      Sentences.append(sent)
  return Sentences[:num_max_sentences] #'''DA CONTROLLARE'''

# Get statistics of documents in terms of number of sentences
def sentence_statistics(texts):
  Docs_in_sents = [sent_tokenize(text) for text in texts]
  nSents = [len(doc) for doc in Docs_in_sents]
  mean = np.mean(nSents)
  std = np.std(nSents)
  return Docs_in_sents, nSents, mean, std

# Get statistics of sentences in terms of number of tokens
def token_statistics(Sentences):
  Tokens = [tokenizer.encode(sent, add_special_tokens=True) for sent in Sentences]
  nToks = [len(toks) for toks in Tokens]
  mean = np.mean(nToks)
  std = np.std(nToks)
  return Tokens, nToks, mean, std

def get_histogram(items, color='tab:blue', verbose=False):
  if type(items) != type(Counter()):
    counter = Counter(items)
  else:
    counter = items
  if verbose:
    print(counter)
  plt.figure()
  plt.bar(counter.keys(), counter.values(), color=color)
  return plt

def get_upper_bound(items, percentage=0.9, verbose=False):
  if type(items) != type(Counter()):
    counter = Counter(items)
  else:
    counter = items
  if verbose:
    print(counter)
  somma = 0
  total = sum(counter.values())
  #set_trace()
  th = percentage*total
  for key in sorted(counter):
    somma += counter[key]
    if somma >= th:
      bound = key
      break
  print(somma, th, total, bound)
  return bound

def statistics():
# Text preprocessing
  texts, _ = preprocessing(data['train'])
  print(len(texts))

# Get document statistics in terms of number of sentences
  Docs_in_sents, nSents, meanSents, stdSents = sentence_statistics(texts)
  print(meanSents, stdSents)

# Get sentences statistics in terms of number of tokens
  Sentences = Doc_tolistof_sents(Docs_in_sents)
  _, nToks, meanToks, stdToks = token_statistics(Sentences)
  print(meanToks, stdToks)

# Get the histograms
  sent_plt = get_histogram(nSents)
  sent_plt.title('Sentences histogram', color='C0')
  sent_plt.xlim((0, 80))
  sent_plt.xlabel('# sentences')
  sent_plt.ylabel('# occurences')
  sent_plt.savefig('./Statistics/Sents_histogram.png')
  
  tok_plt = get_histogram(nToks, 'tab:orange')
  tok_plt.title('Tokens histogram', color='C0')
  tok_plt.xlim((0, 100))
  tok_plt.xlabel('# tokens')
  tok_plt.ylabel('# occurences')
  tok_plt.savefig('./Statistics/Toks_histogram.png')

# Get the upper bounds based on a percentage and get the plot
  uppS = []
  uppT = []
  percentages = np.arange(0.1, 1, 0.05)
  for p in percentages:
    uppSents = get_upper_bound(nSents, percentage=p)
    uppToks = get_upper_bound(nToks, percentage=p)
    uppS.append(uppSents)
    uppT.append(uppToks)

  def get_bar(x, y, color):
    plt.figure()
    plt.bar(x, y, color=color)
    return plt
  
  uppS_plt = get_bar([str(x) for x in uppS], percentages, color='tab:purple')
  uppS_plt.title('Upper bounds sentences chart', color='C0')
  uppS_plt.xlabel('Upper bounds')
  uppS_plt.ylabel('Percentages')
  uppS_plt.yticks(percentages)
  uppS_plt.savefig('./Statistics/UppS_chart.png')

  uppT_plt = get_bar([str(x) for x in uppT], percentages, color='tab:green')
  uppT_plt.title('Upper bounds tokens chart', color='C0')
  uppT_plt.xlabel('Upper bounds')
  uppT_plt.ylabel('Percentages')
  uppT_plt.yticks(percentages)
  uppT_plt.savefig('./Statistics/UppT_chart.png')
  #set_trace()

#endregion

# ------------------------------ MODELING -------------------------------

#region
# Building the model
def build_model(bert_model, n_heads, avg_or_flat='flatten'):
# Define inputs
  sequences_outputs = []
  all_inputs = {}
  sentence_models = []
  for x in range(0, num_max_sentences):
    input_word_ids = tf.keras.layers.Input(shape=(max_sentence_length,), name='input_ids_%s' % x, dtype='int32')
    all_inputs['input_ids_%s' % x] = input_word_ids
    mask_inputs = tf.keras.layers.Input(shape=(max_sentence_length,), name='attention_mask_%s'% x, dtype='int32')
    all_inputs['attention_mask_%s' % x] = mask_inputs
  # Transformer 1
    output_bert = bert_model({'input_ids': input_word_ids, 'attention_mask': mask_inputs}, output_attentions=True)
    m = tf.keras.Model(inputs={'input_ids_%s' % x: input_word_ids, 'attention_mask_%s' %x: mask_inputs}, outputs=output_bert.attentions, name="my_test_%s" % x)
    sentence_models.append(m)
    sequences_outputs.append(output_bert.pooler_output)
  # Avg (sentence) mask
  avg_mask = tf.keras.layers.Input(shape=(num_max_sentences,), name='avg_mask', dtype='int32')
  all_inputs['avg_mask'] = avg_mask

# Transformer 2
  configuration = DistilBertConfig(n_layers=2, hidden_dim=240, n_heads=int(n_heads),
                                   dim=768)
  bert_model2 = TFDistilBertModel(configuration)
  sequences_outputs = tf.stack(sequences_outputs, axis=1)
  output_bert2 = bert_model2({'inputs_embeds': sequences_outputs},
                             output_attentions=True)

  sequence_output = output_bert2.last_hidden_state

# Merging strategies
  if avg_or_flat == 'FLATTEN':
    flat = tf.keras.layers.Flatten()(sequence_output)
    classifier = tf.keras.layers.Dense(2, activation='softmax')(flat)
  elif avg_or_flat == 'BILSTM':
    biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768))(sequence_output)
    classifier = tf.keras.layers.Dense(2, activation='softmax')(biLSTM)
  elif avg_or_flat == 'MAVG':
    avg = tf.keras.layers.GlobalAveragePooling1D()(sequence_output, mask=avg_mask)
    classifier = tf.keras.layers.Dense(2, activation='softmax', name='Dense')(avg)
  elif avg_or_flat == 'AVG':
    avg = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    classifier = tf.keras.layers.Dense(2, activation='softmax')(avg)
  
# Compiling the model
  model = tf.keras.Model(inputs=all_inputs, outputs=classifier)
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
  model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
  model.summary(line_length=240)
  return model

# Training the model
def training(model, train_dataset, val_dataset, n_heads, avg_or_flat):
  # Callbacks
  es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3, verbose=1)
  mc = tf.keras.callbacks.ModelCheckpoint(new_path + '/weights_{epoch:03d}.h5', save_weights_only=True, period=1)
  mc2 = tf.keras.callbacks.ModelCheckpoint(new_path + '/weights_best.h5', save_best_only=True, save_weights_only=True, period=1)
  # Training
  history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[es, mc, mc2])
  # Saving results
  hist_df = pd.DataFrame(history.history)
  hist_csv_file = res_path + '/history.csv'  
  with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
#endregion

# ------------------------------ MAIN -------------------------------

def main(args):
  t = args.train_mode
  p = args.test_mode
  s = args.stats_mode
  print([s, t, p], len([s, t, p]))
# Check arguments
  if not any([s, t, p, i]):
    try:
      raise Exception("Mandatory argument missing")
    except Exception as e:
      print(e)
      sys.exit()
# Statistics
  if s:
    statistics()
# Building model
  else:
    n_heads = input('How many heads per layer? ')
    avg_or_flat = ""
    while avg_or_flat not in ['FLATTEN', 'AVG', 'MAVG', 'BILSTM']:
      avg_or_flat = input('Which merging strategy?\n\
                          - FLATTEN (concatenation)\n\
                          - AVG (average)\n\
                          - MAVG (masked average)\n\
                          - BILSTM').upper()
    bert_model = TFRobertaModel.from_pretrained('roberta-base')  
    model = build_model(bert_model, n_heads, avg_or_flat.upper())
  # Training the model
    if t:
      train_dataset, test_dataset, _ = build_train_test_datasets(test=True)
      training(model, train_dataset, test_dataset, n_heads, avg_or_flat)
  # Testing the model
    elif p:
      model.load_weights('%sWeights/weights_%s.h5' % (res_path, n_epoch))
      print("TEST")
      layer_name = 'tf_distil_bert_model'
      intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                                outputs=model.get_layer(layer_name).output.attentions)
      roberta_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('tf_roberta_model').output.attentions)

      dataname = ""
      while dataname not in ['train', 'test']:
        dataname = input('Which dataset (train, test)? ')
      texts, labels = preprocessing(data[dataname])
      size = len(list(labels))
      size_reminder = int(input('Size reminder? '))
      path = res_path + '/' + dataname
      try:
        os.mkdir(path)
      except OSError:
        print ("Creation of the directory %s failed" % path)
      else:
        print ("Successfully created the directory %s" % path)
      for i in range(size_reminder, size):
        if i % 100 == 0:
          print(i)
        processor = SentenceSplitClassificationProcessor()
        processor.add_examples([texts[i]], [labels[i]])
        example = processor.get_features(tokenizer, max_length=max_sentence_length, return_tensors="tf", num_max_sentences=num_max_sentences)
        example = example.batch(1)
        prediction = model.predict(example)
        intermediate_output = intermediate_layer_model.predict(example)
        
        all_sentences = sent_tokenize(texts[i])
        sentences = all_sentences[0:min(num_max_sentences,
                                            len(all_sentences))]
        if len(sentences) < num_max_sentences:
          while len(sentences) < num_max_sentences:
            sentences.append("")

        filename = path + '/ID' + str(i) + '.txt'
        with open(filename, 'w') as f:
          for j, sentence in enumerate(sentences):
            f.write(str(j) + '\t' + sentence + '\n')
          f.write('----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----\n')
          f.write('Label_' + str(labels[i]))
          if prediction[0][0] > prediction[0][1]:
            f.write('\nPrediction\t' + '0')
          else: 
            f.write('\nPrediction\t' + '1')
          f.write('\nProbabilities\t' + str(prediction[0][0]) + '\t' + str(prediction[0][1]))
          f.write('\n----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----\n')
        # Save attention maps for every layer and for every head
          for layer, _ in enumerate(intermediate_output):
            for head, _ in enumerate(intermediate_output[layer][0]):
              f.write('Layer_' + str(layer) + '\tHead_' + str(head) + '\n')
              f.write(tabulate.tabulate(intermediate_output[layer][0][head], floatfmt=".2f"))
              f.write('\n')
  
if __name__ == "__main__":
  print('Hello World!')
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--stats_mode", action="store_true", help="statistics analysis mode")
  parser.add_argument("-t", "--train_mode", action="store_true", help="training mode")
  parser.add_argument("-p", "--test_mode", action="store_true", help="test mode")
  args = parser.parse_args()
  main(args)  
