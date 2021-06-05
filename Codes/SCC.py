 # ------------------------------ IMPORTS -------------------------------
import re
import os
import sys
import argparse
import tabulate
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

from nltk.tokenize import sent_tokenize

# ------------------------------ CONSTANTS -------------------------------

#region
data = tfds.load('imdb_reviews')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
re_br = re.compile(r'<br /><br />')
BATCH_SIZE = 240
num_max_sentences = int(input('Max number of sentences? '))
max_sentence_length = 32
num_examples = -1

roberta_path = # PATH TO THE RESULTS
n_epoch = 'best' # CHANGE THIS IF YOU ARE TESTING THE MODEL WITH ANOTHER EPOCH
try:
  os.mkdir(roberta_path)
except OSError:
  print ("Creation of the directory %s failed" % roberta_path)
else:
  print ("Successfully created the directory %s" % roberta_path)
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
#endregion

# ------------------------------ MODELING -------------------------------

#region
# Building the model
def build_model(bert_model):
# Define inputs
  all_inputs = {}
  input_word_ids = tf.keras.layers.Input(shape=(max_sentence_length,), name='input_ids_%s' % '0', dtype='int32')
  all_inputs['input_ids_%s' % '0'] = input_word_ids
  mask_inputs = tf.keras.layers.Input(shape=(max_sentence_length,), name='attention_mask_%s'% '0', dtype='int32')
  all_inputs['attention_mask_%s' % '0'] = mask_inputs

# Transformer 1
  output_bert = bert_model({'input_ids': input_word_ids, 'attention_mask': mask_inputs}, output_attentions=True)
  sequence_output = output_bert.pooler_output

# Classifier
  classifier = tf.keras.layers.Dense(2, activation='softmax')(sequence_output)

# Compiling model
  model = tf.keras.Model(inputs=all_inputs, outputs=classifier)
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
  optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
  model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
  model.summary(line_length=240)
  return model

# Training the model
def training(model, train_dataset, val_dataset, epochs=10):
  # Callbacks
  es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3, verbose=1)
  mc = tf.keras.callbacks.ModelCheckpoint(roberta_path + 'weights_{epoch:03d}.h5', save_weights_only=True, period=1)
  mc2 = tf.keras.callbacks.ModelCheckpoint(roberta_path + '/weights_best.h5', save_best_only=True, save_weights_only=True, period=1)
  # Training
  history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[es, mc, mc2])
  # Saving results
  hist_df = pd.DataFrame(history.history)
  hist_csv_file = roberta_path + 'history.csv'
  with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
#endregion

# ------------------------------ MAIN -------------------------------

def main(args):
  t = args.train_mode
  p = args.test_mode
  # Check arguments
  if not any([t, p]):
    try:
      raise Exception("Mandatory argument missing")
    except Exception as e:
      print(e)
      sys.exit()

  bert_model = TFRobertaModel.from_pretrained('roberta-base')
  model = build_model(bert_model)
  # Training the model
  if t:
    train_texts, train_labels, _ = preprocessing(data['train'])
    test_texts, test_labels, _ = preprocessing(data['test'])
    train_dataset = build_dataset(train_texts, train_labels)
    test_dataset = build_dataset(test_texts, test_labels)
    print('Training...')
    training(model, train_dataset, test_dataset)
  # Testing the model
  elif p:
    model.load_weights(roberta_path + '/weights_%s' % n_epoch + '.h5')
    print("Testing...")
    while True:
      dataname = ""
      while dataname not in ['train', 'test']:
        dataname = input('Which dataset (train, test)? ')
      texts, labels, numlist = preprocessing(data[dataname])
      rec = 0
      predictions = []
      doclabel = []
      preds = []
      print('Start!')
      path = res_path + '/' + dataname
      try:
        os.mkdir(path)
      except OSError:
        print ("Creation of the directory %s failed" % path)
      else:
        print ("Successfully created the directory %s" % path)
      for k, numsentences in enumerate(numlist):
        if k % 100 == 0:
          print(k)
        if numsentences == 0:
          print(k, texts[i+rec])
        neg = 0
        pos = 0
        prediction = []
        lines = []
        for i in range(numsentences):
          processor = SentenceSplitClassificationProcessor()
          text = texts[i+rec]
          lbl = labels[i+rec]
          processor.add_examples([text], [lbl])
          example = processor.get_features(tokenizer, max_length=max_sentence_length, return_tensors="tf", num_max_sentences=1)
          example = example.batch(1)
          prediction.append(model.predict(example))
          n_pred = prediction[i][0][0]
          p_pred = prediction[i][0][1]
          neg += n_pred
          pos += p_pred
          lines.append(str(i) + ' \t' + text + '\t' +  str(n_pred) + '\t' + str(p_pred) +'\n')
        predictions.append(prediction)
        neg /= numsentences
        pos /= numsentences  
        lines.append('Label:\t' + str(lbl))
        lines.append('\nProbabities:\t' + str(neg) + '\t' + str(pos))
        rec += numsentences
        pred = 1 if pos > neg else 0
        lines.append('\nPrediction:\t' + str(pred))
        with open(path + '/ID' + str(k) + '.txt', 'wt') as f:
          f.writelines(lines)
        doclabel.append(lbl)
        preds.append(pred)
      with open(path + 'predictions.txt', 'wt') as f:
        for lbl, pred in zip(doclabel, preds):
          f.write(str(lbl) + '\t' + str(pred) + '\n')

if __name__ == "__main__":
  print('Hello World!')
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train_mode", action="store_true", help="training mode")
  parser.add_argument("-p", "--test_mode", action="store_true", help="test mode")
  args = parser.parse_args()
  main(args)  