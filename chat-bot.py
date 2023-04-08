'''
Panagiotis Christakakis
'''

# Import Libraries

# For parsing
import numpy as np
import pandas as pd 
import random 
import glob
import json
import re
import os

# Preprocessing step Model building
from keras.layers import LSTM,Dense, Dropout, Embedding, CuDNNLSTM, Bidirectional, Embedding, Input, TimeDistributed
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
import tensorflow as tf

# Plotting
import matplotlib.pyplot as plt
# %matplotlib inline

# # For G-Drive attach
# from google.colab import drive
# drive.mount('/content/drive')

# Functions

def clean_text(txt):
  # Input: text
  # This function converts it's text input to lowercase letters
  # and then many contraction words into their corresponding ones.  
  txt = txt.lower()
  txt = re.sub(r"where's", "where is", txt)
  txt = re.sub(r"that's", "that is", txt)
  txt = re.sub(r"what's", "what is", txt)
  txt = re.sub(r"won't", "will not", txt)
  txt = re.sub(r"can't", "can not", txt)
  txt = re.sub(r"she's", "she is", txt)
  txt = re.sub(r"\'ll", " will", txt)
  txt = re.sub(r"\'ve", " have", txt)
  txt = re.sub(r"he's", "he is", txt)
  txt = re.sub(r"\'d", " would", txt)
  txt = re.sub(r"\'re", " are", txt)
  txt = re.sub(r"[^\w\s]", "", txt)
  txt = re.sub(r"i'm", "i am", txt)
  return txt

def plot_metric(history, metric):
  # Input: history of model, metrics
  # This function can plot two different graphs.
  # Training and Validation loss and accuracy
  train_metrics = history.history[metric]
  val_metrics = history.history['val_'+metric]
  epochs = range(1, len(train_metrics) + 1)
  plt.plot(epochs, train_metrics)
  plt.plot(epochs, val_metrics)
  plt.title('Training and Validation '+ metric)
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend(["train_"+metric, 'val_'+metric])
  plt.show()

# Folder specification
# %Link_To_Your_Path%
path_to_files = 'C:/Users/panos/Desktop/metalwoz-v1/metalwoz_data/'
os.chdir(path_to_files)

# Obtain all .txt path files from folder
data_folder_path = '/dialogues/*.txt'

# With glob module we obtain all the pathnames matching a specified pattern 
files_list = glob.glob(path_to_files + data_folder_path)

print('A total of', len(files_list), 'files loaded.\n')

# Parsing step

list_of_dicts = []

# Loop for each file and insert in a dictionary
for filename in files_list:
  with open(filename) as f:
      for line in f:
          list_of_dicts.append(json.loads(line))

# Create a new dict containing only useful data
new_list_of_dicts = [] 

# Create key-value pairs for useful data of conversation only.
for old_dict in list_of_dicts:
  # k == 'turns' because from there it starts the conversation.
  foodict = {k: v for k, v in old_dict.items() if (k == 'turns')} 
  new_list_of_dicts.append(foodict)

# Delete and replace to free memory
del(list_of_dicts)
list_of_dicts = new_list_of_dicts

questions = []
answers = []

matrix_greetings = ["Hey", "Hi", " "]
matrix_byes = ["Ok", " ", "Bye"]

# For each dictionary in the list
for dictionary in list_of_dicts:
  matrix_QA = dictionary['turns']
   
  # In order to split the QAs to 2 matrices (questions & answers),
  # we will use a flag to indicate if the sentence is given from the bot or from the user
  #bot_flag = True # Init

  # For each Q/A in the matrix
  # Remove "hey how can i help you"
  matrix_QA.pop(0) 

  bot_flag = False
  
  for sentence in matrix_QA:
    if bot_flag == True:
      # Used for bot's answers
      answers.append(sentence) 
      # Switch
      bot_flag = False 
      continue
    else:
      # Used for user's questions
      questions.append(sentence)
      # Switch 
      bot_flag = True 

  # The last loop (ideally) ends with a bot's answer, thus making bot_flag equal to False.
  # Although, with data visualization and exploring, we can see that this does not happen all the time.

  # Corner case: If the last answers was from the user, 
  # then we need to add one artificial 'ghost' response from the bot to make the dataset even.
  if bot_flag == True: 
    answers.append(random.choice(matrix_byes))

# If list of questions and answers aren't the same return an error.
assert len(questions) == len(answers), "ERROR: The length of the questions and answer matrices are different."
print('A total of', len(questions), 'Questions-Answers were loaded.\n')

# We will shuffle them to ensure that our bot isn't overfitting on
# limited goal-oriented dialogs like setting an alarm or a exlplaining a catalogue
# Last, but not least, this way will enrich the vocabulary of our bot.
questions, answers = shuffle(np.array(questions), np.array(answers))

sorted_ques = []
sorted_ans = []

# We'll try to keep a lower number of dialogs, based on the lenght of
# the question. We do this in order to avoid RAM errors.
# So questions with lenght of 13 words and lower will be filtered and
# create new lists for QAs.

# Define the of length questions you want to keep. 
q_len = 13

for i in range(len(questions)):
  if len(questions[i]) < q_len:
    sorted_ques.append(questions[i])
    sorted_ans.append(answers[i])

clean_ques = []
clean_ans = []

# Apply the clean_text function we defined earlier.
for line in sorted_ques:
  clean_ques.append(clean_text(line))

for line in sorted_ans:
  clean_ans.append(clean_text(line))  

for i in range(len(clean_ans)):
    clean_ans[i] = ' '.join(clean_ans[i].split()[:11])

# Delete lists to free memory
del(sorted_ans, sorted_ques, answers, questions)

# # Keep a subset of the QA lists. Only used when training
# # on Google Colab because of RAM error messages.
# NUM_DIALOGS = 30000
# clean_ques = clean_ques[:NUM_DIALOGS]
# clean_ans = clean_ans[:NUM_DIALOGS]

# Count occurences
word2count = {}

for line in clean_ques:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in clean_ans:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

## # Delete to free memory
del(word, line)

# Remove the least frequent key-value pairs from the vocabulary.
# Set a threshold of minimum value.
thresh = 5

# Create the new vocabulary.
vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1
        
# Delete to free memory
del(word2count, word, count, thresh)       
del(word_num)

# Import <Start Of Sentence> and <End Of Sentence> tokens
# in front and after each answer.
for i in range(len(clean_ans)):
    clean_ans[i] = '<SOS> ' + clean_ans[i] + ' <EOS>'

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

vocab['us'] = vocab['<PAD>']
vocab['<PAD>'] = 0

# Delete to free memory.
del(token, tokens) 
del(x)

# Invert dictionary of answers
inv_vocab = {w:v for v, w in vocab.items()}

# Delete to free memory.
del(i)

# Starting to build the encoder-decoder 
# by defining and creating its inputs.

encoder_inp = []
for line in clean_ques:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
        
    encoder_inp.append(lst)

decoder_inp = []
for line in clean_ans:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])        
    decoder_inp.append(lst)

# Delete to free memory.
del(clean_ans, clean_ques, line, lst, word)

# Ensure that all sequences in a list have 
# the same lenght with the help of padding.
encoder_inp = pad_sequences(encoder_inp, q_len, padding = 'post', truncating = 'post')
decoder_inp = pad_sequences(decoder_inp, q_len, padding = 'post', truncating = 'post')

decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:]) 

decoder_final_output = pad_sequences(decoder_final_output, q_len, padding = 'post', truncating='post')

# Delete to free memory.
del(i)

decoder_final_output = to_categorical(decoder_final_output, len(vocab))

# Model creation

enc_inp = Input(shape=(q_len, ))
dec_inp = Input(shape=(q_len, ))

VOCAB_SIZE = len(vocab)
embed = Embedding(VOCAB_SIZE + 1, output_dim = 20, input_length=q_len, trainable=True)

enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_state = True, dropout = 0.30, return_sequences = True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_state = True, dropout = 0.30, return_sequences = True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state = enc_states)

dense = Dense(VOCAB_SIZE, activation = 'softmax')

dense_op = dense(dec_op)

# Input and output of our model.
model = Model([enc_inp, dec_inp], dense_op)

# Print a summary of the created model.
model.summary()

# Compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), 
              loss='categorical_crossentropy', metrics=['acc'])

# Train the model
history = model.fit([encoder_inp, decoder_inp], decoder_final_output, 
                    epochs = 50, batch_size = 128, validation_split = 0.15)

# Save the models and it's weights.
model.save('chatbot.h5')
model.save_weights('chatbot_weights.h5')

# Plot the metrics function we defined earlier.
plot_metric(history, 'loss')
plot_metric(history, 'acc')

enc_model = Model([enc_inp], enc_states)

decoder_state_input_h = Input(shape = (400,))
decoder_state_input_c = Input(shape = (400,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = dec_lstm(dec_embed, 
                                            initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

dec_model = tf.keras.models.Model([dec_inp, decoder_states_inputs],
                                      [decoder_outputs] + decoder_states)

print("**Starting the chat**")

prepro1 = ""
while prepro1 != 'q':
    prepro1  = input("User : ")
    ## prepro1 = "Hey"

    prepro1 = clean_text(prepro1)
    ## prepro1 = "hey"

    prepro = [prepro1]
    ## prepro1 = ["hey"]

    txt = []
    for x in prepro:
        # x = "hey"
        lst = []
        for y in x.split():
            ## y = "hey"
            try:
                lst.append(vocab[y])
                ## vocab['hey'] = (i.e: 454)
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)

    ## txt = [[454]]
    txt = pad_sequences(txt, q_len, padding='post')

    ## txt = [[454,0,0,0,.........q_len]]

    stat = enc_model.predict( txt )

    empty_target_seq = np.zeros( ( 1 , 1) )
     ##   empty_target_seq = [0]

    empty_target_seq[0, 0] = vocab['<SOS>']
    ##    empty_target_seq = [255]

    stop_condition = False
    decoded_translation = ''

    while not stop_condition :

        dec_outputs , h, c= dec_model.predict([ empty_target_seq] + stat )
        decoder_concat_input = dense(dec_outputs)
        ## decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

        sampled_word_index = np.argmax( decoder_concat_input[0, -1, :] )
        ## sampled_word_index = [2]

        sampled_word = inv_vocab[sampled_word_index] + ' '

        ## inv_vocab[2] = 'hi'
        ## sampled_word = 'hi '

        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word  

        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > q_len:
            stop_condition = True 

        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        ## <SOS> - > hi
        ## hi --> <EOS>
        stat = [h, c]  

    print("Chatbot: ", decoded_translation )
    print("==============================================")