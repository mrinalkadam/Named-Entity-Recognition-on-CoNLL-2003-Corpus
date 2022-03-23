######################################### TASK 3 #########################################
print("######################################### TASK 3 #########################################")

# import required libraries and methods from them

from platform import python_version

import random
import os
import csv
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from time import time

start = time()

# check the python version being used by the jupyter notebook
python_version()

# define constants
PADDING = "<PADDING>"
UNKNOWN = "<UNKNOWN>"

# define parameters and hyper-parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 100

# change below parameters according to Task 1,2 or 3
# Task 1 - False, False
# Task 2 - True, False
# Task 3 - True, True

glove_embeddings_to_be_used = True
cnn_for_character_level_embedding_to_be_used = True

embedding_dimension = 100
character_embedding_dimension = 30
lstm_hidden_dimension = 256
dropout = 0.33
output_dimension = 128
out_channels = 30

# hyper-parameters
num_epochs = 100
batch_size = 16
learning_rate = 0.015
momentum = 0.9
decay_rate = 0.05
gradient_clip = 5.0

# define loss function
loss_fn = torch.nn.CrossEntropyLoss

# model file name
model_name = 'bilstm3.pt'

# initialize
embeddings = None

# labels
label_to_id = {}
id_to_label = []
size_of_labels = 0

# words
word_to_id = {}
id_to_word = []

# characters
char_to_id = {}
id_to_char = []

# training or testing(saved model will be used directly without any training)
to_be_tested_flag = True


class Data:
    words = None
    original_words = None
    tags = None
    output = None
    ids_of_words = None
    ids_of_chars = None
    ids_of_tags = None
    ids_of_output = None

    def __init__(self, words, original_words, tags=None):
        self.words = words
        self.original_words = original_words
        self.tags = tags


def read_data_input_file(file_name):
    print("\nFile being read: " + file_name)
    rows = []
    with open(file_name, 'r', encoding='utf-8') as f:
        words = []
        original_words = []
        labels = []
        for line in f.readlines():
            line = line.rstrip()
            if line == "":
                rows.append(Data(words, original_words, labels))
                original_words = []
                words = []
                labels = []
                continue
            row = line.split()
            original_words.append(row[1])
            word = row[1]
            # make any digit 0
            word = re.sub('\d', '0', word)
            words.append(word)
            if len(row) == 2:
                labels = None
            else:
                labels.append(row[2])
        if len(words) > 0:
            rows.append(Data(words, original_words, labels))
    print("No. of sentences in the file: {}".format(len(rows)))

    return rows


def read_embedding_file(embedding_file, embedding_dimension):
    embedding = {}
    with open(embedding_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split()
            assert (embedding_dimension + 1 == len(words))
            embedd = np.empty([1, embedding_dimension])
            embedd[:] = words[1:]
            first_column = words[0]
            embedding[first_column] = embedd

    return embedding


def label_id_building(dataset):
    label_to_id[PADDING] = len(label_to_id)
    id_to_label.append(PADDING)
    for data in dataset:
        for label in data.tags:
            if label not in label_to_id:
                id_to_label.append(label)
                label_to_id[label] = len(label_to_id)
    size_of_labels = len(label_to_id)

    print("\nNo. of unique labels: {}".format(size_of_labels))
    print("Label to ID: {}".format(label_to_id))


def word_id_building(train):
    word_to_id[PADDING] = 0
    id_to_word.append(PADDING)
    word_to_id[UNKNOWN] = 1
    id_to_word.append(UNKNOWN)

    char_to_id[PADDING] = 0
    id_to_char.append(PADDING)
    char_to_id[UNKNOWN] = 1
    id_to_char.append(UNKNOWN)

    # extract char on train
    training_set = train
    for data in training_set:
        for word in data.words:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
                id_to_word.append(word)

    # extract char only on train
    for data in train:
        for word in data.words:
            for char in word:
                if char not in char_to_id:
                    char_to_id[char] = len(id_to_char)
                    id_to_char.append(char)

    print("\nVocab size: %d \n" % len(word_to_id))


def embedding_table_building():
    global embeddings

    scale = np.sqrt(3.0 / embedding_dimension)
    if training_embeddings is not None:
        embeddings = np.empty([len(word_to_id), embedding_dimension])
        for word in word_to_id:
            if word in training_embeddings:
                embeddings[word_to_id[word], :] = training_embeddings[word]
            elif word.lower() in training_embeddings:
                embeddings[word_to_id[word], :] = training_embeddings[word.lower()]
            else:
                embeddings[word_to_id[word], :] = np.random.uniform(-scale, scale, [1, embedding_dimension])
    else:
        embeddings = np.empty([len(word_to_id), embedding_dimension])
        for word in word_to_id:
            embeddings[word_to_id[word], :] = np.random.uniform(-scale, scale, [1, embedding_dimension])


def row_id_mapping(dataset):
    for data in dataset:
        words = data.words
        data.ids_of_words = []
        data.ids_of_chars = []
        if data.tags:
            data.ids_of_tags = []
        else:
            None
        for word in words:
            if word in word_to_id:
                data.ids_of_words.append(word_to_id[word])
            else:
                data.ids_of_words.append(word_to_id[UNKNOWN])
            char_id = []
            for char in word:
                if char in char_to_id:
                    char_id.append(char_to_id[char])
                else:
                    char_id.append(char_to_id[UNKNOWN])
            data.ids_of_chars.append(char_id)
        if data.tags:
            for label in data.tags:
                if label in label_to_id:
                    data.ids_of_tags.append(label_to_id[label])
                else:
                    data.ids_of_tags.append(label_to_id['O'])


def data_batching_list(data):
    train_num = len(data)
    if train_num % batch_size != 0:
        batch_total = train_num // batch_size + 1
    else:
        train_num // batch_size
    data_batched = []
    for batch_id in range(batch_total):
        data_in_one_batch = data[batch_id * batch_size:(batch_id + 1) * batch_size]
        data_batched.append(batching(data_in_one_batch))

    return data_batched


def batching(data):
    batch_size = len(data)
    batch_data = data

    length_of_word_sequence = torch.LongTensor(list(map(lambda row: len(row.words), batch_data)))
    max_length_of_word_sequence = length_of_word_sequence.max()

    length_of_character_sequence = torch.LongTensor(
        [list(map(len, row.words)) + [1] * (int(max_length_of_word_sequence) - len(row.words)) for row in batch_data])
    max_length_of_character_sequence = length_of_character_sequence.max()

    word_sequence_tensor = torch.zeros((batch_size, max_length_of_word_sequence), dtype=torch.long)
    label_sequence_tensor = torch.zeros((batch_size, max_length_of_word_sequence), dtype=torch.long)
    char_sequence_tensor = torch.zeros((batch_size, max_length_of_word_sequence, max_length_of_character_sequence),
                                       dtype=torch.long)

    for index in range(batch_size):
        word_sequence_tensor[index, :length_of_word_sequence[index]] = torch.LongTensor(batch_data[index].ids_of_words)
        if batch_data[index].ids_of_tags:
            label_sequence_tensor[index, :length_of_word_sequence[index]] = torch.LongTensor(
                batch_data[index].ids_of_tags)

        for word_index in range(length_of_word_sequence[index]):
            char_sequence_tensor[index, word_index,
            :length_of_character_sequence[index, word_index]] = torch.LongTensor(
                batch_data[index].ids_of_chars[word_index])
        for word_index in range(length_of_word_sequence[index], max_length_of_word_sequence):
            char_sequence_tensor[index, word_index, 0: 1] = torch.LongTensor([char_to_id[PADDING]])

    word_sequence_tensor = word_sequence_tensor.to(device)
    label_sequence_tensor = label_sequence_tensor.to(device)
    char_sequence_tensor = char_sequence_tensor.to(device)

    length_of_word_sequence = length_of_word_sequence.to(device)

    return word_sequence_tensor, length_of_word_sequence, char_sequence_tensor, label_sequence_tensor, data


# this class implements a simple BiLSTM or BiLSTM_CNN model based on parameters chosen at the top
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.vocab_size = len(word_to_id)
        self.tags_size = len(label_to_id)

        if cnn_for_character_level_embedding_to_be_used:
            self.character_embedds = nn.Embedding(len(char_to_id), character_embedding_dimension)
            torch.nn.init.xavier_uniform_(self.character_embedds.weight)
            self.char_cnn3 = nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=(3, character_embedding_dimension),
                padding=(2, 0),
            )

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dimension)
        self.word_embeds.weight = nn.Parameter(torch.FloatTensor(embeddings))
        self.lstm_dropout = nn.Dropout(dropout)

        if cnn_for_character_level_embedding_to_be_used:
            self.lstm = nn.LSTM(embedding_dimension + out_channels, lstm_hidden_dimension, num_layers=1,
                                batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(embedding_dimension, lstm_hidden_dimension, num_layers=1, batch_first=True,
                                bidirectional=True)

        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
        self.lstm_out_dropout = nn.Dropout(dropout)
        self.lstm_to_linear = nn.Linear(lstm_hidden_dimension * 2, output_dimension)
        self.linearELU = nn.ELU()
        self.hidden_to_tag = nn.Linear(output_dimension, self.tags_size)
        nn.init.xavier_normal_(self.hidden_to_tag.weight.data)
        nn.init.normal_(self.hidden_to_tag.bias.data)

    def forward(self, word_sequence_tensor: torch.Tensor, word_sequence_length: torch.Tensor,
                character_sequence_tensor: torch.Tensor):
        word_embedds = self.word_embeds(word_sequence_tensor)

        if cnn_for_character_level_embedding_to_be_used:
            batch_size = character_sequence_tensor.size(0)
            sent_length = character_sequence_tensor.size(1)
            character_sequence_tensor = character_sequence_tensor.view(batch_size * sent_length, -1)
            chars_embedds = self.character_embedds(character_sequence_tensor).unsqueeze(1)
            cnn_output = self.char_cnn3(chars_embedds)
            chars_embedds = nn.functional.max_pool2d(cnn_output, kernel_size=(cnn_output.size(2), 1)).view(
                cnn_output.size(0),
                out_channels)

            character_features = chars_embedds.view(batch_size, sent_length, -1)
            word_embedds = torch.cat([word_embedds, character_features], 2)

        words_rem = self.lstm_dropout(word_embedds)
        sorted_seq_len, index = word_sequence_length.sort(0, descending=True)
        _, sorted_index = index.sort(0, descending=False)
        sorted_seq_tensor = words_rem[index]
        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        output, _ = self.lstm(packed_words, None)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[sorted_index]
        output = self.lstm_out_dropout(output)
        output = self.lstm_to_linear(output)
        output = self.linearELU(output)
        output = self.hidden_to_tag(output)

        return output


def model_evaluation(model: BiLSTM, data, best_f_score, dataset="Train"):
    new_f_score = 0.0
    save = False

    prediction_file_name = dataset + '3.out'
    score_file_name = dataset + '3_score.out'

    prediction_file = open(prediction_file_name, 'w')
    for index in range(len(data)):
        batch = data[index]
        rows = batch[4]
        prediction_scores = model(*batch[:3])
        prediction_tags = prediction_scores.argmax(-1)
        for row, predictions in zip(rows, prediction_tags):
            for index, (word, gold, pred) in enumerate(zip(row.original_words, row.tags, predictions), start=1):
                prediction_file.write(' '.join([str(index), word, gold, id_to_label[pred]]))
                prediction_file.write('\n')
            prediction_file.write('\n')
    prediction_file.close()

    # comparison with official evaluation script using perl
    os.system('perl conll03eval.txt < %s > %s' % (prediction_file_name, score_file_name))
    evaluation_lines = [l.rstrip() for l in open(score_file_name, 'r', encoding='utf8')]

    for i, line in enumerate(evaluation_lines):
        print(line)
        if i == 1:
            new_f_score = float(line.strip().split()[-1])
            if new_f_score > best_f_score:
                best_f_score = new_f_score
                save = True
                print('Best F1-score is ', new_f_score)

    print("%s: New F1-score: %f Best F1-score: %f " % (dataset, new_f_score, best_f_score))

    return best_f_score, new_f_score, save


def train_model(model: BiLSTM, batches_of_train_data, batches_of_dev_data):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    losses = []
    best_dev_f_score = -1.0
    best_train_f_score = -1.0
    f_scores = csv.writer(open('f_scores_3.csv', 'w'))
    f_scores.writerow(['epoch', 'train', 'dev'])
    loss_function = loss_fn(ignore_index=label_to_id[PADDING])

    for epoch in range(1, num_epochs + 1):
        print("\nEpoch %d" % epoch)
        epoch_loss = 0
        model.zero_grad()
        for index in np.random.permutation(len(batches_of_train_data)):
            model.train()
            batch = batches_of_train_data[index]
            prediction_scores = model(*batch[:3])
            output = prediction_scores.view(-1, prediction_scores.shape[-1])
            gold = batch[3].view(-1)
            loss = loss_function(output, gold)
            epoch_loss = epoch_loss + loss.data
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            model.zero_grad()

        print("     Epoch Loss:", float(epoch_loss))
        losses.append(epoch_loss)

        # evaluating on train & dev sets
        model.eval()
        best_train_f_score, new_train_f, _ = model_evaluation(model, batches_of_train_data, best_train_f_score, "train")
        best_dev_f_score, new_dev_f, save = model_evaluation(model, batches_of_dev_data, best_dev_f_score, "dev")

        if save:
            print("\nSaving the model to ", model_name)
            torch.save(model.state_dict(), model_name)

        f_scores.writerow([epoch, new_train_f, new_dev_f])
        model.zero_grad()

        # performing decay(adjusting) on the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate / (1 + decay_rate * epoch)

    return None


def tag_prediction(model: BiLSTM, data, file_name):
    with open(file_name, "w") as file:
        for batch in data:
            data = batch[4]
            prediction_scores = model(*batch[:3])
            prediction_tags = prediction_scores.argmax(-1)

            for row, predictions in zip(data, prediction_tags):
                for index, (word, pred) in enumerate(zip(row.original_words, predictions), start=1):
                    file.write(' '.join([str(index), word, id_to_label[pred]]))
                    file.write('\n')
                file.write('\n')


# based on whether random embeddings/glove embeddings are to be used, read the embeddings file
training_embeddings = None
if glove_embeddings_to_be_used:
    print("Glove embeddings")
    training_embeddings = read_embedding_file("glove.6B.100d.txt", embedding_dimension)
else:
    print("Random embeddings")

# main function
if __name__ == "__main__":

    # read and pre-process the data
    train = read_data_input_file("data/train")
    dev = read_data_input_file("data/dev")
    test = read_data_input_file("data/test")

    # build metadata
    label_id_building(train)
    word_id_building(train)
    embedding_table_building()
    row_id_mapping(train)
    row_id_mapping(dev)
    row_id_mapping(test)

    random.seed(seed)
    random.shuffle(train)

    model = BiLSTM()
    print(model)

    train_batches = data_batching_list(train)
    dev_batches = data_batching_list(dev)
    test_batches = data_batching_list(test)

    model.to(device)
    if not to_be_tested_flag:
        train_model(model, train_batches, dev_batches)

    # reload the model with the best dev f1-score
    model.load_state_dict(torch.load(model_name))
    model.eval()

    tag_prediction(model, dev_batches, 'dev3.out')
    tag_prediction(model, test_batches, 'pred')

end = time()
print("\nTime taken to run the Jupyter NB:", (end - start))
