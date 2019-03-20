import argparse
import json
import os

import tensorflowjs as tfjs
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from model import make_ner_model
from utils import (MAX_SEQUENCE_LENGTH, PAD_ID, load_data,
                   make_embedding_tensor, make_sequences, make_vocab)


def export_model(model, words_vocab, tags_vocab, site_path):
    tfjs.converters.save_keras_model(
        model,
        os.path.join(site_path, './tfjs_models/ner/')
        )

    with open(os.path.join(site_path, "./vocabs.js"), 'w') as f:
        f.write('const words_vocab = {\n')
        for l in json.dumps(words_vocab)[1:-1].split(","):
            f.write("\t"+l+',\n')
        f.write('};\n')
        
        f.write('const tags_vocab = {\n')
        for l in json.dumps(tags_vocab)[1:-1].split(","):
            f.write("\t"+l+',\n')
        f.write('};')
    print('model exported to ', site_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple NER model")
    parser.add_argument('--data', type=str, default='./data',
                        help="path to directory with data files")
    parser.add_argument('--glove', type=str, default='./glove.6B',
                        help="path to directory with glove data")
    parser.add_argument('--num_hidden_units', type=int, default=128,
                        help="num GRU units")
    parser.add_argument('--attention_units', type=int, default=64,
                        help="num hidden states in simple attention")
    parser.add_argument('--epoches', type=int, default=30,
                        help="num epoches for traning")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size for traning")
    parser.add_argument('--site_path', type=str, default='nodejs_brows/static',
                        help='path to your site for storing model')
    args = parser.parse_args()


    train_data = load_data(os.path.join(args.data, 'train.txt'))
    valid_data = load_data(os.path.join(args.data, 'valid.txt'))

    words_vocab = make_vocab(train_data['words'])
    tags_vocab = make_vocab(train_data['tags'])

    train_data['words_sequences'] = make_sequences(train_data['words'], words_vocab)
    valid_data['words_sequences'] = make_sequences(valid_data['words'], words_vocab)

    train_data['tags_sequences'] = make_sequences(train_data['tags'], tags_vocab)
    valid_data['tags_sequences'] = make_sequences(valid_data['tags'], tags_vocab)

    train_X = pad_sequences(train_data['words_sequences'],
                            maxlen=MAX_SEQUENCE_LENGTH,
                            value=PAD_ID, padding='post',
                            truncating='post')
    valid_X = pad_sequences(valid_data['words_sequences'],
                            maxlen=MAX_SEQUENCE_LENGTH,
                            value=PAD_ID,
                            padding='post',
                            truncating='post')

    train_y = pad_sequences(train_data['tags_sequences'],
                            maxlen=MAX_SEQUENCE_LENGTH,
                            value=PAD_ID,
                            padding='post',
                            truncating='post')
    valid_y = pad_sequences(valid_data['tags_sequences'],
                            maxlen=MAX_SEQUENCE_LENGTH,
                            value=PAD_ID,
                            padding='post',
                            truncating='post')

    train_y = to_categorical(train_y)
    valid_y = to_categorical(valid_y)


    embedding_tensor = make_embedding_tensor(args.glove, words_vocab)
    model = make_ner_model(embedding_tensor,
                           len(words_vocab), len(tags_vocab),
                           args.num_hidden_units, args.attention_units)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['categorical_accuracy']
        )
    model.fit(train_X, train_y,
              epochs=args.epoches,
              batch_size=args.batch_size,
              validation_data=(valid_X, valid_y))

    export_model(model, words_vocab, tags_vocab, args.site_path)
