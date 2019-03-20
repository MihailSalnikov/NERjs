from tensorflow.keras.layers import (GRU, Dense, Dropout, Embedding, Flatten,
                                     Input, Multiply, Permute, RepeatVector,
                                     Softmax)
from tensorflow.keras.models import Model

from utils import MAX_SEQUENCE_LENGTH


def make_ner_model(embedding_tensor, words_vocab_size, tags_vocab_size,
                   num_hidden_units=128, attention_units=64):
    EMBEDDING_DIM = embedding_tensor.shape[1]

    words_input = Input(dtype='int32', shape=[MAX_SEQUENCE_LENGTH])

    x = Embedding(words_vocab_size + 1,
                  EMBEDDING_DIM,
                  weights=[embedding_tensor],
                  input_length=MAX_SEQUENCE_LENGTH,
                  trainable=False)(words_input)

    outputs = GRU(num_hidden_units,
                  return_sequences=True,
                  dropout=0.5,
                  name='RNN_Layer')(x)

    # Simple attention
    hidden_layer = Dense(attention_units, activation='tanh')(outputs)
    hidden_layer = Dropout(0.25)(hidden_layer)
    hidden_layer = Dense(1, activation=None)(hidden_layer)
    hidden_layer = Flatten()(hidden_layer)
    attention_vector = Softmax(name='attention_vector')(hidden_layer)
    attention = RepeatVector(num_hidden_units)(attention_vector)
    attention = Permute([2, 1])(attention)
    encoding = Multiply()([outputs, attention])

    encoding = Dropout(0.25)(encoding)
    ft1 = Dense(num_hidden_units)(encoding)
    ft1 = Dropout(0.25)(ft1)
    ft2 = Dense(tags_vocab_size)(ft1)
    out = Softmax(name='Final_Sofmax')(ft2)

    model = Model(inputs=words_input, outputs=out)
    return model
