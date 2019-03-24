# NERjs
## Keras and tensorflow.js

In order to prepare all requirements: 
it will download glove.6b and instal all pip requirements to your current virtual env.
```
make
```

### train model
```
python train.py
```
```
python3 train.py --help
Using TensorFlow backend.
usage: train.py [-h] [--data DATA] [--glove GLOVE]
                [--num_hidden_units NUM_HIDDEN_UNITS]
                [--attention_units ATTENTION_UNITS] [--epoches EPOCHES]
                [--batch_size BATCH_SIZE] [--site_path SITE_PATH]

Train simple NER model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to directory with data files
  --glove GLOVE         path to directory with glove data
  --num_hidden_units NUM_HIDDEN_UNITS
                        num GRU units
  --attention_units ATTENTION_UNITS
                        num hidden states in simple attention
  --epoches EPOCHES     num epoches for traning
  --batch_size BATCH_SIZE
                        batch size for traning
  --site_path SITE_PATH
                        path to your site for storing model

```

### prepare node.js development env
```
cd nodejs_brows && npm install
```

### run local server
```
node server.js
```
