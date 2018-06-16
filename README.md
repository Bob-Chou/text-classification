# Chinese Text Binary Classification (News)
## Setup
Extract features from raw data.
### Word-wise
Cut the raw data 'xx/data.txt' into word-wise sequence (split the word). The default name of output word-wise sequence is 'xx/cut.txt'. 
```shell
python3 setup/cut_word.py --input datasets/ --output datasets/ --stopword stopword.txt
```
Derive the vocabulary.
```shell
if [[ ! -d feature/word_wise ]]; then
    mkdir -p feature/word_wise/
fi
python3 setup/word_to_id.py --input ./datasets/train --output ./feature/word_wise/
```
Derive the TF-IDF features.
```shell
if [[ ! -d feature/word_wise/tf-idf/ ]]; then
    mkdir -p feature/word_wise/tf-idf/
fi
python3 setup/tf_idf.py --input datasets/ --output feature/word_wise/tf_idf --vocabulary feature/word_wise/word2id.pkl
```
Derive word2vec.
```shell
if [[ ! -d feature/word_wise/word2vec/ ]]; then
    mkdir -p feature/word_wise/word2vec/
fi
python3 setup/word2vec.py --input datasets/train/ --output feature/word_wise/word2vec/word2vec.embedding --vocabulary feature/word_wise/word2id.pkl --dim 1024 --epoch 10 --window 10 --min_count 100 --workers 8
```
### Char-wise
Derive the vocabulary.
```shell
if [[ ! -d feature/char_wise ]]; then
    mkdir -p feature/char_wise/
fi
python3 setup/word_to_id.py --input ./datasets/train --output ./feature/char_wise/ --char
```
## Model
### Naive-Bayes (word-wise)
Train naive-bayes (Gaussian) based on TF-IDF features.
```shell
if [[ ! -d model/naive_bayes/checkpoint ]]; then
    mkdir -p model/naive_bayes/checkpoint
fi
python3 model/naive_bayes/naive_bayes.py --input feature/word_wise/tf_idf/train/ --checkpoint model/naive_bayes/checkpoint/nb_tf_idf.clf --sparse
```
Use naive-bayes (Gaussian) for predicting
```shell
if [[ ! -d prediction ]]; then
    mkdir prediction
fi
python3 model/naive_bayes/naive_bayes.py --input feature/word_wise/tf_idf/test/ --test prediction/naive_bayes_pred.txt --checkpoint model/naive_bayes/checkpoint/nb_tf_idf.clf --sparse
```
### MLP-TFIDF (word-wise)
Train MLP based on TF-IDF features.
```shell
if [[ ! -d model/mlp/checkpoint ]]; then
    mkdir -p model/mlp/checkpoint
fi
python3 model/mlp/mlp.py --input feature/word_wise/tf_idf/train --feature 20000 --epoch 5 --batch 16 --hidden_dim 256 --neg_sample 0.1 --checkpoint model/mlp/checkpoint/mlp_tf_idf.clf --save_every 1000 --print_every 200 --cuda --sparse
```
Use MLP for predicting.
```shell
if [[ ! -d prediction ]]; then
    mkdir prediction
fi
python3 model/mlp/mlp.py --input feature/word_wise/tf_idf/test --output prediction/mlp_tf_idf_pred.txt --feature 20000 --batch 16 --hidden_dim 256 --checkpoint model/mlp/checkpoint/mlp_tf_idf.clf --cuda --sparse --test
```
### MLP-Word2Vec (word-wise)
Train MLP based on word2vec features.
```shell
if [[ ! -d model/mlp/checkpoint ]]; then
    mkdir -p model/mlp/checkpoint
fi
python3 model/mlp/mlp.py --input feature/word_wise/word2vec/train --epoch 5 --batch 16 --hidden_dim 256 --checkpoint model/mlp/checkpoint/mlp_word2vec.clf --save_every 1000 --print_every 200 --cuda
```
Use MLP for predicting.
```shell
if [[ ! -d prediction ]]; then
    mkdir prediction
fi
python3 model/mlp/mlp.py --input feature/word_wise/word2vec/test --output prediction/mlp_word2vec_pred.txt --batch 16 --hidden_dim 256 --checkpoint model/mlp/checkpoint/mlp_word2vec.clf --cuda --test
```
### Char CNN
```shell
if [[ ! -d model/char_cnn/checkpoint ]]; then
    mkdir -p model/char_cnn/checkpoint
fi
python3 model/char_cnn/char_cnn.py --input datasets/train/ --vocabulary feature/char_wise/word2id.pkl --kernel 5,7,9 --embed_dim 1024 --word_num 8192 --hidden_dim 1024 --keep_prob 0.5 --learning_rate 1e-3 --checkpoint model/char_cnn/checkpoint/char_cnn.ckpt --save_every 1000 --print_every 100 --sequence_length 1024
```
## Evaluation Metric
ROC and AUC evaluation of a model.
```bash
python3 evaluation/roc.py --input datasets/test --prediction prediction/char_cnn_pred.txt,prediction/mlp_word2vec_pred.txt,prediction/mlp_tf_idf_pred.txt,prediction/naive_bayes_pred.txt --name Char\ CNN,MLP-Word2Vec,MLP-TF-IDF,Naive\ Bayes
```
