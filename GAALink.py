from __future__ import division

import pickle as pkl
import numpy as np
import pandas as pd
import sys
import time
import os
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import scipy
import scipy.stats

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import backend as K
import tensorflow.keras

from graph_attention_layer import GraphAttention
from utils import load_data, Evaluation
from sklearn.metrics import roc_auc_score,average_precision_score

import argparse

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_str', default='Biase',type=str, help='name of dataset')
parser.add_argument('--n_clusters', default = 1, type=int, help='expected number of clusters')
parser.add_argument('--subtype_path', default=None, type=str, help='path of true labels for evaluation of ARI and NMI')
parser.add_argument('--k', default=None, type=int, help='number of neighbors to construct the gene graph')
parser.add_argument('--PCA_dim', default=250, type=int, help='dimensionality of input feature matrix that transformed by PCA') #512
parser.add_argument('--F1', default=128, type=int, help='number of neurons in the 1-st layer of encoder') #128
parser.add_argument('--F2', default=64, type=int, help='number of neurons in the 2-nd layer of encoder') #64
parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')#4
parser.add_argument('--dropout_rate', default=0.45, type=float, help='dropout rate of neurons in autoencoder') #0.4
parser.add_argument('--l2_reg', default=0.4, type=float, help='coefficient for L2 regularizition') #0.2
parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for training') #5e-4
parser.add_argument('--pre_lr', default=2e-4, type=float, help='learning rate for pre-training') #2e-4
parser.add_argument('--pre_epochs', default=200, type=int, help='number of epochs for pre-training')
parser.add_argument('--epochs', default=300, type=int, help='number of epochs for pre-training')
parser.add_argument('--c1', default=1, type=float, help='weight of reconstruction loss')#
parser.add_argument('--c2', default=1, type=float, help='weight of clustering loss')

args = parser.parse_args()


if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('result/'):
    os.makedirs('result/')


dataset_str = 'Benchmark Dataset/STRING Dataset/hESC/TFs+500'
n_clusters = args.n_clusters
if args.k == 1:
    dropout_rate = 0. # To avoid absurd results\

else:
    dropout_rate = args.dropout_rate

# Paths
data_path = 'Dataset/'+dataset_str+'/BL--ExpressionData.csv'
GAT_autoencoder_path = 'logs/GATae_'+dataset_str+'.h5'

intermediate_path = 'logs/model_'+dataset_str+'_'



# Read data
start_time = time.time()
A, X, cells, genes = load_data(data_path, dataset_str,
                               args.PCA_dim, n_clusters, args.k)
end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-process: run time is %.2f '%run_time, 'minutes')


# Parameters
N = X.shape[0]                  # Number of nodes in the graph
F = X.shape[1]                  # Original feature dimension

# Loss functions
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true))

def DAEGC_class_loss_1(y_pred):
    return K.mean(K.exp(-1 * A * K.sigmoid(K.dot(y_pred, K.transpose(y_pred)))))

def maie_class_loss(y_true, y_pred):
    loss_E = mae(y_true, y_pred)
    return loss_E


# Model definition
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout1, A_in])

dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(args.F2,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout2, A_in])

dropout3 = Dropout(dropout_rate)(graph_attention_2)
graph_attention_3 = GraphAttention(args.F1,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout3, A_in])

dropout4 = Dropout(dropout_rate)(graph_attention_3)
graph_attention_4 = GraphAttention(F,
                                   attn_heads=args.n_attn_heads,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l1(args.l2_reg),
                                   attn_kernel_regularizer=l1(args.l2_reg))([dropout4, A_in])

# Build GAT autoencoder model
GAT_autoencoder = Model(inputs=[X_in, A_in], outputs=graph_attention_4)
optimizer = Adam(lr=args.pre_lr)
GAT_autoencoder.compile(optimizer=optimizer,
              loss=maie_class_loss)
#GAT_autoencoder.summary()

# Callbacks
es_callback = EarlyStopping(monitor='loss', min_delta=0.1, patience=50)
tb_callback = TensorBoard(batch_size=N)
mc_callback = ModelCheckpoint(GAT_autoencoder_path,
                              monitor='loss',
                              save_best_only=True,
                              save_weights_only=True)

# Train GAT_autoencoder model
start_time = time.time()
GAT_autoencoder.fit([X, A],X,epochs=args.pre_epochs,batch_size=N,
                    verbose=0,shuffle=False)
end_time = time.time()
run_time = (end_time - start_time) / 60
print('Pre-train: run time is %.2f '%run_time, 'minutes')


# Construct a model for hidden layer
hidden_model = Model(inputs=GAT_autoencoder.input,outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)


def pred_loss(y_true, y_pred):
    return y_true-y_pred

# Construct total model
model = Model(inputs=[X_in, A_in],
              outputs=[graph_attention_4,
                       graph_attention_2])


optimizer = Adam(lr=args.learning_rate)
model.compile(optimizer=optimizer,
              loss=[maie_class_loss,  pred_loss],
                    loss_weights=[args.c1,  0])


# Train model
start_time = time.time()

tol = 1e-5
loss = 0

sil_logs = []
update_interval = 2
res_ite = 0
final_pred = None
max_sil = 0


for ite in range(args.epochs + 1):
    print(ite)
    if ite % update_interval == 0:
        res_ite = ite

        _,  hid = model.predict([X, A], batch_size=N, verbose=0)
    loss = model.train_on_batch(x=[X, A], y=[X,  hid])
    print('loss:',loss)




end_time = time.time()
run_time = (end_time - start_time) / 60
print('Train: run time is %.2f '%run_time, 'minutes')





# Get hidden representation
hidden_model = Model(inputs=model.input, outputs=graph_attention_2)
hidden = hidden_model.predict([X, A], batch_size=N)
hidden = hidden.astype(float)
Predict_value = np.dot(hidden,hidden.T)