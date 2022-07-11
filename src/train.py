from __future__ import division
from __future__ import print_function

import random
from utils import *
from metrics import *
from models import GCN_Align, PU_classifier
import os
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('lang', 'zh_en', 'Dataset string.')
flags.DEFINE_float('learning_rate', 25, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('k', 5, 'Number of negative samples for each positive seed.')
flags.DEFINE_integer('se_dim', 200, 'Dimension for SE.')
flags.DEFINE_integer('seed', 3, 'Proportion of seeds, 3 means 30%')

pu = True

# Load data
adj, e, train, test, pretrain_truth, train_noise, unlabeled = load_data(FLAGS.lang)

label = np.concatenate([np.ones(len(pretrain_truth)), np.zeros(len(unlabeled))])

batch_size = len(train)*2*FLAGS.k

train_dataset = TrainSet(train, FLAGS.k)
train_dataset.init_subsampling(pretrain_truth)
train_dataset.get_pretrain(pretrain_truth)
train_dataset.get_noise_involved(train_noise)

clf_train = np.asarray(random.sample(train_dataset.pretrain, len(train_dataset.pretrain)))
clf_dataset = TrainSet(clf_train, FLAGS.k)
clf_dataset.subsampling_weight = train_dataset.subsampling_weight

# Some preprocessing
support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN_Align
k = FLAGS.k

ph_se = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}

model_se = model_func(ph_se, input_dim=e, output_dim=FLAGS.se_dim, ILL=train_dataset, sparse_inputs=False, featureless=True, logging=True)
PU_c = PU_classifier(n_input=FLAGS.se_dim, n_hidden_1=100, model=model_se, batch_size= batch_size, mode="clf")

# Initialize session
c = tf.ConfigProto(inter_op_parallelism_threads=3, intra_op_parallelism_threads=3)
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

t = len(train)
L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
neg_left = L.reshape((t * k,))
L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
neg2_right = L.reshape((t * k,))

best_hit1 = 0
best_hit5 = 0
best_mrr = 0
best_epoch = 0

iter_pacing = 0

# Train model
for epoch in range(FLAGS.epochs):

    if epoch % 10 == 0:
        neg2_left = np.random.choice(e, t * k)
        neg_right = np.random.choice(e, t * k)

    feed_dict_se = construct_feed_dict(1.0, support, ph_se)
    feed_dict_se.update({ph_se['dropout']: FLAGS.dropout})
    feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left,
                         'neg2_right:0': neg2_right, 'subsampling:0': train_dataset.get_subsampling()})

    outs_se = sess.run([model_se.opt_op, model_se.loss], feed_dict=feed_dict_se)

    cost_val.append((outs_se[1]))

    if epoch % 1000 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "SE_train_loss=", "{:.5f}".format(outs_se[1]))

    ## Test every 1000 epoch.
    # if epoch % 1000 == 0 and epoch != 0:
    #     feed_dict_se_test = construct_feed_dict(1.0, support, ph_se)
    #     vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se_test)
    #     hit1, hit5, mrr = get_hits(vec_se, test)

    if pu == True and epoch % 2000 == 0 and epoch != 0:

        vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se)

        for sub_epoch in range(5000):
            _, _ = sess.run([PU_c.opt_prior, PU_c.loss_prior], feed_dict={PU_c.model_output: vec_se, PU_c.input_sample_unlabeled: unlabeled,
                                                             PU_c.input_sample_truth: pretrain_truth, PU_c.label: label})
        for sub_epoch in range(2000):
            _, _ = sess.run([PU_c.opt_pu, PU_c.loss_pu], feed_dict={PU_c.model_output: vec_se, PU_c.input_sample_unlabeled: unlabeled,
                                                             PU_c.input_sample_truth: pretrain_truth, PU_c.label: label})

        train_dataset.init_subsampling(pretrain_truth)
        PU_c.find_topK_pair(vec_se, PU_c, train_dataset, clf_dataset, sess, iter_pacing)
        iter_pacing += 1

print("Optimization Finished!")
feed_dict_se = construct_feed_dict(1.0, support, ph_se)
vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se)
get_hits(vec_se, test)

