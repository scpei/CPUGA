from layers import *
from metrics import *
from inits import *
from utils import *
import heapq
import tensorflow_probability as tfp

flags = tf.app.flags
FLAGS = flags.FLAGS
tfd = tfp.distributions

def pacing_func(iter, a=15, b=8, c=20):
    return a + b*np.exp(-iter/c)

class TopKHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_Align(Model):
    def __init__(self, placeholders, input_dim, output_dim, ILL, sparse_inputs=False, featureless=True, **kwargs):
        super(GCN_Align, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ILL = ILL
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        self.loss += align_loss(self.outputs, self.ILL, FLAGS.gamma, FLAGS.k)

    def _accuracy(self):
        pass

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            transform=False,
                                            init=trunc_normal,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            transform=False,
                                            logging=self.logging))


class PU_classifier(object):
    def __init__(self, n_input, n_hidden_1, model, batch_size, mode):

        self.model = model
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.batch_size = batch_size
        self.opt = None
        self.cost = None
        self.mode = mode
        self.build()

    def forward(self, in_sample):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(in_sample, self.weights['h1']), self.biases['b1']))
        pred = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        return pred

    def build(self):

        def mode_n(ass_clust):
            unique, _, count = tf.unique_with_counts(ass_clust)
            return tf.scatter_nd(tf.reshape(unique, [-1, 1]), count, shape=tf.constant([2]))

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([1]))
        }

        self.Nc = 2
        self.Nd = FLAGS.se_dim

        self.locs = tf.Variable(tf.random_normal([self.Nc, self.Nd]))
        self.scales = tf.Variable(tf.pow(tf.random_gamma([self.Nc, self.Nd], 1, 1), -0.5))

        self.alpha = tf.Variable(tf.random_uniform([self.Nc, self.Nd], 1., 2.))
        self.beta = tf.Variable(tf.random_uniform([self.Nc, self.Nd], 1., 2.))

        self.counts = tf.Variable(2 * tf.ones((self.Nc,)))

        self.mu_prior = tfd.Normal(tf.zeros((self.Nc, self.Nd)), tf.ones((self.Nc, self.Nd)))

        self.sigma_prior = tfd.Gamma(tf.ones((self.Nc, self.Nd)), tf.ones((self.Nc, self.Nd)))

        self.theta_prior = tfd.Dirichlet(2 * tf.ones((self.Nc,)))

        self.track_c = tf.Variable([0, 0], trainable=False)

        self.model_output = tf.placeholder(shape=[None, FLAGS.se_dim], dtype=tf.float32)
        self.input_sample_truth = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.input_sample_unlabeled = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.label = tf.placeholder(shape=[None, ], dtype=tf.int32)

        self.h_t = tf.nn.embedding_lookup(self.model_output, self.input_sample_truth[:, 0])
        self.t_t = tf.nn.embedding_lookup(self.model_output, self.input_sample_truth[:, 1])

        self.h_u = tf.nn.embedding_lookup(self.model_output, self.input_sample_unlabeled[:, 0])
        self.t_u = tf.nn.embedding_lookup(self.model_output, self.input_sample_unlabeled[:, 1])

        ht_t = tf.abs(self.h_t - self.t_t)
        ht_u = tf.abs(self.h_u - self.t_u)

        ht_ = tf.concat([ht_t, ht_u], 0)

        self.mu, self.sig, self.log_likelihoods, self.kl_sum, self.theta, self.prob_ = self.GMM(ht_, False)
        self.test = tf.reduce_mean(self.log_likelihoods)
        self.loss_prior = (self.kl_sum - tf.reduce_sum(self.log_likelihoods)) / 151300

        self.ht_post = tfd.Normal(loc=tf.reshape(tf.ones_like(ht_[:, 0], dtype=tf.float32), [-1, 1, 1]) * self.mu,
                            scale=tf.reshape(tf.ones_like(ht_[:, 0], dtype=tf.float32), [-1, 1, 1]) * tf.sqrt(self.sig))

        self.ht_broadcasted = tf.tile(tf.reshape(ht_, [-1, 1, FLAGS.se_dim]), [1, 2, 1])
        self.log_liks = self.ht_post.log_prob(self.ht_broadcasted)
        self.log_liks = tf.reduce_mean(self.log_liks, 2)
        self.clusters = tf.argmax(self.log_liks, 1, output_type=tf.int32)

        self.label = tf.less_equal(self.label, 1)
        has_pos = tf.reduce_any(self.label)
        self.add_num = tf.cond(has_pos, lambda: mode_n(self.clusters[self.label]), lambda: tf.constant([0, 0]))
        self.r_opp = self.track_c.assign_add(self.add_num)
        self.ass_clust = tf.argmax(self.track_c)
        self.prior = tf.gather(tf.reshape(tf.reduce_mean(self.theta, 0), [-1]), self.ass_clust)

        self.score_t = self.forward(ht_t)
        self.score_u = self.forward(ht_u)

        self.sig_score_t = tf.nn.sigmoid(-self.score_t)
        self.sig_score_t_negative = tf.nn.sigmoid(self.score_t)
        self.sig_score_u = tf.nn.sigmoid(self.score_u)
        self.sig_untrusted = self.sig_score_u

        self.sig_noise = tf.nn.sigmoid(self.score_t)
        self.logistic_noise = tf.log(1 + tf.exp(self.score_t))

        self.loss_true = tf.reduce_mean(self.sig_score_t)
        self.loss_untrusted = tf.reduce_mean(self.sig_untrusted)
        self.loss_true_neg = tf.reduce_mean(self.sig_score_t_negative)

        self.loss_pu = self.prior * self.loss_true + tf.maximum(0.0, self.loss_untrusted - self.prior * self.loss_true_neg)

        self.opt_pu = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.loss_pu, var_list=[self.weights, self.biases])

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.0005, global_step,
                                                   500, 0.8, staircase=False)

        self.opt_prior = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_prior,
                                                var_list=[self.locs, self.scales, self.alpha, self.beta, self.counts], global_step=global_step)

    def GMM(self, x, sampling=True):

        mu = tfd.Normal(self.locs, self.scales)
        sigma = tfd.Gamma(self.alpha, self.beta)
        theta = tfd.Dirichlet(self.counts)

        if sampling:
            Nb = tf.shape(x)[0]
            mu_sample = mu.sample(Nb)
            sigma_sample = tf.pow(sigma.sample(Nb), -0.5)
            theta_sample = theta.sample(Nb)
        else:
            mu_sample = tf.reshape(mu.mean(), (1, self.Nc, self.Nd))
            sigma_sample = tf.pow(tf.reshape(sigma.mean(), (1, self.Nc, self.Nd)), -0.5)
            theta_sample = tf.reshape(theta.mean(), (1, self.Nc))

        density = tfd.Mixture(
            cat=tfd.Categorical(probs=theta_sample),
            components=[
                tfd.MultivariateNormalDiag(loc=mu_sample[:, i, :],
                                           scale_diag=sigma_sample[:, i, :])
                for i in range(self.Nc)])

        log_likelihoods = density.log_prob(x)
        prob_ = density.prob(x)

        mu_div = tf.reduce_sum(tfd.kl_divergence(mu, self.mu_prior))
        sigma_div = tf.reduce_sum(tfd.kl_divergence(sigma, self.sigma_prior))
        theta_div = tf.reduce_sum(tfd.kl_divergence(theta, self.theta_prior))

        kl_sum = mu_div + sigma_div + theta_div

        return mu_sample, sigma_sample, log_likelihoods, kl_sum, theta_sample, prob_

    def find_topK_pair(self, vec_se, classifier, train_dataset, clf_dataset, sess, iter):

        new_set_discrim = []
        iter_pacing = pacing_func(iter=iter)

        i = 0
        weight_list = []
        while i < len(train_dataset.noise):
            j = min(i + 2000, len(train_dataset.noise))
            sample = train_dataset.noise[i:j]
            orig_weight = sess.run(classifier.logistic_noise, feed_dict={classifier.model_output: vec_se,
                                                                           classifier.input_sample_truth: sample})
            weight_list.extend(orig_weight)

            for x, pair in enumerate(train_dataset.noise[i:j]):
                if orig_weight[x][0] > iter_pacing:
                    train_dataset.subsampling_weight[pair] = 1
                    new_set_discrim.append(pair)
            i = j


        all_weights = np.array(weight_list)
        all_weights = np.reshape(all_weights, (all_weights.shape[0], ))

        new_weight_list = []
        for i in range(len(all_weights)):
            if all_weights[i] > iter_pacing:
                new_weight_list.append(1)
            else:
                new_weight_list.append(0)

        clf_dataset.train = train_dataset.pretrain + new_set_discrim
        clf_dataset.len = len(clf_dataset.train)






