import tensorflow as tf
import numpy as np
import scipy

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name+":0")
    except:
        return tf.placeholder(tf.int32, name=name)

def align_loss(outlayer, train_dataset, gamma, k):
    ILL = train_dataset.orgin_train
    subsampling_in = get_placeholder_by_name("subsampling")
    subsampling = tf.cast(subsampling_in, tf.float32)
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = get_placeholder_by_name("neg_left")
    neg_right = get_placeholder_by_name("neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.multiply(tf.reshape(subsampling, [t, 1]), tf.add(C, tf.reshape(D, [t, 1]))))
    neg_left = get_placeholder_by_name("neg2_left")
    neg_right = get_placeholder_by_name("neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.multiply(tf.reshape(subsampling, [t, 1]), tf.add(C, tf.reshape(D, [t, 1]))))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * tf.reduce_sum(subsampling))

def get_hits(vec, test_pair, top_k=(1, 5, 10, 20)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    rankl = []
    rankr = []
    # sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='Euclidean')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        rankl.append(1.0 / (1.0 + rank_index))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        rankr.append(1.0 / (1.0+rank_index))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print(np.mean(rankl))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print(np.mean(rankr))

    return top_lr[0] / len(test_pair) * 100, top_lr[1] / len(test_pair) * 100, np.mean(rankl)

def get_combine_hits(se_vec, ae_vec, beta, test_pair, top_k=(1, 5, 10, 20)):
    vec = np.concatenate([se_vec*beta, ae_vec*(1.0-beta)], axis=1)
    get_hits(vec, test_pair, top_k)
