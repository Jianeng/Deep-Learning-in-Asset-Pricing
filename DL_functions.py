# copy right @ Feng, Polson, and Xu "Deep Learning in Characteristics-Sorted Factor Models" (2019)

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def data_split(z_data, r_data, m_data, target_data, ratio, split):
    '''
    split data
    :param z_data: characteristics
    :param r_data: stock return
    :param m_data: benchmark model
    :param target_data: target portfolio
    :param ratio: train/test ratio for split
    :param split: if "future", split data into past/future periods using "ratio",
                  if integer "t", select test data every t periods
    :return: train and test data
    '''

    ff_n = m_data.shape[1]  # factor number
    port_n = target_data.shape[1]  # target (portfolio) number
    [t, n] = r_data.shape  # time length and stock number
    p = int(z_data.shape[1] / n)  # characteristics number
    z_data = z_data.reshape((t, p, n)).transpose((0, 2, 1))   # dim: (t,n,p)

    # train sample and test sample
    if split == 'future':
        test_idx = np.arange(int(t * ratio), t)
    else:
        test_idx = np.arange(0, t, split)

    train_idx = np.setdiff1d(np.arange(t), test_idx)
    t_train = np.alen(train_idx)
    z_train = z_data[train_idx]
    z_test = z_data[test_idx]
    r_train = r_data[train_idx]
    r_test = r_data[test_idx]
    target_train = target_data[train_idx]
    target_test = target_data[test_idx]
    m_train = m_data[train_idx]
    m_test = m_data[test_idx]

    return z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p


def add_layer_1(inputs, in_size, out_size, activation, keep=0.5):
    '''
    add a neural layer on top of "inputs"
    :param inputs: lower layer
    :param in_size: size of lower layer (number of characteristics)
    :param out_size: size of new layer
    :param activation: activation function
    :param keep: dropout
    :return: new layer
    '''
    weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.random.normal([out_size]))
    wxb = tf.tensordot(tf.nn.dropout(inputs, 1 - (keep)),
                       weights, axes=[[2], [0]]) + biases
    outputs = activation(wxb)
    return outputs


def get_batch(total, batch_number):
    '''
    create batches
    :param total: number of data points
    :param batch_number: number of batches
    :return: batches
    '''
    sample = np.arange(total)
    np.random.shuffle(sample)
    batch = np.array_split(sample, batch_number)
    return batch


def dl_alpha(data, layer_size, para):
    '''
    construct deep factors
    :param data: a dict of input data
    :param layer_size: a list of neural layer sizes (from bottom to top)
    :param para: training and tuning parameters
    :param value_index: index of market equity in the characteristic dataset (which column)
    :return: constructed deep factors and loss paths
    '''

    # split data to training sample and test sample
    z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p = \
        data_split(data['characteristics'], data['stock_return'], data['factor'], data['target_return'],
                   para['train_ratio'], para['split'])

    # the last element of layer_size is the number of deep factors
    fsort_number = layer_size[-1]

    # create deep learning graph
    z = tf.compat.v1.placeholder(tf.float32, [None, n, p])
    r = tf.compat.v1.placeholder(tf.float32, [None, None])
    target = tf.compat.v1.placeholder(tf.float32, [None, port_n])
    m = tf.compat.v1.placeholder(tf.float32, [None, ff_n])

    # create graph for sorting
    with tf.compat.v1.name_scope('sorting_network'):
        # add 1st network (prior to sorting)
        layer_number_tanh = layer_size.__len__()
        layer_size = np.insert(layer_size, 0, p)
        layers_1 = [z]
        for i in range(layer_number_tanh):
            new_layer = add_layer_1(
                layers_1[i], layer_size[i], layer_size[i + 1], para['activation'])
            layers_1.append(new_layer)

        # softmax rank weight
        normalized_char = tf.keras.layers.BatchNormalization(axis=1)(layers_1[-1], training=True)
        transformed_char_a = -50*tf.exp(-5*normalized_char)
        transformed_char_b = -50*tf.exp(5*normalized_char)
        w_tilde = tf.transpose(a=tf.nn.softmax(transformed_char_a, axis=1) - tf.nn.softmax(transformed_char_b, axis=1), perm=[0,2,1])

        # construct factors
        nobs = tf.shape(input=r)[0]
        r_tensor = tf.reshape(r, [nobs, n, 1])
        f_tensor = tf.matmul(w_tilde, r_tensor)
        f = tf.reshape(f_tensor, [nobs, fsort_number])

        # forecast  and alpha
        beta = tf.Variable(tf.random.normal([layer_size[-1], port_n]))
        gamma = tf.Variable(tf.random.normal([ff_n, port_n]))
        target_hat = tf.matmul(f, beta) + tf.matmul(m, gamma)
        alpha  = tf.reduce_mean(input_tensor=target - target_hat,axis=0) 

        # define loss and training parameters
        zero = tf.zeros([port_n,])
        loss = tf.compat.v1.losses.mean_squared_error(target, target_hat) + para['Lambda']*tf.compat.v1.losses.mean_squared_error(zero, alpha)
        train = para['train_algo'](para['learning_rate']).minimize(loss)

    batch_number = int(t_train / para['batch_size'])
    loss_path = []

    # SGD training
    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        # train sorting network
        for i in range(para['epoch']):
            batch = get_batch(t_train, batch_number)

            for idx in range(batch_number):
                sess.run(train, feed_dict={
                    z: z_train[batch[idx]], r: r_train[batch[idx]], target: target_train[batch[idx]],
                    m: m_train[batch[idx]]})

            current_loss = sess.run(
                loss, feed_dict={z: z_train, r: r_train, target: target_train, m: m_train})
            print("current epoch:", i,
                  "; current loss:", current_loss)
            loss_path.append(current_loss)

        # save constructed sort factors
        factor = sess.run(
            f, feed_dict={z: z_train, r: r_train, target: target_train, m: m_train})
        factor_oos = sess.run(
            f, feed_dict={z: z_test, r: r_test, target: target_test, m: m_test})

        return factor, factor_oos, loss_path
