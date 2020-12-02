# copy right @ Feng, Polson, and Xu "Deep Learning in Characteristics-Sorted Factor Models" (2019)

import numpy as np
import tensorflow as tf


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
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([out_size]))
    wxb = tf.tensordot(tf.nn.dropout(inputs, keep),
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
    :return: constructed deep factors and deep characteristics
    '''
    print(tf.__version__)
    # split data to training sample and test sample
    z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p = \
        data_split(data['characteristics'], data['stock_return'], data['factor'], data['target_return'],
                   para['train_ratio'], para['split'])

    n_train = r_train.shape[0]

    # the last element of layer_size is the number of deep factors
    fsort_number = layer_size[-1]

    # create deep learning graph
    z = tf.placeholder(tf.float32, [None, n, p])
    r = tf.placeholder(tf.float32, [None, None])
    target = tf.placeholder(tf.float32, [None, port_n])
    m = tf.placeholder(tf.float32, [None, ff_n])

    # create graph for sorting
    with tf.compat.v1.name_scope('sorting_network'):
        # subtract gamma*benchmark from excess stock returns and keep the residual for deep factor construction

        # add 1st network (prior to sorting)
        layer_number_tanh = layer_size.__len__()
        layer_size = np.insert(layer_size, 0, p)
        layers_1 = [z]
        weights_l1 = tf.constant(0.0)
        for i in range(layer_number_tanh):
            new_layer, weights, wxb = add_layer_1(
                layers_1[i], layer_size[i], layer_size[i + 1], para['activation'])
            layers_1.append(new_layer)
            if i < layer_number_tanh -1:
                weights_l1 += (tf.reduce_sum(tf.abs(weights)) - tf.reduce_sum(tf.abs(tf.linalg.diag_part(weights))))

        # softmax for factorweight
        mean, var = tf.nn.moments(layers_1[-1],axes=1,keep_dims=True)
        normalized_char = (layers_1[-1] - mean)/(tf.sqrt(var)+0.00001)
        transformed_char_a = -50*tf.exp(-5*normalized_char)
        transformed_char_b = -50*tf.exp(5*normalized_char)
        w_tilde = tf.transpose(a=tf.nn.softmax(transformed_char_a, axis=1) - tf.nn.softmax(transformed_char_b, axis=1), perm=[0,2,1])

        # construct factors
        nobs = tf.shape(r)[0]
        r_tensor = tf.reshape(r, [nobs, n, 1])
        f_tensor = tf.matmul(w_tilde, r_tensor)
        f = tf.reshape(f_tensor, [nobs, fsort_number])

        # forecast return and alpha
        beta = tf.Variable(tf.random_normal([layer_size[-1], port_n]))
        gamma = tf.Variable(tf.random_normal([ff_n, port_n]))
        target_hat = tf.matmul(f, beta) + tf.matmul(m, gamma)
        alpha  = tf.reduce_mean(target - target_hat,axis=0) 

        # define loss and training parameters
        zero = tf.zeros([port_n,])
        loss1 = tf.compat.v1.losses.mean_squared_error(target, target_hat)
        loss2 = tf.compat.v1.losses.mean_squared_error(zero, alpha)
        loss = loss1 + para['Lambda']*loss2 + para['Lambda2']*weights_l1
        train = para['train_algo'](para['learning_rate']).minimize(loss)

    batch_number = int(t_train / para['batch_size'])
    loss_path = []
    early_stopping = 10
    thresh = 0.000005
    stop_flag = 0

    # SGD training
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # train sorting network
        for i in range(para['epoch']):
            batch = get_batch(t_train, batch_number)

            for idx in range(batch_number):
                sess.run(train, feed_dict={
                    z: z_train[batch[idx]], r: r_train[batch[idx]], target: target_train[batch[idx]],
                    m: m_train[batch[idx]]})

            current_loss, current_loss1, current_loss2, current_loss3 = sess.run(
                [loss, loss1, loss2, weights_l1], feed_dict={z: z_train, r: r_train, target: target_train, m: m_train})
            print("current epoch:", i,
                  "; current loss1:", current_loss1, "; current_loss2:", current_loss2,
                  "; current loss3:", current_loss3)
            loss_path.append(current_loss)

            if np.isnan(current_loss):
                break

            if i > 0:
                if loss_path[i-1] - loss_path[i] < thresh:
                    stop_flag += 1
                else:
                    stop_flag = 0
                if stop_flag >= early_stopping:
                    print('Early stopping at epoch:', i)
                    break


        # save constructed sort factors
        factor_in = sess.run(
            f, feed_dict={z: z_train, r: r_train, target: target_train, m: m_train})
        factor_out = sess.run(
            f, feed_dict={z: z_test, r: r_test, target: target_test, m: m_test})

        # characteristics

        deep_char = sess.run(layers_1[-1], feed_dict={z: np.concatenate((z_train,z_test),axis=0)})
        

    factor = np.concatenate((factor_in,factor_out),axis=0)
    nt, nnn, pp = deep_char.shape
    deep_char = deep_char.reshape(nt*nnn, pp)
    return factor, deep_char
