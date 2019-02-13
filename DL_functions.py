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


def value_sort(weight, value, n_factor, n_firm, prop=0.2):
    '''
    creating long-short portfolio weight via sorting
    :param weight: input characteristics
    :param value: market equity of stocks
    :param n_factor: number of factors
    :param n_firm: number of stocks
    :param prop: top/bottom proportion of stock universe to be selected
    :return: long-short weight
    '''

    a1, _ = tf.nn.top_k(weight, int(n_firm * prop))
    top_breakpoint = tf.reduce_min(a1, axis=2, keepdims=True)
    a2, _ = tf.nn.top_k(weight, int(n_firm * (1 - prop)))
    bottom_breakpoint = tf.reduce_min(a2, axis=2, keepdims=True)

    top = 0.5 * tf.nn.tanh(10000 * (weight - top_breakpoint)) + 0.5
    bottom = 0.5 - 0.5 * tf.nn.tanh(10000 * (weight - bottom_breakpoint))

    value_diag = tf.tile(tf.expand_dims(value, axis=1), [1, n_factor, 1])

    top_prep = top * value_diag
    top = top_prep / tf.maximum(tf.reduce_sum(top_prep, 2, keepdims=True), 1)
    bottom_prep = bottom * value_diag
    bottom = bottom_prep / \
        tf.maximum(tf.reduce_sum(bottom_prep, 2, keepdims=True), 1)
    final_weight = top - bottom

    return final_weight


def dl_alpha(data, layer_size, para, value_index, ens=1):
    '''
    construct deep factors
    :param data: a dict of input data
    :param layer_size: a list of neural layer sizes (from bottom to top)
    :param para: training and tuning parameters
    :param value_index: index of market equity in the characteristic dataset (which column)
    :param ens: size of ensembles
    :return: constructed deep factors and loss/alpha_loss paths
    '''

    # split data to training sample and test sample
    z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p = \
        data_split(data['characteristics'], data['stock_return'], data['factor'], data['target_return'],
                   para['train_ratio'], para['split'])

    # the last element of layer_size is the number of deep factors
    fsort_number = layer_size[-1]

    # create deep learning graph
    z = tf.placeholder(tf.float32, [None, n, p])
    r = tf.placeholder(tf.float32, [None, None])
    target_org = tf.placeholder(tf.float32, [None, port_n])
    m = tf.placeholder(tf.float32, [None, ff_n])
    value = z[:, :, value_index]

    # create graph for sorting
    with tf.name_scope('sorting_network'):
        # subtract gamma*benchmark from excess stock returns and keep the residual for deep factor construction
        gamma = tf.Variable(tf.random_normal([ff_n, port_n, ens]))
        target_gamma_hat = tf.tensordot(m, gamma, [[1], [0]])
        target_gamma_ens = tf.tile(
            tf.expand_dims(target_org, axis=2), [1, 1, ens])
        target = tf.reduce_mean(target_gamma_ens - target_gamma_hat, axis=2)

        # add 1st network (prior to sorting)
        layer_number_tanh = layer_size.__len__()
        layer_size = np.insert(layer_size, 0, p)
        layers_1 = [z]
        for i in range(layer_number_tanh):
            new_layer = add_layer_1(
                layers_1[i], layer_size[i], layer_size[i + 1], para['activation'])
            layers_1.append(new_layer)

        # sort deep characteristics
        w_trans = tf.transpose(layers_1[-1], [0, 2, 1])

        w_tilde = value_sort(w_trans, value, 1, n)

        # construct factors
        nobs = tf.shape(r)[0]
        r_tensor = tf.reshape(r, [nobs, n, 1])
        f_tensor = tf.matmul(w_tilde, r_tensor)
        f = tf.reshape(f_tensor, [nobs, fsort_number])

        # forecast return and alpha
        beta_mid = tf.Variable(tf.random_normal(
            [layer_size[-1], port_n, ens]))
        target_mid_hat = tf.tensordot(
            f, beta_mid, [[1], [0]])
        target_ens = tf.tile(tf.expand_dims(
            target, axis=2), [1, 1, ens])
        alpha_mid = tf.reduce_mean(target_ens - target_mid_hat, axis=1)
        alpha_mse = tf.reduce_mean(tf.square(alpha_mid))

        # define loss and training parameters
        loss_mid = tf.losses.mean_squared_error(
            target_ens, target_mid_hat)
        train_mid = para['train_algo'](
            para['learning_rate']).minimize(loss_mid)

    batch_number = int(t_train / para['batch_size'])
    alpha_path = []
    loss_path = []

    # SGD training
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # train sorting network
        for i in range(para['epoch']):
            batch = get_batch(t_train, batch_number)

            for idx in range(batch_number):
                sess.run(train_mid, feed_dict={
                    z: z_train[batch[idx]], r: r_train[batch[idx]], target_org: target_train[batch[idx]],
                    m: m_train[batch[idx]]})

            current_loss, current_alpha = sess.run(
                [loss_mid, alpha_mse], feed_dict={z: z_train, r: r_train, target_org: target_train, m: m_train})
            print("current epoch:", i,
                  "; current loss:", current_loss)
            alpha_path.append(current_alpha)
            loss_path.append(current_loss)

        # save constructed sort factors
        factor = sess.run(
            f, feed_dict={z: z_train, r: r_train, target_org: target_train, m: m_train})
        factor_oos = sess.run(
            f, feed_dict={z: z_test, r: r_test, target_org: target_test, m: m_test})

        return factor, factor_oos, alpha_path, loss_path


def dl_conditional(fg_in, fg_out, return_in, return_out, nrelu, para):
    '''
    conditional factor model via ReLU network
    :param fg_in: in-sample deep + benchmark factors
    :param fg_out: out-of-sample deep + benchmark factors
    :param return_in: in-sample target returns
    :param return_out: out-of-sample target returns
    :param nrelu: number of ReLU units
    :param para: training and tuning parameters
    :return: in-sample and out-of-sample predictions
    '''

    t, p = fg_in.shape
    k = return_in.shape[1]

    X = tf.placeholder(tf.float32, [None, p])
    Y = tf.placeholder(tf.float32, [None, k])

    w1 = tf.Variable(tf.random_normal([p, nrelu]))
    w2 = tf.Variable(tf.random_normal([nrelu, k]))

    Z = tf.nn.relu(tf.matmul(X, w1))
    Yhat = tf.matmul(Z, w2)
    loss = tf.losses.mean_squared_error(Y, Yhat)
    train = para['train_algo'](para['learning_rate']).minimize(loss)

    batch_number = int(t / para['batch_size'])
    loss_path = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # train ReLU network
        for i in range(epoch):
            batch = get_batch(t, batch_number)

            for idx in range(batch_number):
                sess.run(train, feed_dict={
                         X: fg_in[batch[idx]], Y: return_in[batch[idx]]})

            current_loss = sess.run(loss, feed_dict={X: fg_in, Y: return_in})
            print("current epoch:", i,
                  "; current loss:", current_loss)
            loss_path.append(current_loss)

        insample_prediction = sess.run(
            Yhat, feed_dict={X: fg_in, Y: return_in})
        oos_prediction = sess.run(Yhat, feed_dict={X: fg_out, Y: return_out})

    return insample_prediction, oos_prediction, loss_path
