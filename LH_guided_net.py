#LH_guided_net
import tensorflow as tf
import utils_guided_filter
import unet_more_concat
import CBAM
def low_net(input_image,edge_image):
    input_L=utils_guided_filter.decomposition(input_image,edge_image)#高斯模糊
    #resized_input = tf.image.resize_images(input_L, [50, 50], method=tf.image.ResizeMethod.BICUBIC)#下采样，三次插值
    #with tf.variable_scope("generatorL"):
    enhanced_L=unet_more_concat.unet(input_L)
    return enhanced_L

def high_net(input_image,edge_image):
    input_H=tf.subtract(input_image,utils_guided_filter.decomposition(input_image,edge_image))#高频图像
    with tf.variable_scope("generatorH"):
        Wh1 = weight_variable([3, 3, 3, 64], name="Wh1"); bh1 = bias_variable([64], name="bh1");
        ch1 = tf.nn.relu(conv2d(input_H, Wh1) + bh1)

        # residual 1

        Wh2 = weight_variable([3, 3, 64, 64], name="Wh2"); bh2 = bias_variable([64], name="bh2");
        ch2 = tf.nn.relu(_instance_norm(conv2d(ch1, Wh2) + bh2))

        Wh3 = weight_variable([3, 3, 64, 64], name="Wh3"); bh3 = bias_variable([64], name="b3h");
        ch3 = tf.nn.relu(_instance_norm(conv2d(ch2, Wh3) + bh3)) + ch1

        # residual 2

        Wh4 = weight_variable([3, 3, 64, 64], name="Wh4"); bh4 = bias_variable([64], name="bh4");
        ch4 = tf.nn.relu(_instance_norm(conv2d(ch3, Wh4) + bh4))

        Wh5 = weight_variable([3, 3, 64, 64], name="Wh5"); bh5 = bias_variable([64], name="bh5");
        ch5 = tf.nn.relu(_instance_norm(conv2d(ch4, Wh5) + bh5)) + ch3
        # Final
        Wh6 = weight_variable([3, 3, 64, 3], name="Wh6"); bh6 = bias_variable([3], name="bh6");
        ch6 = conv2d(ch5, Wh6) + bh6
        enhanced_H=ch6
    return enhanced_H

def fusion(input_image,edge_image):
    with tf.variable_scope("fusion"):
        feature_L=low_net(input_image,edge_image)
        print(feature_L.get_shape())
        feature_H=high_net(input_image,edge_image)
        print(feature_H.get_shape())
        feature_F=tf.concat([feature_L,feature_H],-1)
        #feature_attention=CBAM.cbam_block_parallel(feature_F)
        Wf1 = weight_variable([3, 3, 6, 64], name="Wf1"); bf1 = bias_variable([64], name="bf1");
        cf1 = tf.nn.relu(conv2d(feature_F, Wf1) + bf1)
        # residual 1
        Wf2 = weight_variable([3, 3, 64, 64], name="Wf2"); bf2 = bias_variable([64], name="bf2");
        cf2 = tf.nn.relu(_instance_norm(conv2d(cf1, Wf2) + bf2))

        Wf3 = weight_variable([3, 3, 64, 64], name="Wf3"); bf3 = bias_variable([64], name="bf3");
        cf3 = tf.nn.relu(_instance_norm(conv2d(cf2, Wf3) + bf3)) + cf1
        # Final
        Wf4 = weight_variable([3, 3, 64, 3], name="Wf4"); bf4 = bias_variable([3], name="bf4");
        cf4 = conv2d(cf3, Wf4) + bf4
        enhanced_out=tf.add(input_image,cf4)
    return enhanced_out

def adversarial1(image_):

    with tf.variable_scope("discriminator1"):

        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)#下采样了，stride=2
        
        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out
def adversarial2(image_):
    with tf.variable_scope("discriminator2"):

        conv1 = _conv_layer(image_, 64, 9, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 256, 3, 2)#下采样了，stride=2
        conv6 = _conv_layer(conv5, 256, 3, 1)#下采样了，stride=2        
        flat_size = 256 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out
def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias   
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init
