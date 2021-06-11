import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn


def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift
def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample(x1,x2,output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    #deconv_output =  tf.concat([deconv, x2],3)
    deconv_output =  deconv
    deconv_output.set_shape([None, None, None, output_channels])

    return deconv_output

def unet(input):
    with tf.variable_scope("generator2"):
        input1 = slim.conv2d(input, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv_x')#50*50
        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)

        conv2_1=slim.conv2d(conv1_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_3')
        conv2_1_1=tf.add(conv2_1,conv1_1)
        conv2_2=slim.conv2d(conv2_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_4')
        conv2_3=_instance_norm(conv2_2)

        conv2_4=tf.add(conv2_3,conv1_4)

        conv3_1=slim.conv2d(conv2_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_5')
        conv3_1_1=tf.add(conv2_1_1,conv3_1)
        conv3_1_2=tf.add(conv3_1_1,conv1_1)
        conv3_2=slim.conv2d(conv3_1_2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_6')
        conv3_3=_instance_norm(conv3_2)

        conv3_4=tf.add(conv3_3,conv2_4)

        pool1=slim.conv2d(conv3_4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25
        conv4_1=slim.conv2d(pool1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv4_2=slim.conv2d(conv4_1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,pool1)

        conv5_1=slim.conv2d(conv4_4,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_3')
        conv5_1_1=tf.add(conv4_1,conv5_1)
        conv5_2=slim.conv2d(conv5_1_1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_4')
        conv5_3=_instance_norm(conv5_2)

        conv5_4=tf.add(conv5_3,conv4_4)

        conv6_1=slim.conv2d(conv5_4,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_5')
        conv6_1_1=tf.add(conv5_1_1,conv6_1)
        conv6_1_2=tf.add(conv6_1_1,conv4_1)
        conv6_2=slim.conv2d(conv6_1_2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_6')
        conv6_3=_instance_norm(conv6_2)

        conv6_4=tf.add(conv6_3,conv5_4)

        up1 =  upsample(conv6_4,input1,64,128) #50*50
        up1_1 = tf.concat([up1, conv1_4],3)
        conv7_1=slim.conv2d(up1_1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,up1)

        conv8_1=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_3')
        conv8_1_1=tf.add(conv7_1,conv8_1)
        conv8_2=slim.conv2d(conv8_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_4')
        conv8_3=_instance_norm(conv8_2)

        conv8_4=tf.add(conv8_3,conv7_4)

        conv9_1=slim.conv2d(conv8_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_5')
        conv9_1_1=tf.add(conv8_1_1,conv9_1)
        conv9_1_2=tf.add(conv9_1_1,conv7_1)
        conv9_2=slim.conv2d(conv9_1_2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_6')
        conv9_3=_instance_norm(conv9_2)

        conv9_4=tf.add(conv9_3,conv8_4)

        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, output_shape=[tf.shape(input)[0],tf.shape(input)[1],tf.shape(input)[2],3], strides=[1, 2, 2, 1])
        out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out

def unet2(input):
    with tf.variable_scope("generator2"):
        input1 = slim.conv2d(input, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv_x')#50*50
        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')

        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)
        conv2_1=slim.conv2d(conv1_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_3')
        conv2_1_1=tf.add(conv2_1,conv1_1)
        conv2_2=slim.conv2d(conv2_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_4')
        conv2_3=_instance_norm(conv2_2)

        conv2_4=tf.add(conv2_3,conv1_4)

        pool1=slim.conv2d(conv2_4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25
        conv4_1=slim.conv2d(pool1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv4_2=slim.conv2d(conv4_1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,pool1)


        up1 =  upsample(conv4_4,input1,64,128) #50*50
        up1_1 = tf.concat([up1, conv1_4],3)
        conv7_1=slim.conv2d(up1_1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,up1)
        conv8_1=slim.conv2d(conv7_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_3')

        conv8_1_1=tf.add(conv7_1,conv8_1)
        conv8_2=slim.conv2d(conv8_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_4')
        conv8_3=_instance_norm(conv8_2)

        conv8_4=tf.add(conv8_3,conv7_4)


        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv10 = tf.nn.conv2d_transpose(conv8_4, deconv_filter, output_shape=[tf.shape(input)[0],tf.shape(input)[1],tf.shape(input)[2],3], strides=[1, 2, 2, 1])

        out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52


    return out

def unet3(input):
    with tf.variable_scope("generator3"):
        input1 = slim.conv2d(input, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv_x')#50*50
        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')

        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)

        pool1=slim.conv2d(conv1_4,64,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25
        conv4_1=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv4_2=slim.conv2d(conv4_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,pool1)


        up1 =  upsample(conv4_4,input1,64,64) #50*50
        up1_1 = tf.concat([up1, conv1_4],3)
        conv7_1=slim.conv2d(up1_1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,up1)      
        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv10 = tf.nn.conv2d_transpose(conv7_4, deconv_filter, output_shape=[tf.shape(input)[0],tf.shape(input)[1],tf.shape(input)[2],3], strides=[1, 2, 2, 1])

        out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52


    return out
