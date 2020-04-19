import tensorflow as tf

def print_activation(t):
    print(t.op.name,'',t.get_shape().as_list())
def model(images,dropout):
    parameters = []
    #conv1
    with tf.variable_scope('conv1'):
        filter = tf.Variable(tf.truncated_normal([5,5,3,96],dtype=tf.float32 ,stddev =1e-1),name='weights')
        conv = tf.nn.conv2d(images,filter,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[96],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.add(conv,biases)
        conv1 = tf.nn.relu(bias)
        print_activation(conv1)
        parameters +=[filter,biases]

    #lrn1
    with tf.name_scope("pool1"):
        lrn1 = tf.nn.local_response_normalization(conv1,alpha=1e-4,beta=0.75,depth_radius=2,bias=2.0)
        #pool1
        pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
        print_activation(pool1)

    #conv2
    with tf.variable_scope('conv2'):
        filter = tf.Variable(tf.truncated_normal([5,5,96,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool1,filter,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.add(conv,biases)
        conv2 = tf.nn.relu(bias)
        print_activation(conv2)
        parameters +=[filter,biases]

    #lrn2
    with tf.name_scope('pool2'):
        lrn2 = tf.nn.local_response_normalization(conv2,alpha=1e-4,beta=0.75,depth_radius=2,bias=2.0)
        #pool2
        pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
        print_activation(pool2)

    #conv3
    with tf.variable_scope('conv3'):
        filter = tf.Variable(tf.truncated_normal([3,3,256,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool2,filter,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.add(conv,biases)
        conv3 = tf.nn.relu(bias)
        print_activation(conv3)
        parameters +=[filter,biases]
     #   pool3 = tf.nn.max_pool(conv3,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool3')
    #    print_activation(pool3)

    #conv4
    with tf.variable_scope('conv4'):
        filter = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.add(conv, biases)
        conv4 = tf.nn.relu(bias)
        print_activation(conv4)
        parameters += [filter, biases]
#        pool4 = tf.nn.max_pool(conv4,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool4')
 #       print_activation(pool4)

    #conv5
    with tf.variable_scope('conv5'):
        filter = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, filter, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.add(conv, biases)
        conv5 = tf.nn.relu(bias)
        print_activation(conv5)
        parameters += [filter, biases]
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
        print_activation(pool5)

    #fcn(layer6)
    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable("weight",[4096 ,4096],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc = tf.reshape(pool5,[-1,fc1_weights.get_shape().as_list()[0]])
        fc1_biases = tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),trainable=True,name='biases')
        fc1=tf.add(tf.matmul(fc,fc1_weights),fc1_biases)
        fc1 = tf.nn.relu(fc1)
        parameters += [fc1_weights, fc1_biases]
        print_activation(fc1)
        fc1 = tf.nn.dropout(fc1,dropout)

    #fc2(layer7)
    with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable("weight", [4096, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases')
        fc2 = tf.add(tf.matmul(fc1, fc2_weights), fc2_biases)
        fc2 = tf.nn.relu(fc2)
        parameters += [fc2_weights, fc2_biases]
        print_activation(fc2)
        fc2 = tf.nn.dropout(fc2, dropout)
    #layer8
    with tf.variable_scope('layer8-out'):
        # 輸出層
        out_weights = tf.get_variable("weight", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        out_biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
        out = tf.add(tf.matmul(fc2, out_weights), out_biases)
        parameters += [out_weights, out_biases]
    return out,parameters
