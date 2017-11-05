import numpy as np
import scipy as sp
from glob import glob
import tensorflow as tf
from sklearn import model_selection

data = bad_data + good_data
print('Bad & Good : %s' % len(data))

train, test = model_selection.train_test_split(data, train_size=.6, test_size=.4, random_state=11)

# Labels
train_label = np.array(tuple(item[0] for item in train), dtype=np.float32)
test_label = np.array(tuple(item[0] for item in test), dtype=np.float32)

# Images
#train_image = np.array(tuple(item[1] for item in train), dtype=np.uint8)
#test_image = np.array(tuple(item[1] for item in test), dtype=np.uint8)
train_image = np.stack((item[1].astype(np.float32).reshape((100 * 100,)) for item in train))#.astype(np.float32)
test_image = np.stack((item[1].astype(np.float32).reshape((100 * 100,)) for item in test))#.astype(np.float32)

print('Train : %s' % len(train))
print('Test : %s' % len(test))

n_class = len(set(tuple(item[0] for item in data)))

train_onehot = tf.one_hot(indices = train_label,
                          depth = n_class,
                          on_value = 1.,
                          off_value = 0.,
                          axis = -1)  # col : -1, idx : 0

test_onehot = tf.one_hot(indices = test_label,
                         depth = n_class,
                         on_value = 1.,
                         off_value = 0.,
                         axis = -1)  # col : -1, idx : 0

# Training Parameters

#learningRate = .1
#threshold = .5
#trainingEpochs = 1000
#displayStep = 100

learning_rate = .001
num_steps = 1000
batch_size = 5

# Network Parameters

num_input = 100 * 100 # data input (img shape)
num_classes = n_class # total classes (0-9 digits)
dropout = .75 # Dropout, probability to keep units

def conv_net(x_dict, n_classes, dropout, reuse, is_training):

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # The data input is a 1-D vector of 1444 features (38*38 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 100, 100, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out
  

def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_image}, y=train_label,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)


# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_image}, y=test_label,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
