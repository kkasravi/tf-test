from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os

assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'

assert float('.'.join(tf.__version__.split('.')[:2])) >= 1.14, 'Make sure that Tensorflow version is at least 1.14'

TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']

def create_model(input_shape):
  """Creates a simple convolutional neural network model using the Keras API"""
  return tf.keras.Sequential([
      tf.keras.layers.Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])

def loss(model, x, y):
  """Calculates the loss given an example (x, y)"""
  logits = model(x)
  return logits, tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

def grad(model, x, y):
  """Calculates the loss and the gradients given an example (x, y)"""
  logits, loss_value = loss(model, x, y)
  return logits, loss_value, tf.gradients(loss_value, model.trainable_variables)


tf.keras.backend.clear_session()

resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.contrib.distribute.TPUStrategy(resolver)

# Load MNIST training and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# All MNIST examples are 28x28 pixel greyscale images (hence the 1
# for the number of channels).
input_shape = (28, 28, 1)

# Only specific data types are supported on the TPU, so it is important to
# pay attention to these.
# More information:
# https://cloud.google.com/tpu/docs/troubleshooting#unsupported_data_type
x_train = x_train.reshape(x_train.shape[0], *input_shape).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], *input_shape).astype(np.float32)
y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

# The batch size must be divisible by the number of workers (8 workers),
# so batch sizes of 8, 16, 24, 32, ... are supported.
BATCH_SIZE = 32

NUM_EPOCHS = 5

train_steps_per_epoch = len(x_train) // BATCH_SIZE
test_steps_per_epoch = len(x_test) // BATCH_SIZE

with strategy.scope():
  model = create_model(input_shape)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

  training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
  training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      'training_accuracy', dtype=tf.float32)
  test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      'test_accuracy', dtype=tf.float32)

with strategy.scope():
  def train_step(inputs):
    """Each training step runs this custom function which calculates
    gradients and updates weights.
    """
    x, y = inputs

    logits, loss_value, grads = grad(model, x, y)

    update_loss = training_loss.update_state(loss_value)
    update_accuracy = training_accuracy.update_state(y, logits)

    # Show that this is truly a custom training loop
    # Multiply all gradients by 2.
    grads = grads * 2

    update_vars = optimizer.apply_gradients(
        zip(grads, model.trainable_variables))

    with tf.control_dependencies([update_vars, update_loss, update_accuracy]):
      return tf.identity(loss_value)

  def test_step(inputs):
    """Each training step runs this custom function"""
    x, y = inputs

    logits, loss_value = loss(model, x, y)

    update_loss = test_loss.update_state(loss_value)
    update_accuracy = test_accuracy.update_state(y, logits)

    with tf.control_dependencies([update_loss, update_accuracy]):
      return tf.identity(loss_value)

def run_train(profiler, epoch):
  # Train
  if epoch % 10000 == 0:
    run_meta = tf.compat.v1.RunMetadata()
    print('before session.run')
    session.run(train_iterator_init,
                options=tf.compat.v1.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE), 
                run_metadata=run_meta)
    profiler.add_step(epoch, run_meta)
    opts = tf.python.profiler.ProfileOptionBuilder.time_and_memory()
    profiler.profile_operations(options=opts)
    print('after profiler.add_step')
  else:
    session.run(train_iterator_init)
    while True:
      try:
        session.run(dist_train)
      except tf.errors.OutOfRangeError:
        break
    print('Train loss: {:0.4f}\t Train accuracy: {:0.4f}%'.format(
        session.run(training_loss_result),
        session.run(training_accuracy_result) * 100))
    training_loss.reset_states()
    training_accuracy.reset_states()

def run_test():
  # Test
  session.run(test_iterator_init)
  while True:
    try:
      session.run(dist_test)
    except tf.errors.OutOfRangeError:
      break
  print('Test loss: {:0.4f}\t Test accuracy: {:0.4f}%'.format(
      session.run(test_loss_result),
      session.run(test_accuracy_result) * 100))
  test_loss.reset_states()
  test_accuracy.reset_states()

with strategy.scope():
  training_loss_result = training_loss.result()
  training_accuracy_result = training_accuracy.result()
  test_loss_result = test_loss.result()
  test_accuracy_result = test_accuracy.result()
  
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  cluster_spec = resolver.cluster_spec()
  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())

  print('Starting training...')

  # Do all the computations inside a Session (as opposed to doing eager mode)
  with tf.Session(target=resolver.master(), config=config) as session:
    all_variables = (
        tf.global_variables() + training_loss.variables +
        training_accuracy.variables + test_loss.variables +
        test_accuracy.variables)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE, drop_remainder=True)
    train_iterator = strategy.make_dataset_iterator(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE, drop_remainder=True)
    test_iterator = strategy.make_dataset_iterator(train_dataset)
    
    train_iterator_init = train_iterator.initialize()
    test_iterator_init = test_iterator.initialize()

    session.run([v.initializer for v in all_variables])
    
    dist_train = strategy.experimental_run(train_step, train_iterator).values
    dist_test = strategy.experimental_run(test_step, test_iterator).values

    profiler = tf.python.profiler.Profiler(session.graph)
    # Custom training loop
    for epoch in range(0, NUM_EPOCHS):
      print('Starting epoch {}'.format(epoch))

      run_train(profiler, epoch)

      run_test()
