import mnist
import tensorflow as tf

batch_size = 64
single_worker_dataset = mnist.mnist_dataset(batch_size)
single_worker_model = mnist.ConnectedModel()
single_worker_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy'])
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)