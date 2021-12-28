import os
import json

import tensorflow as tf
import mnist
import cluster_config

import cooridinator_model

import requests


per_worker_batch_size = 64
# tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = 1
NUM_PS = 1

# global_batch_size = per_worker_batch_size * num_workers
# multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

working_dir = '/tmp/my_working_dir'
log_dir = os.path.join(working_dir, 'log')
ckpt_filepath = os.path.join(working_dir, 'ckpt')
backup_dir = os.path.join(working_dir, 'backup')

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir),
]

def create_cluster_spec():
    cluster_spec = tf.train.ClusterSpec(cluster_config.CLUSTER_CONFIG["cluster"])
    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
    cluster_spec, rpc_layer="grpc")
    return cluster_resolver

cluster_resolver = create_cluster_spec()
variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)

multi_worker_dataset = mnist.mnist_dataset()

with strategy.scope():
    multi_worker_model: tf.keras.Model = cooridinator_model.CoordinatorModel()
    multi_worker_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
    
    current_epoch = tf.Variable(1)

    print(current_epoch)

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(multi_worker_dataset):
            with tf.GradientTape() as tape:
                print(x_batch_train)
                y_pred = requests.post("localhost:8000", data=x_batch_train)

print("fit model")
print(multi_worker_dataset)



# multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70, callbacks=callbacks)