import tensorflow as tf
import os 
import cluster_config

def create_paramter_server():
    """Creates and starts local servers and returns the cluster_resolver."""
    cluster_spec = tf.train.ClusterSpec(cluster_config.CLUSTER_CONFIG["cluster"])
    worker_server = tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=0,
        protocol="grpc")
    worker_server.start()

    ps_server = tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=0,
        protocol="grpc")
    ps_server.join()

os.environ["GRPC_FAIL_FAST"] = "use_caller"
create_paramter_server()
