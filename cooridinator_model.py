import tensorflow as tf
import requests

class CoordinatorModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
    def train_step(self, data):
        x, y = data 
        with tf.GradientTape() as tape:
            y_pred = requests.post("localhost:8000", data=x)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
