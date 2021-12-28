import tensorflow as tf

import mnist

from fastapi import FastAPI


class MnistServer():
    def __init__(self):
        self._model: tf.keras.Model = mnist.build_and_compile_cnn_model()
    
    def call(self, data):
        print("hello")
        self._model.train_step(data)
        return {"message": "Hello World"}

app = FastAPI()
mnist_server = MnistServer()

@app.post("/")
def train_mnist_step(data):
    mnist_server.call(data)
