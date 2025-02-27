import tensorflow as tf
from memoraith import profile_model, set_output_path

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

@profile_model(memory=True, computation=True, gpu=True)
def train_model(model, epochs=5):
    model.compile(optimizer='adam', loss='mse')

    # Generate some dummy data
    x_train = tf.random.normal((1000, 10))
    y_train = tf.random.normal((1000, 1))

    model.fit(x_train, y_train, epochs=epochs, verbose=1)

if __name__ == "__main__":
    set_output_path('tensorflow_profiling_results/')
    model = create_model()
    train_model(model)
