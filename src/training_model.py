"""Here we train a DNN with the dataset which is created from the interactions with the env.

We assume that the data is available as a pandas DataFrame with the following columns:
- state: The state of the environment.
- action: The action taken by the data_creator.
- reward: The reward received by the data_creator.
- next_state: The next state of the environment.
- terminated: Whether the episode is terminated.
- truncated: Whether the episode is truncated.

We use a simple DNN with a single hidden layer to predict the next state based on the state and
action. The loss function is the mean squared error between the predicted next state and the actual
next state. We implement the model in keras and train it for 100 epochs, then save it.

"""

# 0. Import necessary libraries
import keras
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pdb


def create_model(hidden_layers: list[int]) -> keras.Model:
    """Create a simple NN with the provided hidden_layers.

    Args:
        hidden_layers (list[int]): List of hidden layer sizes.

    Returns:
        keras.Model: The created model.
    """
    state_input = layers.Input(shape=(4,))
    action_input = layers.Input(shape=(1,))
    concat = layers.Concatenate()([state_input, action_input])
    hidden = layers.Dense(hidden_layers[0], activation="relu")(concat)
    for hidden_size in hidden_layers[1:]:
        hidden = layers.Dense(hidden_size, activation="relu")(hidden)
    next_state_output = layers.Dense(4)(hidden)
    _model = Model(inputs=[state_input, action_input], outputs=next_state_output)
    _model.compile(optimizer="adam", loss="mse")
    _model.summary()
    return _model


# 1. Load the data set
list_of_data_files = ["data/random_agent_history.pkl"]
data = pd.concat([pd.read_pickle(file) for file in list_of_data_files])

# Check the shape of the data
print(f"Data shape: {data.shape}")

# 1.1. Extract the state, action, and next_state columns
state = np.array(data["state"].tolist())
action = np.array(data["action"].tolist())
next_state = np.array(data["next_state"].tolist())

# 2. Split the data into training and testing sets (80% training, 20% testing)
(
    state_train,
    state_test,
    action_train,
    action_test,
    next_state_train,
    next_state_test,
) = train_test_split(state, action, next_state, test_size=0.2, random_state=42)

# 3. Define the model
model = create_model([64, 64, 64])

# 4. Train the model with validation data (optional early stopping or validation split)
model.fit(
    [state_train, action_train], next_state_train, epochs=40, validation_split=0.2
)

# 5. Predict the next state on the test set
predicted_next_state = model.predict([state_test, action_test])

# 6. Calculate the mean squared error (MSE) between the predicted and actual next states
mse = mean_squared_error(next_state_test, predicted_next_state)
print(f"Mean Squared Error on Test Set: {mse}")

# 7. Save the model
# model.save("nn_models/random_model.h5")
model.save("nn_models/random_env.h5")
print("Model saved.")
