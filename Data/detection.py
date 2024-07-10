import os
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import itertools
import pickle

# Function to load data from folder
def load_data_from_folder(folder, label):
    data_list = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            data = pd.read_csv(os.path.join(folder, filename))
            if 'Unnamed: 0' in data.columns:
                data = data.drop(columns=['Unnamed: 0'])  # Drop 'Unnamed: 0' column
            data_list.append(data)
    return data_list

# Load dyslexia (D) and control (C) samples
dyslexia_folder = "Data/Dyslexic"
control_folder = "Data/Control"
D_data = load_data_from_folder(dyslexia_folder, 1)  # Label Dyslexia as 1
C_data = load_data_from_folder(control_folder, 0)  # Label Control as 0

# Function to create a graph from data
def create_graph_from_data(data):
    G = nx.Graph()
    for index, row in data.iterrows():
        node_attrs = {'LX': row['LX'], 'LY': row['LY'], 'RX': row['RX'], 'RY': row['RY']}
        G.add_node(index, **node_attrs)
    
    for i in range(len(data) - 1):
        coords_i = data.loc[i, ['LX', 'LY', 'RX', 'RY']].values
        coords_next = data.loc[i+1, ['LX', 'LY', 'RX', 'RY']].values
        distance = np.linalg.norm(coords_i - coords_next)
        
        if distance < 5:
            G.add_edge(i, i+1)
    
    return G

# Create graphs for control (C) and dyslexia (D) datasets
C_graphs = [create_graph_from_data(data) for data in C_data]
D_graphs = [create_graph_from_data(data) for data in D_data]

# Function to extract node features and labels from a NetworkX graph
def extract_features_and_labels(graph):
    num_nodes = graph.number_of_nodes()
    node_features = [list(graph.nodes[node].values()) for node in range(num_nodes)]
    node_features = tf.constant(node_features, dtype=tf.float32)
    return node_features

# Extract features and labels for all graphs
C_features = [extract_features_and_labels(G) for G in C_graphs]
D_features = [extract_features_and_labels(G) for G in D_graphs]

# Concatenate features and labels for all graphs
all_features = np.concatenate([features.numpy() for features in C_features] + [features.numpy() for features in D_features], axis=0)
all_labels = np.concatenate([np.zeros(len(features), dtype=int) for features in C_features] + [np.ones(len(features), dtype=int) for features in D_features], axis=0)

# Split data into training, validation, and test sets
train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

# Define the GraphSAGE model
class GraphSAGEModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGEModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# Build and compile the GraphSAGE model
input_dim = train_features.shape[1]
hidden_dim = 64
output_dim = 2  # Binary classification
model = GraphSAGEModel(input_dim, hidden_dim, output_dim)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_features, train_labels, epochs=20, validation_data=(val_features, val_labels))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_features, test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Plot training and validation metrics
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.plot(history.history['loss'], label='Training Loss', color='green')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.title('Training and Validation Metrics')
plt.grid(True)
plt.show()

# Predictions on the test set
y_pred = np.argmax(model.predict(test_features), axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(test_labels, y_pred, target_names=["Control", "Dyslexia"])
print("Classification Report:")
print(class_report)

# Plotting Confusion Matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(["Control", "Dyslexia"]))
plt.xticks(tick_marks, ["Control", "Dyslexia"], rotation=45)
plt.yticks(tick_marks, ["Control", "Dyslexia"])
thresh = conf_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Save the model
model.save('graph_sage_model')

# Load the model
loaded_model = tf.keras.models.load_model('graph_sage_model')

# Serialize model architecture using JSON and weights using pickle
model_data = {
    'architecture': loaded_model.to_json(),
    'weights': loaded_model.get_weights()
}
with open('graph_sage_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

model.save_weights('graph_sage_model_weights.h5')
