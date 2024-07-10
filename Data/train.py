from tkinter import *
import tkinter as tk
import cv2
import imutils
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from PIL import Image, ImageTk 
from keras import models
import tensorflow.keras.models as models
from tkinter import filedialog, Text, Button, END
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])  # Use 'accuracy' instead of 'acc' for TensorFlow 2.x
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
class App(tk.Tk):
    def __init__(self,master=None):
        super().__init__()

        # changing the title of our master widget      
        self.title("Dyslexia Prediction App")
        self.geometry("1400x800")

        # Initialize main container for frames
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Create content frame
        self.content_frame = tk.Frame(self.container, bg="white")
        self.content_frame.pack(fill="both", expand=True)

        # Load welcome background image
        welcome_image = Image.open("Data/welcomed.jpg")
        welcome_image = welcome_image.resize((1400, 800), Image.LANCZOS)  # Resize image
        self.welcome_bg = ImageTk.PhotoImage(welcome_image)
        self.welcome_label = tk.Label(self.content_frame, image=self.welcome_bg)
        self.welcome_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Schedule transition to next page after 3 seconds (3000 milliseconds)
        self.after(3000, self.create_widgets)
        contents ="  Waiting for Results..."
        
        print(contents)

    def create_widgets(self):
        # Destroy welcome image label
        self.welcome_label.destroy()

        # Black background frame
        black_frame = tk.Frame(self.content_frame, bg="black", width=1000, height=50)
        black_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Title label centered on black background
        title_label = tk.Label(black_frame, text="Dyslexia Prediction Tool", fg="blue", bg="black", font="Helvetica 30 bold italic", pady=10,height=0)
        title_label.pack()

        # Frame for buttons and result box
        button_frame = tk.Frame(black_frame, bg="white", width=800, height=600)
        button_frame.pack(pady=20)

        # Button: Start Eye Tracking
        start_eye_tracking_button = Button(button_frame, text="Start Eye Tracking", fg="blue", activebackground="dark red", width=20, command=self.EYE_Tracking)
        start_eye_tracking_button.grid(row=0, column=0, padx=20, pady=20)

        # Button: Select CSV File
        select_csv_button = Button(button_frame, text="Select CSV File", fg="blue", activebackground="dark red", width=20, command=self.select_csv_file)
        select_csv_button.grid(row=0, column=1, padx=20, pady=20)

        # Frame for viewing result or image
        result_frame = tk.Frame(black_frame, width=600, height=400, bg="light gray")
        result_frame.pack(pady=20)

        self.result_text = Text(black_frame, height=0, width=60,background='black')
        self.result_text.pack(pady=20)

        # Button: Predict Dyslexia
        predict_button = Button(black_frame, text="Predict", fg="green", activebackground="dark red", width=20, command=self.predict_dyslexia)
        predict_button.pack(pady=20)

    def select_csv_file(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select a CSV File",
                                              filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if filename:
            self.process_csv_file(filename)
    def train_model(self):
        global T
        contents="Training EYE tracking dataset"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        
        # Load dyslexia (D) and control (C) samples
        dyslexia_folder = "Data/Dyslexic"
        control_folder = "Data/Control"
        D_data = self.load_data_from_folder(dyslexia_folder, "Dyslexia")
        C_data = self.load_data_from_folder(control_folder, "Control")

        # Create graphs for control (C) and dyslexia (D) datasets
        C_graphs = [self.create_graph_from_data(data) for data in C_data]
        D_graphs = [self.create_graph_from_data(data) for data in D_data]

        # Extract features and labels for all graphs
        C_features_labels = [self.extract_features_and_labels(G) for G in C_graphs]
        D_features_labels = [self.extract_features_and_labels(G) for G in D_graphs]

        # Concatenate features and labels for all graphs
        all_features = np.concatenate([features for features, _ in C_features_labels] + [features for features, _ in D_features_labels], axis=0)
        all_labels = np.concatenate([labels.numpy() for _, labels in C_features_labels] + [labels.numpy() for _, labels in D_features_labels], axis=0)

        # Split data into training, validation, and test sets
        train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)
        train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

        # Build and compile the GraphSAGE model
        input_dim = train_features.shape[1]
        hidden_dim = 64
        output_dim = 2  # Assuming binary classification
        model = self.build_model(input_dim, hidden_dim, output_dim)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(train_features, train_labels, epochs=10, validation_data=(val_features, val_labels))

        # Save the trained model
        model_path = 'graph_sage_model'
        model.save(model_path)

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(test_features, test_labels)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Plot training history
        self.plot_training_history(history)

        # Convert the TensorFlow model into a format that can be serialized by pickle
        model_data = {
            'architecture': model.to_json(),
            'weights': model.get_weights()
        }

        # Save the model_data dictionary as a .pkl file using pickle
        with open('graph_sage_model.pkl', 'wb') as file:
            pickle.dump(model_data, file)

    def load_data_from_folder(self, folder, label):
        data_list = []
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                data = pd.read_csv(os.path.join(folder, filename))
                data["patient_id"] = filename.split(".")[0]  # Extract patient ID from filename
                data["label"] = label  # Add label column
                data_list.append(data)
        return data_list

    def create_graph_from_data(self, data):
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

    def extract_features_and_labels(self, graph):
        num_nodes = graph.number_of_nodes()
        node_features = [list(graph.nodes[node].values()) for node in range(num_nodes)]
        node_features = tf.constant(node_features, dtype=tf.float32)

        labels = [graph.nodes[node].get('label', 0) for node in range(num_nodes)]
        labels = tf.constant(labels, dtype=tf.int32)

        return node_features, labels

    def build_model(self, input_dim, hidden_dim, output_dim):
        # Custom GraphSAGE model for NetworkX graphs
        class GraphSAGEModel(tf.keras.Model):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(GraphSAGEModel, self).__init__()
                self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
                self.dense2 = tf.keras.layers.Dense(output_dim)

            def call(self, inputs):
                x = self.dense1(inputs)
                x = self.dense2(x)
                return x

        return GraphSAGEModel(input_dim, hidden_dim, output_dim)

    def plot_training_history(self, history):
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
        
    def Train_Eye_Movement(self, event=None):
        global T
        contents="Training EYE movement"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)
        
        data2=[]
        data=[]
        featurematrix=[]
        label=[]
        label2=[]
        cw_directory = os.getcwd()
        #cw_directory='D:/Hand gesture/final_code'
        folder='C:\sumantha\project\Data\eye dataset'
        for filename in os.listdir(folder):
            
            sub_dir=(folder+'/' +filename)
            for img_name in os.listdir(sub_dir):
                img_dir=str(sub_dir+ '/' +img_name)
                print(int(filename),img_dir)
                img = cv2.imread(img_dir)
                # Resize image
                img = cv2.resize(img,(128,128))
                if len(img.shape)==3:
                    img2 = cv2.resize(img,(32,32))
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    img2=img2.flatten()
                    data2.append(img2/255.0)
                    label2.append(int(filename))
                    
                data11=np.array(img)
                data.append(data11/255.0)
                label.append(int(filename))
         

        #target1=train_targets[label]
        ##

        def train_CNN(data, label):
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(36))  # Assuming 36 classes for output

            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

            X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.20)

            history = model.fit(np.array(X_train), np.array(Y_train), epochs=20, 
                                validation_data=(np.array(X_test), np.array(Y_test)))
            
            show_history_graph(history)

            test_loss, test_acc = model.evaluate(np.array(X_test), np.array(Y_test), verbose=2)
            print("Testing Accuracy is ", test_acc)
            print("Testing loss is ", test_loss)

            model.save('eye_movement_trained.h5')
            return model



        # CNN Training
        model_CNN = train_CNN(data,label)
        Y_CNN=model_CNN.predict(np.array(data))
        contents="Training EYE movement completed"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END,contents)
        print(contents)

    def EYE_Tracking(self, event=None):
        global T
        contents = "Starting Eye Movement based Dyslexia Prediction"
        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END, contents)
        print(contents)

        list1 = ['looking at center', 'looking at left', 'looking at right', 'looking at up', 'looking at down']
        eye_cnn = tf.keras.models.load_model('C:\sumantha\project\eye_movement_trained.h5')
        def histogram_equalization(img):
            r,g,b = cv2.split(img)
            f_img1 = cv2.equalizeHist(r)
            f_img2 = cv2.equalizeHist(g)
            f_img3 = cv2.equalizeHist(b)
            img = cv2.merge((f_img1,f_img2,f_img3))
            return img
        def get_index_positions_2(list_of_elems, element):
            ''' Returns the indexes of all occurrences of give element in
            the list- listOfElements '''
            index_pos_list = []
            for i in range(len(list_of_elems)):
                if list_of_elems[i] == element:
                    index_pos_list.append(i)
            return index_pos_list

        # Load the Haar cascade for eye detection
        eye_cascade = cv2.CascadeClassifier('Data/haar cascade files/haarcascade_eye.xml')
        prototxt_path = os.path.join('Data/model_data/deploy.prototxt')
        caffemodel_path = os.path.join('Data/model_data/weights.caffemodel')
        model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        # Start video capture
        vs = cv2.VideoCapture(0)
        eyemovement = []
        max_eye_images = 20  # Maximum number of eye images to capture
        eye_images_captured = 0
        Dyslexia_result = []
        n1 = 0
        n2 = 10

        while True:
            ret, frame = vs.read()
            frame = imutils.resize(frame, width=750, height=512)

            # Preprocess the frame
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            model.setInput(blob)
            detections = model.forward()

            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                confidence = detections[0, 0, i, 2]

                if confidence > 0.40:
                    # Extract the eye region
                    eye_region = frame[startY:endY, startX:endX]
                    eye_region = histogram_equalization(eye_region)
                    eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

                    # Detect eyes within the eye region
                    eyes = eye_cascade.detectMultiScale(eye_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    # Process detected eyes
                    for (ex, ey, ew, eh) in eyes:
                        eye_img = eye_region[ey:ey + eh, ex:ex + ew]
                        eye_img_resized = cv2.resize(eye_img, (128, 128))
                        eye_prediction = np.argmax(eye_cnn.predict(np.expand_dims(eye_img_resized, axis=0)))

                        # Append eye movement prediction to the list
                        eyemovement.append(eye_prediction)

                        # Draw bounding box around the eye
                        cv2.rectangle(frame, (startX + ex, startY + ey), (startX + ex + ew, startY + ey + eh), (0, 255, 0), 2)

            cv2.imshow("Frame", frame)

            # Detect dyslexia based on collected eye movement predictions
            if len(eyemovement) >= n2:
                eye_array = eyemovement[n1:n2]
                if len(np.unique(eye_array)) > 2:
                    Dyslexia = 1  # Symptoms of Dyslexia detected
                else:
                    Dyslexia = 0  # No Symptoms of Dyslexia detected

                n1 += 10
                n2 += 10
                Dyslexia_result.append(Dyslexia)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vs.release()
        cv2.destroyAllWindows()

        # Analyze Dyslexia results
        num_positive = sum(Dyslexia_result)
        num_negative = len(Dyslexia_result) - num_positive

        if num_positive >= 5 or num_positive > num_negative:
            contents = "Symptoms of Dyslexia detected"
            Dyslexia_eye = 1
        else:
            contents = "No Symptoms of Dyslexia detected"
            Dyslexia_eye = 0

        T = Text(self, height=20, width=25)
        T.pack()
        T.place(x=650, y=100)
        T.insert(END, contents)
        print(contents)

        self.Dyslexia_eye = Dyslexia_eye


    
        
app = App()
app.mainloop()
