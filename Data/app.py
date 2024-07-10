from tkinter import *
import tkinter as tk
import cv2
import imutils
import numpy as np
import os
import pickle
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
        self.after(7000, self.create_widgets)
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
        self.select_csv_button = Button(button_frame, text="Select CSV File", fg="blue", activebackground="dark red", width=20, command=self.select_csv_file)
        self.select_csv_button.grid(row=0, column=1, padx=20, pady=20)

        # Frame for viewing result or image
        self.result_frame = tk.Frame(black_frame, width=600, height=400, bg="light gray")
        self.result_frame.pack(pady=20)

        self.result_text = Text(black_frame, height=3, width=60)
        self.result_text.pack(pady=20)

        # Button: Predict Dyslexia
        predict_button = Button(black_frame, text="Predict", fg="green", activebackground="dark red", width=20,command=self.predict_dyslexia)
        predict_button.pack(pady=20)


    def EYE_Tracking(self):
        contents = "Starting Eye Movement based Dyslexia Prediction"
        self.update_result_text(contents)

        list1 = ['looking at center', 'looking at left', 'looking at right', 'looking at up', 'looking at down']
        eye_cnn = tf.keras.models.load_model('eye_movement_trained.h5')

        def histogram_equalization(img):
            if img is None or img.size == 0:
                return img
            r, g, b = cv2.split(img)
            f_img1 = cv2.equalizeHist(r)
            f_img2 = cv2.equalizeHist(g)
            f_img3 = cv2.equalizeHist(b)
            img = cv2.merge((f_img1, f_img2, f_img3))
            return img

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
                        pil_img = Image.fromarray(cv2.cvtColor(eye_img_resized, cv2.COLOR_BGR2RGB))

                        # Display the image in result frame
                        self.display_eye_image(pil_img)

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
            self.Dyslexia_eye = 1
        else:
            contents = "No Symptoms of Dyslexia detected"
            self.Dyslexia_eye = 0

        self.update_result_text(contents)

    def update_result_text(self, message):
        # Clear the text widget and update with new message
       self.result_text.config(state=NORMAL)
       self.result_text.delete(1.0, END)
       self.result_text.insert(END, message)
       self.result_text.config(state=DISABLED)
    def display_eye_image(self, pil_image):
        try:
        # Convert PIL Image to numpy array
            cv2_image = np.array(pil_image)

        # Check if the numpy array has valid shape (height, width, channels)
            if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
                # Convert OpenCV BGR image to RGB format
                rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

                # Resize the RGB image to fit the result_frame
                pil_image_resized = Image.fromarray(rgb_image).resize((600, 400), Image.LANCZOS)

                # Convert PIL Image to Tkinter PhotoImage
                tk_image = ImageTk.PhotoImage(pil_image_resized)

                # Clear the result_frame
                for widget in self.result_frame.winfo_children():
                    widget.destroy()

                # Display the image in the result_frame using a Label widget
                label = tk.Label(self.result_frame, image=tk_image)
                label.image = tk_image  # Keep a reference to prevent garbage collection
                label.pack()

            else:
                print("Invalid image format: Expected RGB image with 3 channels")

        except Exception as e:
            print(f"Error displaying eye image: {e}")
    def select_csv_file(self):
        # Open file dialog to select a CSV file
        filename = filedialog.askopenfilename(initialdir="/", title="Select a CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        
        if filename:
            # Update the button text to display the selected file name
            self.select_csv_button.config(text=f"Selected CSV file: {os.path.basename(filename)}")
            
            # Store the selected filename for later use
            self.selected_filename = filename

    def predict_dyslexia(self):
        if not hasattr(self, 'selected_filename') or not self.selected_filename:
            self.update_result_text("Please select a CSV file first")
            return

        dyslexia_detected = self.predict_dyslexia_from_csv(self.selected_filename)

        if dyslexia_detected:
            self.result_text.config(state=NORMAL)
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, "Symptoms of Dyslexia detected")
            self.result_text.config(state=DISABLED)
        else:
            self.result_text.config(state=NORMAL)
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, "No Symptoms of Dyslexia detected")
            self.result_text.config(state=DISABLED)

    def predict_dyslexia_from_csv(self, filename):
        try:
            df = pd.read_csv(filename)
            
            # Load the saved GraphSAGE model
            with open('graph_sage_model.pkl', 'rb') as file:
                model_data = pickle.load(file)

            loaded_model = tf.keras.models.model_from_json(model_data['architecture'])
            loaded_model.set_weights(model_data['weights'])

            # Create a graph from the CSV data
            G = create_graph_from_data(df)
            
            # Extract features and labels from the graph
            features, _ = extract_features_and_labels(G, None)  # Labels are not used for prediction
            
            # Make predictions using the loaded model
            predictions = loaded_model.predict(features)
            
            # Convert predictions to labels
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Assuming class 0 corresponds to control and class 1 corresponds to dyslexia
            # You can adjust this based on your actual class labels
            num_control = np.sum(predicted_labels == 0)
            num_dyslexia = np.sum(predicted_labels == 1)

            if num_control > num_dyslexia:
                return "Control"
            else:
                return "Dyslexia"

        except Exception as e:
                print(f"Error processing CSV file: {e}")
                self.update_result_text(f"Error processing CSV file: {e}")
                return "Error"


    def update_result_text(self, message):
        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, message)
        self.result_text.config(state=DISABLED)

if __name__ == "__main__":
    app = App()
    app.mainloop()
