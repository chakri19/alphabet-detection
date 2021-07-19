import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps     
import os, ssl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image 

df = pd.read_csv("labels.csv")
classes = df["labels"]

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

x_train, x_test, y_train, y_test = train_test_split(classes, random_state = 9, test_size = 0.25, train_size = 0.75)
x_train_scale = x_train / 255
x_test_scale = x_test / 255

classifier = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_train_scale, y_train)

y_pred = classifier.predict(x_test_scale)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of model is ", accuracy)

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #drawing a box in center of videor
        height, width = gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0,255,0), 2)

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        im_pil = Image.fromarray(roi)
        image_pw = im_pil.convert("L")

        #inverting image correctly
        image_bw_resized = image_pw.resize((28,28), Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized) 
        pixel_filter = 20 
        #min pixel converts values into scaler quantities
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) 
        #limits the value
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255) 
        #gets max number
        max_pixel = np.max(image_bw_resized_inverted) 
        #converts values into an array
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)

        test_predict = classifier.predict(test_sample)
        cv2.imshow("frame", gray)

        if cv2.waitKey(1) & 0xFF == ord("Q"):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()