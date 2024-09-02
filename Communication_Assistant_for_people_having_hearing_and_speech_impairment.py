#!/usr/bin/env python
# coding: utf-8



# In[62]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tkinter import *
from  PIL import Image, ImageTk

# In[63]:


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# In[64]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results


# In[ ]:





# In[65]:


def draw_style_landmarks(image,results):
#     #draw face connections
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#                              mp_drawing.DrawingSpec(color=(244,164,96),thickness=1,circle_radius=1),
#                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1) 
#                              )
    #draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)
                             )     
    #draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(0,0,128),thickness=2,circle_radius=2)
                             )
    #draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                             )


# In[68]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_DATA')
# Actions that we try to detect
actions = np.array(['sign','more','learn','yes','no','thank you','sorry','hello','i love you','please','again','food','done','help','i','you'])
#Thirty videos worth of data
no_sequences = 30
#Videos are going to be 30 frames in length
sequence_length = 30


# In[70]:


for action in actions:
    for sequence in range (no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# In[ ]:





# In[71]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[72]:


label_map = {label:num for num, label in enumerate(actions)}


# In[73]:


label_map


# In[74]:


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH,action, str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[75]:


X = np.array(sequences)


# In[76]:


y = to_categorical(labels).astype(int)


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.05)


# In[78]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[79]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[80]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #change
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[81]:


res = [0.7, 0.2, 0.1]


# In[82]:


actions[np.argmax(res)]


# In[83]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[84]:


res = model.predict(X_test)


# In[85]:


model.load_weights('action.h5')


# In[87]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[88]:


yhat = model.predict(X_test)


# In[89]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


def window(frame,prediction):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    l1["image"] = img
    l2.config(text="Current Prediction: "+ str(prediction))
    x.update()


def onKeyPress(event):
    x.destroy()
    exit()


def detection():
    #1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.80

    cap = cv2.VideoCapture(0)
    #set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image1, results = mediapipe_detection(frame, holistic)
            print(results)
            draw_style_landmarks(image1, results)
            keypoints = extract_keypoints(results)
            #sequence.insert(0,keypoints)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                            
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                
                cv2.rectangle(image1, (0,0), (640,40), (245,117,16), -1)
                image1 = cv2.putText(image1, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,  cv2.LINE_AA)
                        
                window(image1,actions[np.argmax(res)])
                
                

                #cv2.imshow('OpenCV Feed', image1)
                #Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()



x = Tk()
x.geometry("900x650")
# x.bind('<KeyPress>', onKeyPress)
x.bind('q', onKeyPress)
x.configure(bg="light blue")
i = 0
l1 = Label(x,font=("times new roman",24))
l1.pack()
l2 = Label(x,font=("times new roman",24))
l2.pack()
l1.place(relx=0.5,rely=0.5,anchor="center")
l2.place(x=1,y=1)




detection()


# In[56]:

#cap.release()
#cv2.destroyAllWindows()

x.mainloop()