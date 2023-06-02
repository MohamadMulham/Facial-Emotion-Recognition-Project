import cv2
import numpy as np
from keras.models import model_from_json
from matplotlib import pyplot as plt
from PIL import Image


img_size= 224

# Start video capture from default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



emotion = ['angry', 'fear', 'happy','neutral','sad','surprise']
name = './model'
modelFile = open( name+'.json', 'r')
loaded_model_json = modelFile.read()
modelFile.close()
emotion_model = model_from_json(loaded_model_json)
# load weights into new model
emotion_model.load_weights(name+".h5")
print("Loaded model from disk")
# load json and create mode

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    cv2.resize(frame, (1280, 720))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        break
    #time.sleep(0.3)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray[y:y + h, x:x + w]
        cropped_img = roi_gray_frame
        cv2.imwrite("./img.jpg", roi_gray_frame)
        img = Image.open("./img.jpg")
        img= img.resize((img_size,img_size)) 
        img = np.expand_dims(np.expand_dims(img, -1), 0)
        emotion_prediction = emotion_model.predict(img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
