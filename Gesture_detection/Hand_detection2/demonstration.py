import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
 
# %%
model = load_model('./models/question_model.h5')
model.summary()
 
# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()
      
# loop through frames

is_question = []
# previous_status = 0
# current_status = 0
# count = 0
while webcam.isOpened():
    
    # read frame from webcam 
    status, frame = webcam.read()
    
    if not status:
        break
    
    img = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
 
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction[0]) # 예측된 클래스 0, 1
    # print(prediction[0])
    print(predicted_class)
    
    if predicted_class == 0:
        me = "question"
        is_question.append(1)
    elif predicted_class == 1:
        me = "idle"
        is_question.append(0)
        
    if sum(is_question[-15:]) == 15:
        print('I GOT A QUESTION!!!')
    
    # display
    fontpath = "font/gulim.ttc"
    font1 = ImageFont.truetype(fontpath, 100)
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text((50, 50), me, font=font1, fill=(0, 0, 255, 3))
    frame = np.array(frame_pil)
    cv2.imshow('RPS', frame)
        
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows() 