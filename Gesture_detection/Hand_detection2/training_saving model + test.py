# https://bskyvision.com/749
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# - 어플리케이션
#     : 이미지넷 (ImageNet)으로 사전 학습된 10가지 잘 알려진 모델을 제공
#     : (eg) Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet, DenseNet, NASNet, MobileNetV2TK 
#     : 이 모델들을 사용해 이미지 분류를 예측하고 특징을 추출하고 다양한 클래스 집합으로 모델을 세부 튜닝할 수 있음 (학습 속도 향상 가능)
#     : 케라스 예제 저장소 : 40개 이상의 샘플 모델 : 시각모델, 텍스트 및 시퀀스, 생성 모델 등
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
# from sklearn.model_selection import train_test_split

path_dir1 = './images/idle/'
path_dir2 = './images/question2/'

file_list1 = os.listdir(path_dir1) # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)

#%% train용 이미지 준비
num = 0;
train_img = np.float32(np.zeros((148, 224, 224, 3))) # 데이터셋 개수 140데이터 있는 자리에 지정하
train_label = np.float64(np.zeros((148, 1)))

for img_name in file_list1:
    img_path = path_dir1+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0
    num = num + 1

for img_name in file_list2:
    img_path = path_dir2+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 1
    num = num + 1
# train_img.shape
# 이미지 섞기
n_elem = train_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
train_label = train_label[indices]
train_img = train_img[indices]

# 학습, 시험 데이터로 나누기
trainX, testX, trainY, testY = train_test_split(train_img, train_label, test_size = 0.2)
#%% 
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)
 
base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))
 
GAP_layer = GlobalAveragePooling2D()
dense_layer = Dense(3, activation=tf.nn.softmax)
 
model = Sequential([
        base_model,
        GAP_layer,
        dense_layer
        ])
 
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
 
h = model.fit(trainX, trainY, epochs=10, validation_data = (testX,testY))

# plt.plot(h.history['loss'], label='Train loss')
# plt.plot(h.history['test_loss'], label='Test loss')
# plt.title('Loss trajectory')
# plt.legend()
# plt.show()

# save model
model.save("./models/question_model3.h5")

print("Saved model to disk")  