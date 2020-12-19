# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
 
# path_question_train = './images/question/'
# path_question_test = './images/question_test/'

# path_idle_train = './images/idle/'
# path_idel_test = './images/idle_test/'

# train_question = os.listdir(path_question_train) # path에 존재하는 파일 목록 가져오기
# test_qustion = os.listdir(path_question_test)

# train_idle = os.listdir(path_idle_train)
# test_idle = os.listdir(path_idel_test)
 
# #%% train용 이미지 준비
# num = 0;
# train_img = np.float32(np.zeros((1268, 224, 224, 3))) # 394+413+461
# train_label = np.float64(np.zeros((1268, 1)))

# test_img = np.float32(np.zeros((1268, 224, 224, 3)))
# test_label = np.float64(np.zeros((1268, 1)))

# test_label

# for img_name in train_question:
#     img_path = path_question_train+img_name
#     img = load_img(img_path, target_size=(224, 224))
    
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     train_img[num, :, :, :] = x
    
#     train_label[num] = 0 # question
#     num = num + 1

# for img_name in test_qustion:
#     img_path = path_question_test+img_name
#     img = load_img(img_path, target_size=(224, 224))
    
#     y = img_to_array(img)
#     y = np.expand_dims(x, axis=0)
#     y = preprocess_input(x)
#     train_img[num, :, :, :] = y
    
#     test_label[num] = 0 # question
#     num = num + 1
    
# for img_name in train_idle:
#     img_path = path_idle_train+img_name
#     img = load_img(img_path, target_size=(224, 224))
    
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     train_img[num, :, :, :] = x
    
#     train_label[num] = 1 # question
#     num = num + 1

# for img_name in test_idle:
#     img_path = path_idel_test+img_name
#     img = load_img(img_path, target_size=(224, 224))
    
#     y = img_to_array(img)
#     y = np.expand_dims(x, axis=0)
#     y = preprocess_input(x)
#     test_img[num, :, :, :] = y
    
#     train_label[num] = 1 # question
#     num = num + 1
# # 이미지 섞기
     
# n_elem = train_label.shape[0]
# indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
# train_label = train_label[indices]
# train_img = train_img[indices]
 
# #%% 
# # create the base pre-trained model
# IMG_SHAPE = (224, 224, 3)
 
# base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
# base_model.trainable = False
# base_model.summary()
# print("Number of layers in the base model: ", len(base_model.layers))
 
# GAP_layer = GlobalAveragePooling2D()
# dense_layer = Dense(3, activation=tf.nn.softmax)
 
# model = Sequential([
#         base_model,
#         GAP_layer,
#         dense_layer
#         ])
 
# base_learning_rate = 0.001
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()
 
# model.fit(train_img, train_label, epochs=5)
 
# # save model
# model.save("./models/question_model.h5")
 
# print("Saved model to disk")  

# ----------
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
 
path_dir1 = './images/question/'
path_dir2 = './images/idle/'
 
file_list1 = os.listdir(path_dir1) # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
 
#%% train용 이미지 준비
num = 0;
train_img = np.float32(np.zeros((1268, 224, 224, 3))) # 394+413+461
train_label = np.float64(np.zeros((1268, 1)))

for img_name in file_list1:
    img_path = path_dir1+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1
 
for img_name in file_list2:
    img_path = path_dir2+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 1 # paper
    num = num + 1

# 이미지 섞기
     
n_elem = train_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
train_label = train_label[indices]
train_img = train_img[indices]
 
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
 
model.fit(train_img, train_label, epochs=1)
 
# save model
model.save("./models/question_model.h5")

print("Saved model to disk")  