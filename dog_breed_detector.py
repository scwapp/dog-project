import cv2   
import numpy as np
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
import keras.utils as image       
from extract_bottleneck_features import extract_InceptionV3
from glob import glob
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))] # load list of dog names

def path_to_tensor(img_path):
    '''Loads image and converts it to 4D tensor.
    INPUT:
        - img_path: (str) image path
    OUTPUT:
        - tensor: (numpy array) 4d tensor with shape (1, 224, 224, 3)
    '''
    img = image.load_img(img_path, target_size=(224, 224)) #loads RGB image as PIL.Image.Image type
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img) # Output: (height, width, channels)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    tensor = np.expand_dims(x, axis=0)
    return tensor

def ResNet50_predict_labels(img_path):
    '''returns prediction vector for image located at img_path
    '''
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def face_detector(img_path):
    '''returns "True" if face is detected in image stored at img_path
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    '''returns "True" if a dog is detected in the image stored at img_path
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def prepare_inception_model():
    ''' Defines architecture of the CNN model used for classifying dog breeds.
    The model uses pre-trained InceptionV3 model as input. It has already been trained,
    and best weights are uploaded. 
    OUTPUT:
        - inception model: (keras sequential model) ready to use dog breed classifier model.
    '''

    inception_model = Sequential()
    num_classes = 133
    inception_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
    inception_model.add(Dense(256, activation='relu'))
    inception_model.add(Dropout(0.5))
    inception_model.add(Dense(num_classes, activation='softmax'))
    inception_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    inception_model.load_weights('saved_models/weights.best.inception.hdf5')
    return inception_model

inception_model = prepare_inception_model() 

def InceptionV3_predict_breed(img_path):
    '''Predicts dog breed found in image using InceptionV3 pre-trained model.'''
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = inception_model.predict(bottleneck_feature, verbose = 0)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def dog_breed_predictor(img_path):
    ''' This function returns different messages depending on the detected entities in the input image.
    INPUT:
        - img_path: (str) image path
    OUTPUT:
        - result: (dict) two entry dictionary:
            - what_detected: message displaying if dog, human, both or neither were detected
            - dog_breed: predicted dog breed if either a dog or a human were detected in the image
    '''
    is_human = face_detector(img_path)
    is_dog = dog_detector(img_path)
    result = {'what_detected': None, 'dog_breed': '-'}
    if is_human and is_dog:
        result['what_detected'] = 'Both a dog and a human ¯\_(ツ)_/¯'
    elif is_dog:
        result['what_detected'] = 'A dog ヽ(ヅ)ノ'
    elif is_human:
        result['what_detected'] = 'A human ヽ(•‿•)ノ'
    else:
        result['what_detected'] = 'Looks like there is neither a human nor a dog in your image (⌣̩̩́_⌣̩̩̀)'

    dog_breed = InceptionV3_predict_breed(img_path)
    
    if is_human or is_dog:
        result['dog_breed'] = f'{dog_breed}.'
    
    print()
    print(result['dog_breed'])

    return result