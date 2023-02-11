[//]: # (Image References)
[image1]: ./web_view_example.PNG "web app view"

## Dog Breed Detector web app using CNN 
This project has been built following Udacity's Dog Breed Classifier's project. An algorithm has been developped which detects if there is a human or a dog in a given image, and predicts a dog breed. The algorithm uses the InceptionV3 and ResNet50 pre-trained models to detect dogs, and OpenCV's Haar feature based cascade classifier to detect humans. It has been deployed on a simple web app developped in Flask, where you can upload your own pictures and find out their most resembling dog breed.


![Web App view][image1]

## Project Files
### Ipython notebook
- The `dog_app.ipynb` file were that shows the whold process of the project development.
- `dog_app.html`: an HTML export of the `dog_app.ipynb`.
- `/images`: additional images used for the project.
- `requirements.txt`: required python libraries.
- `/saved_models`: weights of best performing model.
- `extract_bottleneck_features.py`: functions to extract bottleneck features from pre-trained models.
### Web app
- `web_app.py`: Flask web application.
- `/templates/index.html`: html file for web app.
- `dog_breed_detector.py`: functions for web app (similar to the ones used in `dog_app.ipynb`).

- `/static/images`: folder to save web app uploaded images.


## Instructions

You can run the web app locally on your browser simply by executing `web_app.py` and navigating to *http://127.0.0.1:5000/*.
The datasets used to train the models can be downloaded following these links: [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip), [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  

## Conclusion
The developped algorithm achieved an 80% accuracy in classifying dog breeds on the test set. The project resulted in a fun web app that allows user uploaded images to be classified.
