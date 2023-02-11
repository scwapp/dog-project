from flask import Flask, request, render_template
import base64
from dog_breed_detector import dog_breed_predictor

app = Flask(__name__,static_folder="static")

img_path = "static/images/uploaded_image.jpg"
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            # Save the image to the specified path
            image.save(img_path)
            result = dog_breed_predictor(img_path)
            return render_template("index.html", human_or_dog = result['what_detected'],
            dog_breed = result['dog_breed'])

    return render_template("index.html", human_or_dog = '', dog_breed = '')

    
if __name__ == "__main__":
    app.run(debug = True)