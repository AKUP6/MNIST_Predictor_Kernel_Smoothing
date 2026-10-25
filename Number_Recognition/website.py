import base64
import io
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template_string

from main import predict_image, X_test, y_test

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
<title>Number Recognition</title>
<style>
  body { font-family: sans-serif; text-align: center; margin-top: 40px; }
  button { font-size: 1.2rem; padding: 10px 24px; cursor: pointer; }
  #result { margin-top: 30px; }
  img { border: 1px solid #ccc; margin: 10px; }
  #prediction { font-size: 1.4rem; margin-top: 10px; }
</style>
</head>
<body>
  <h1>Number Recognition</h1>
  <button onclick="predict()">Predict</button>
  <div id="result">
    <div><img id="digit" width="140" style="display:none;"></div>
    <div id="prediction"></div>
    <div><img id="weights" width="420" style="display:none;"></div>
  </div>
  <script>
    async function predict() {
      document.getElementById('prediction').textContent = 'Predicting...';
      const res = await fetch('/predict');
      const data = await res.json();
      document.getElementById('digit').src = 'data:image/png;base64,' + data.digit_img;
      document.getElementById('digit').style.display = 'inline';
      document.getElementById('weights').src = 'data:image/png;base64,' + data.weights_img;
      document.getElementById('weights').style.display = 'inline';
      document.getElementById('prediction').textContent =
        'Predicted: ' + data.prediction + '  (True label: ' + data.true_label + ')';
    }
  </script>
</body>
</html>
"""


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/predict")
def predict():
    idx = random.randrange(len(X_test))
    image = X_test[idx]
    true_label = int(y_test[idx])

    pred, scores = predict_image(image)

    digit_fig, digit_ax = plt.subplots(figsize=(3, 3))
    digit_ax.imshow(image.reshape(28, 28), cmap="gray")
    digit_ax.axis("off")
    digit_img = _fig_to_base64(digit_fig)

    weights_fig, weights_ax = plt.subplots(figsize=(6, 3))
    weights_ax.bar(range(10), scores)
    weights_ax.set_xticks(range(10))
    weights_ax.set_xlabel("Digit")
    weights_ax.set_ylabel("Prediction weight")
    weights_ax.set_title("Prediction weights")
    weights_img = _fig_to_base64(weights_fig)

    return jsonify(
        prediction=int(pred),
        true_label=true_label,
        digit_img=digit_img,
        weights_img=weights_img,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
