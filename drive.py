# For server and network framework
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

# For image acquisition and manipulation
import base64
import matplotlib.image as img
from io import BytesIO

import torch
from model import Model
from main import configure

# Initialize server and application
sio = socketio.Server()
app = Flask(__name__)

@sio.on('telemetry')
def telemetry(sid, data):
    #print("telemetry success!")

    if data:
        # Current telemetry
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])

        # Current frame from camera
        image = img.imread(BytesIO(base64.b64decode(data["image"])), format='jpeg')

        # Prediction of steering angle
        steering_angle = model.predict(image)

        # Empirical calculation of throttle
        #speed_target = 25 - abs(steering_angle) / 0.4 * 10
        #throttle = 0.2 - abs(steering_angle) / 0.4 * 0.15
        #throttle = (speed_target - speed) * 0.1
        throttle = 1.0 - steering_angle**2 - (speed/25)**2

        print("\rModel prediction -> (steering angle: {:.4f}, throttle: {:.4f})".format(steering_angle, throttle), end="")

        send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, env):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    config = configure()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = Model(config, device)
    model.load_model()  

    # Wrap application with engineio middleware
    app = socketio.Middleware(sio, app)

    # Deploy as eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
