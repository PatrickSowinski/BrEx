#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response
from flask_cors import CORS
from camera import VideoCamera
import json

app = Flask(__name__)
CORS(app)
cam = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen(cam):
    while True:
        frame, cor_position, breatheCorrect = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def plot(cam):
    while True:
        plot = cam.getPlot()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + plot + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calculations')
def getCalculations():
	return {"calculations": cam.state}, 200

@app.route('/success')
def getSuccess():
	cam.cut()
	print("COLOR: " + str(cam.color))
	return {"success": cam.contour_found}, 200

@app.route('/demo_mode')
def getDebugMode():
	cam = VideoCamera(debug=True)
	return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plot')
def getPlotMode():
    return Response(plot(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True)
