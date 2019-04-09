from flask import render_template, jsonify, Response, request
from app import app
import random
from camera import VideoCamera


@app.route('/')

@app.route('/upload')
def upload_file2():
    return render_template('index.html')

@app.route('/uploaded', methods = ['GET','POST'])
def upload_file():
    if request.method=='POST':
        f = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        #model = ##model weights##
        img = image.load_img(f.filename)
        #preds = ##run inference##

        f.save(path)
        return render_template('uploaded.html', title='Success',predictions=preds_decoded, user_image=f.filename)

@app.route('/index')
def index():
    return render_template('index.html', title='Inference Generator')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/capture', methods=['GET','POST'])
def capture():
    if request.method == 'POST':
        return jsonify(request.form['file'])
    return render_template('capture.html', title='Upload a New Photo')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/record_status', methods=['POST'])
# def record_status():
#     global video_camera
#     if video_camera == None:
#         video_camera = VideoCamera()
#
#     json = request.get_json()
#
#     status = json['status']
#
#     if status == "true":
#         video_camera.start_record()
#         return jsonify(result="started")
#     else:
#         video_camera.stop_record()
#         return jsonify(result="stopped")
#
# def video_stream():
#     global video_camera
#     global global_frame
#
#     if video_camera == None:
#         video_camera = VideoCamera()
#
#     while True:
#         frame = video_camera.get_frame()
#
#         if frame != None:
#             global_frame = frame
#             yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         else:
#             yield (b'--frame\r\n'
#                             b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
#
# @app.route('/video_viewer')
# def video_viewer():
#     return Response(video_stream(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/capture')
# def video_feed():
#     return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')
