# YOLO Custom Object Detection (Web App)

A minimal web-based application for running object detection using YOLO models.
Upload a video through the browser and get back a result with bounding boxes and labels.

🚀 Features
* Simple HTML UI (index.html)
* Python backend (server.py) for inference
* Supports multiple YOLO models:
  * yolo11n.pt
  * yolov8n.pt
  * license_plate_detector.pt (custom-trained)
* Stores uploaded + processed files in uploads/
* Lightweight and easy to run locally

📁 Project Structure
<pre>
  - static/                  # CSS, JS, assets
  - uploads/                 # Uploaded & processed files
  - index.html               # Main UI
  - server.py                # Backend + YOLO inference
  - requirements.txt         # Python dependencies
  - *.pt                     # YOLO model weights
  - run.sh                   # Helper script
</pre>
  
🛠 Installation
<pre>
  git clone https://github.com/rupalimits/yolo-custom-obj-det
  
  cd yolo-custom-obj-det
  
  pip install -r requirements.txt
</pre>
▶️ Run the App
<pre>
  python server.py

  Then open your browser at: http://localhost:3000
</pre>

🖼 How It Works
- Open the UI
- Upload a video where you want to do the detection of vehicles, speed of vehicle, number plate of vechicle, and bounding box on vehicle/people.
- Backend loads a YOLO model and performs inference
- Processed image with detections appears in the UI
- Results saved to uploads/

I have also created a demo of the videos that I trained on my local, the demo can be visited at the following site - 
https://rupalimits.github.io/yolo-custom-obj-det/

Select the type of detection from the dropdown and run the video.

You can also select the number of options that is available in the navigation bar on the right side.

Play with it, and let me know the feedback!
