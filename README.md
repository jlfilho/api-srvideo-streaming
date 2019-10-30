# Starting the TensorFlow Serving server
tensorflow_model_server --model_base_path=/tf/joao/Workspace/api-video-streaming/model --rest_api_port=9000 --model_name=upscale2x

# Starting the Tensorflow serving server with Docker conteiner
docker run --runtime=nvidia -p 8500:8500 -p 8501:8501 --mount type=bind,source=/home/joao/Workspace/api-video-streaming/model,target=/models/upscale2x -e MODEL_NAME=upscale2x -t tensorflow/serving:latest-gpu &


# Enter in flask_server dir
# Startint flask server in development mode
export FLASK_ENV=development && flask run --host=0.0.0.0

# Enter in flask_server dir
# Startint flask server in development mode on port 6006
export FLASK_ENV=development && flask run --host=0.0.0.0 --port 6006

# Startint flask server in production mode on port 6006
gunicorn --threads 5 --workers 1 --bind 0.0.0.0:6006 app:app

# Sending a single image 
python3 resquest_to_flaskserver.py -i comic.png

# Sending a dash video 
python resquest_to_flaskserver_seg.py -v ../video1/demo1/
