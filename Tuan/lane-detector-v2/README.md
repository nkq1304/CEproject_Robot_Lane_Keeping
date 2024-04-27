# Requirement
- Download PyTorch from [here](https://pytorch.org/get-started/locally/) (GPU is recommended for better performance).
- If your GPU not support CUDA, you can choose these options:
    - (Not recommend) Install PyTorch without CUDA:
    ```
    pip install torch torchvision torchaudio
    ```
    - Demo without LaneDetector:
    ```
    python test_lane_video.py
    ```

- See `requirements.txt` for additional dependencies and version requirements:

```
pip install -r requirements.txt
```
# Usage
## Demo with LaneDetector

```
python test_video.py
```

## Demo without LaneDetector

```
python test_lane_video.py
```

## For Turtlebot:
- Make sure to uncomment `image_publisher.py` before running the following command:
```
python test_turtlebot.py
```