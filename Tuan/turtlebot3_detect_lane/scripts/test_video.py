import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
from model import TwinLite as net
import cv2


def Run(model, img):
    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    img = img.cuda()
    with torch.no_grad():
        img_out = model(img)
    x0 = img_out[0]
    x1 = img_out[1]

    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    # img_rs[DA>100]=[255,0,0]
    img_rs[LL > 100] = [0, 0, 255]

    return img_rs


model = net.TwinLiteNet()
# TODO: If the model was trained with only one GPU, then comment the following line
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load("test_/model_0.pth"))
model.eval()

# Set start and end time in seconds
start = 0
end = start + 500
video_name = "pp_lane"

video = cv2.VideoCapture("videos/" + video_name + ".mp4")
video.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
fps = video.get(cv2.CAP_PROP_FPS)

if start is None or start < 0:
    start = 0

if end is None or end > video.get(cv2.CAP_PROP_FRAME_COUNT) / fps:
    end = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps

endFrame = end * fps

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("results/" + video_name + "-before.mp4", fourcc, fps, (640, 360))

while video.isOpened():
    ret, img = video.read()
    if ret and video.get(cv2.CAP_PROP_POS_FRAMES) < endFrame:
        img = Run(model, img)
        cv2.imshow("img", img)
        out.write(img)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
