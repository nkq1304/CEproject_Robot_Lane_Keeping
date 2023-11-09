import torch
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from model import TwinLite as net
import cv2


def Run(model, img):
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
    img_rs[LL > 100] = [255, 255, 255]
    img_rs[LL <= 100] = [0, 0, 0]

    return img_rs


model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load("pretrained/best.pth"))
model.eval()

# Set start and end time in seconds
start = 0
end = start + 150
video_name = "original"

video = cv2.VideoCapture("videos/" + video_name + ".mp4")
video.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
fps = video.get(cv2.CAP_PROP_FPS)

if start is None or start < 0:
    start = 0

if end is None or end > video.get(cv2.CAP_PROP_FRAME_COUNT) / fps:
    end = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps

endFrame = end * fps

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
original_out = cv2.VideoWriter(
    "results/" + video_name + "_original.mp4", fourcc, fps, (640, 360)
)
result_out = cv2.VideoWriter("results/" + video_name + ".mp4", fourcc, fps, (640, 360))

while video.isOpened():
    ret, img = video.read()
    if ret and video.get(cv2.CAP_PROP_POS_FRAMES) < endFrame:
        img = cv2.resize(img, (640, 360))
        cv2.imshow("org", img)
        original_out.write(img)
        img = Run(model, img)
        cv2.imshow("img", img)
        result_out.write(img)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
