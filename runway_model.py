from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os

from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import runway
from runway.data_types import *


@runway.setup(options={"checkpoint_dir" : file(is_directory=True)})
def setup(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = opts["checkpoint_dir"]
    print(ckpt_dir)
    config = ckpt_dir + "/" + "yolov3.cfg"
    weights = ckpt_dir + "/" + "yolov3.weights"
    classes_path = ckpt_dir + "/" + "coco.names"
    
    model = Darknet(config, img_size=416).to(device)
    print("Using Darknet Weights")
    model.load_darknet_weights(weights)
    model.eval()

    classes = load_classes(classes_path)

    return {"model" : model,
            "classes" : classes }

 

command_inputs = {"input_image" : image}
command_outputs =  {
    'bboxes': array(image_bounding_box),
    'classes': array(text),
    'scores': array(number)
}
@runway.command("detect_objects", inputs=command_inputs, outputs=command_outputs, description="Detect Images in an image")
def detect_objects(model, inputs):    
    im = inputs["input_image"]
    input_img = np.array(im.resize((416, 416), Image.BICUBIC))
    input_img = transforms.ToTensor()(input_img)
    input_img = input_img.unsqueeze(0)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    print("\nPerforming object detection:")
    input_imgs = Variable(input_img.type(Tensor))
  
    # Get detections
    with torch.no_grad():
        detections = model["model"](input_imgs)
        detections = non_max_suppression(detections, 0.8, 0.4)
    
    bboxes = []
    class_labels = []
    scores = []
    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections[0], 416)
        
        unique_labels = detections[:, -1].cpu().numpy()
        class_preds = [model["classes"][int(i)] for i in unique_labels]
        
        op = detections.cpu().numpy()
        
       
    for i in range(len(op)):
        bboxes.append(op[i,0:4])
        scores.append(op[i, 5])
    
    return dict(bboxes=bboxes, classes=class_preds, scores=scores)

if __name__ == "__main__":
    runway.run(model_options={"checkpoint_dir" : "./checkpoint"})