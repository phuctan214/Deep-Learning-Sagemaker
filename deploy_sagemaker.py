import subprocess
subprocess.call(['pip', 'install', 'smdebug'])

import smdebug
import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import requests


def model_fn(model_dir):
    print("+"*100)
    for (root,dirs,files) in os.walk(model_dir):
        print(root,dirs,files)
    print("="*100)
    model = torch.load(os.path.join(model_dir, "dog_classification.pt"), map_location=torch.device('cpu'))
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object = transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction