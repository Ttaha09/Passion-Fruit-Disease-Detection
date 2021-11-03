import streamlit as st
from PIL import Image
import torch
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms
import base64
from io import BytesIO
############ Model Loading ######################
target2label = {0: 'background',1: 'fruit_woodiness',2: 'fruit_brownspot',3: 'fruit_healthy'}
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    return model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ = get_model().to(device)
PATH = 'Model/pass-OD1'
model_.load_state_dict(torch.load(PATH,map_location='cpu'))
model_.eval()

def decode_output(output):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()
def get_image_download_link(img,filename,text): ## Download Image
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}" style="color:red;text-decoration:none">{text} </a>'
    return href

####################################################
st.markdown("<h1 style='text-align: center; color: black;'>Passion Fruit Dissease Detection</h1>", unsafe_allow_html=True)
st.markdown('##')
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** streamlit, pytorch, opencv, numpy, base64
* **Competition:** [Makerere Passion Fruit Disease Detection Challenge](https://zindi.africa/competitions/makerere-passion-fruit-disease-detection-challenge).
* **Description:** Classify the disease status of a plant given an image of a passion fruit.""")
image = Image.open('Images/passion-fruit.png')
st.image(image, use_column_width=True)
st.markdown("""
    Passion fruit pests and diseases lead to reduced yields and decreased investment in farming over time. Most Ugandan farmers (including passion fruit farmers) are smallholder farmers from low-income households, and do not have sufficient information and means to combat these challenges. Without the required knowledge about the health of their crops, farmers cannot intervene promptly to avoid devastating losses.
In this challenge, you will classify the disease status of a plant given an image of a passion fruit. """)
st.markdown('If successful, this model will be deployed as part of a device to aid smallholder farmers in making a prompt diagnosis in their passion fruit crops.')


uploaded_file = st.file_uploader('Select an Image', ['png'])
w,h = 512,512
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    st.caption("""Your Image before Detection """)
    st.image(image1, use_column_width=True)
    img = np.array(image1.resize((w, h), resample=Image.BILINEAR))/255.
    img = preprocess_image(img)
    bbs, confs, labels = decode_output(model_(img.unsqueeze(0))[0])
    plt.imsave('Images/test.png',img.permute(1,2,0).numpy())
    image = cv2.imread('Images/test.png',cv2.IMREAD_UNCHANGED)
    for boxes_ in range(len(bbs)):
        if confs[boxes_]>=0.4:
            start_point =(bbs[boxes_][0],bbs[boxes_][1])
            end_point= (bbs[boxes_][2],bbs[boxes_][3])
            text_point =(bbs[boxes_][0],bbs[boxes_][3])
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0,0,255)
            color_t =(255,0,0)
            thickness = 1
            thickness_t = 2
            path=uploaded_file
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            image = cv2.putText(image, labels[boxes_], text_point, font, fontScale, color_t, thickness_t, cv2.LINE_AA)
    st.caption("Your Image after Detection")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    result = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    st.markdown(get_image_download_link(result,uploaded_file.name,'Download '+uploaded_file.name), unsafe_allow_html=True)