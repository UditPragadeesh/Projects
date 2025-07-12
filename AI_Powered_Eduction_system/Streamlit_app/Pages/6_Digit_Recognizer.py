import streamlit as st
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

st.title('Digit Recognizer')

image = st.file_uploader('Upload Image',type=['jpg','jpeg','png'])

transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

class Net(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.Feature_extractor = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1,stride=1),
            nn.ELU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),
            nn.ELU(),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(nn.Linear(64*28*28,128),nn.Linear(128,num_classes))
    def forward(self,x):
        x = self.Feature_extractor(x)
        x = self.classifier(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=Net(num_classes=10)
net.load_state_dict(torch.load(r'Models\digit_model_weights.pth',map_location=device))
net.to(device)
net.eval()

if 'pred_bt' not in st.session_state:
    st.session_state.pred_bt = False
if st.button('Find'):
    st.session_state.pred_bt = True

if st.session_state.pred_bt:
    try:
        image = Image.open(image).convert('RGB')
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        output = net(image)
        _,pred =torch.max(output,1)
        st.markdown(f"The digit is:\n <h1 style='font-size:32px;'>{int(pred[0])}</h1>",unsafe_allow_html=True)
        if st.button('OK',key='ok'):
                 st.session_state.pred_bt=False

    except AttributeError:
        st.error('Upload File')
        if st.button('OK',key='ok'):
                 st.session_state.pred_bt=False
