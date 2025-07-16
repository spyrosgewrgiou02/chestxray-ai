import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# Load the model
class XRayMindCNN(torch.nn.Module):
    def __init__(self):
        super(XRayMindCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 14)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# Load model weights
device = torch.device("cpu")
model = XRayMindCNN().to(device)
model.load_state_dict(torch.load("xraymind_model.pth", map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

disease_labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

# Streamlit UI
st.title("🩻 ChestMNIST Disease Predictor")
st.write("Upload a chest X-ray (28×28 grayscale) and let the model tell you what it sees.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = (output > 0.5).int().numpy()[0]

    st.subheader("Model Diagnosis")
    for i, val in enumerate(predicted):
        status = "Present" if val == 1 else "Absent"
        st.write(f"- {disease_labels[i]}: {status}")

        st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 16px; color: gray;'>© Spyros Georgiou 2025</div>",
    unsafe_allow_html=True
)

