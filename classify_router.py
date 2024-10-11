from fastapi import APIRouter, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import io
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

router = APIRouter()

# arquitetura da rede neural
class ConvolutionalNetwork(LightningModule): 
    def __init__(self, class_names):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 4)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return F.log_softmax(X, dim=1)

class_names = ['vidro', 'metal', 'papel', 'plastico'] 
model = ConvolutionalNetwork(class_names)

model.load_state_dict(torch.load("modelo_ML_classificacao_residuos.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@router.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    
    return {"class": predicted_class}
