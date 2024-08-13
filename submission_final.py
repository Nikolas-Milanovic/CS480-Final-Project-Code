import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CombinedNN(nn.Module):
    def __init__(self):
        super(CombinedNN, self).__init__()
        
        # CNN for image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc_image = nn.Linear(128 * 16 * 16, 256) 
        
        # FCN for biological features processing
        self.fc_bio_1 = nn.Linear(163, 256)
        self.fc_bio_2 = nn.Linear(256, 128)
        
        # Combined layers
        self.fc_combined_1 = nn.Linear(256 + 128, 128)
        self.fc_combined_2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 6)  # Output layer for 6 target values

    def forward(self, X_images, X_bio):
        # CNN pathway for images
        x1 = self.pool(F.relu(self.conv1(X_images)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, 128 * 16 * 16)  # Flatten
        x1 = F.relu(self.fc_image(x1))
        
        # FCN pathway for biological features
        x2 = F.relu(self.fc_bio_1(X_bio))
        x2 = F.relu(self.fc_bio_2(x2))
        
        # Concatenate both pathways
        x_combined = torch.cat((x1, x2), dim=1)
        
        # Fully connected layers after concatenation
        x_combined = F.relu(self.fc_combined_1(x_combined))
        x_combined = F.relu(self.fc_combined_2(x_combined))
        output = self.fc_out(x_combined)
        
        return output


model = CombinedNN()


def train_epoch(data_loader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    num_batches = len(data_loader)

    for batch, (X_images, X_bio, labels) in enumerate(data_loader):
        #print(batch)
        X_images = X_images.to(device)
        X_bio = X_bio.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(X_images, X_bio)
        loss = loss_fn(outputs, labels)

        # Compute Loss
        train_loss += loss.item()

        # Backpropagation and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    return average_train_loss


def train(model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_train_loss = []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(data_loader, model, criterion, optimizer)
        total_train_loss += [train_loss]
        print(f'Epoch #{epoch+1}: train loss {train_loss:.3f}\t')
    return train_loss


class BioImageDataset(Dataset):
    def __init__(self, image_folder, bio_data_path, is_training=True):
        bio_data = pd.read_csv(bio_data_path)
        self.is_training = is_training
        if is_training:
            self.bio_features = torch.tensor(bio_data.iloc[:, 1:-6].values, dtype=torch.float32)
            self.targets = torch.tensor(bio_data.iloc[:, -6:].values, dtype=torch.float32)
        else:
            self.bio_features = torch.tensor(bio_data.iloc[:, 1:].values, dtype=torch.float32)
            self.targets = None

        self.image_folder = image_folder
        self.image_ids = bio_data['id'].values.astype(str)
        self.image_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, f'{self.image_ids[idx]}.jpeg')
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)
        bio_features = self.bio_features[idx]

        if self.is_training:
            return image, bio_features, self.targets[idx]
        return image, bio_features, self.image_ids[idx]




#####################################TRAINING#####################################
NUM_EPOCHS  = 10

image_folder = './data/train_images'
bio_data_path = './data/train.csv'
dataset = BioImageDataset(image_folder=image_folder, bio_data_path=bio_data_path, is_training=True)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = CombinedNN().to(device)
train(model)


#####################################TESTING#####################################
def test_and_save_predictions(data_loader, model, device, output_csv_path):
    model.eval()  
    predictions = []
    ids = []

    with torch.no_grad(): 
        for batch, (X_images, X_bio, data_ids) in enumerate(data_loader):
            X_images = X_images.to(device)
            X_bio = X_bio.to(device)

            outputs = model(X_images, X_bio)
            predictions.append(outputs.cpu().numpy())
            ids.extend(data_ids)  


    predictions = np.vstack(predictions)    
    results_df = pd.DataFrame(predictions, columns=['X4', 'X11', 'X18', 'X26', 'X50', 'X3112'])
    results_df.insert(0, 'id', ids) 

    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")




output_csv_path = './submission.csv'
image_folder = './data/test_images'
bio_data_path = './data/test.csv'
dataset = BioImageDataset(image_folder=image_folder, bio_data_path=bio_data_path, is_training=False)
test_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

test_and_save_predictions(test_data_loader, model, device, output_csv_path)







