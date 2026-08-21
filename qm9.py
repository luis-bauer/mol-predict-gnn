import pandas as pd
import numpy as np
from torch_geometric.utils import from_smiles
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils.smiles import x_map
from torch_geometric.utils.smiles import e_map
import torch.nn.functional as F
from sklearn.metrics import r2_score
import copy
from tqdm import tqdm
import os

df = pd.read_csv("qm9.csv")
target_cols = df.loc[:, "mu":].columns.tolist()
#Scales the values so that features with larger numbers do not get interpreted as more important
scaler = StandardScaler()
df[target_cols] = scaler.fit_transform(df[target_cols])

processed_file = "qm9_processed.pt"

if os.path.exists(processed_file):
    print("Load Dataset from hard drive...")
    data_list = torch.load(processed_file, weights_only=False)
else:
    print("Process SMILES Strings...")
    #Only append if the smiles string is valid
    data_list = []
    for smiles, y_row in tqdm(
        zip(df["smiles"], df[target_cols].values), total=len(df)
    ):
        data = from_smiles(smiles)
        if data is not None:
            data.y = torch.tensor(y_row, dtype=torch.float).unsqueeze(0)
            data_list.append(data)

    torch.save(data_list, processed_file)
    print("Gespeichert!")
# Split data into portions: 80% train, 10% validation, 10% test
train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
val_loader = DataLoader(val_data, batch_size=512, shuffle=False)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

class GNNModel(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Builds an embedding layer for each atom feature
        # x_map provides the number of values/categories that each feature can take on
        # +1 as a buffer in case that an index lies outside the known categories
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(vals) + 1, hidden_channels)
            for vals in x_map.values()
        ])
        # Same as before for edge-features
        self.edge_embeddings = nn.ModuleList([
            nn.Embedding(len(vals) +1, hidden_channels)
            for vals in e_map.values()
        ])
        h_c = hidden_channels
        self.conv1 = GATv2Conv(in_channels=h_c, out_channels=h_c, edge_dim=h_c)
        self.conv2 = GATv2Conv(in_channels=h_c, out_channels=h_c, edge_dim=h_c)
        self.conv3 = GATv2Conv(in_channels=h_c, out_channels=h_c, edge_dim=h_c)
        self.conv4 = GATv2Conv(in_channels=h_c, out_channels=h_c, edge_dim=h_c)
        self.conv5 = GATv2Conv(in_channels=h_c, out_channels=h_c, edge_dim=h_c)
        self.fc1 = nn.Linear(h_c, h_c)
        self.fc2 = nn.Linear(h_c, 16)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.long()
        # Picks the atom features that correspond to x from the embedding layer and adds them together
        # Combines all features into a single vector for each atom
        emb = 0
        for i, emb_layer in enumerate(self.embeddings):
            emb = emb + emb_layer(x[:, i])
        # Picks the edge features that correspond to edge_attr from the embedding layer and adds them together
        edge_attr = edge_attr.long()
        edge_emb = 0
        for i, emb_layer in enumerate(self.edge_embeddings):
            edge_emb = edge_emb + emb_layer(edge_attr[:, i])

        # Skip connections (e.g. + h1) allow the network to learn more stably
        h1 = F.relu(self.conv1(emb, edge_index, edge_attr=edge_emb))
        h2 = F.relu(self.conv2(h1, edge_index, edge_attr=edge_emb)) + h1
        h3 = F.relu(self.conv3(h2, edge_index, edge_attr=edge_emb)) + h2
        h4 = F.relu(self.conv4(h3, edge_index, edge_attr=edge_emb)) + h3
        x = F.relu(self.conv5(h4, edge_index, edge_attr=edge_emb)) + h4
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

model = GNNModel(hidden_channels=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epoch = 0
best_epoch = 0
highest_r2 = -float("inf")
# Number of epochs required without accuracy improvement to trigger early stopping
lookback = 15
history = lookback
max_epochs = 2000
best_model_weights = None

while history > 0 and epoch < max_epochs:
    epoch += 1
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(preds, batch.y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch.num_graphs

    train_loss = total_train_loss / len(train_loader.dataset)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    r2 = r2_score(np.vstack(y_true), np.vstack(y_pred))

    if r2 > highest_r2:
        highest_r2 = r2
        best_epoch = epoch
        history = lookback
        # Copy weights of model with the currently best accuracy
        best_model_weights = copy.deepcopy(model.state_dict())
    else:
        history -= 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoche {epoch:03d} | Train Loss (MSE): {train_loss:.4f} | Val R²: {r2:.3f}")


print(f"Optimal epoch is {best_epoch} with Val R²: {highest_r2}")
# Load model with the highest validation accuracy
model.load_state_dict(best_model_weights)
model.eval()
y_test_true, y_test_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.batch)

        y_test_true.extend(batch.y.cpu().numpy())
        y_test_pred.extend(preds.cpu().numpy())
# Display test accuracy of the model with the highest validation accuracy
test_r2 = r2_score(np.vstack(y_test_true), np.vstack(y_test_pred))
print(f"\n---> Final Test R²-Score: {test_r2:.3f} <---")