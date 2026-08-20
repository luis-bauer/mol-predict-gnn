import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils.smiles import x_map
from torch_geometric.utils import from_smiles
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "FreeSolv/database.txt",
    sep=";",
    header=None,
    names=["id", "smiles", "iupac", "exp_value", "exp_unc", "mob_value", "calc_unc", "exp_ref", "calc_ref", "notes"],
    usecols=[1, 3],
    skiprows=3
)

data_list = []
for smiles, y_val in zip(df["smiles"], df["exp_value"]):
    data = from_smiles(smiles)
    if data is not None:
        data.y = torch.tensor([[y_val]], dtype=torch.float)
        data_list.append(data)

train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)



class GNNModel(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Ein Embedding pro kategorialem Atom-Feature (Atomtyp, Hybridisierung, ...)
        # +1 als Puffer falls ein Index außerhalb der bekannten Kategorien liegt
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(vals) + 1, hidden_channels)
            for vals in x_map.values()
        ])

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = x.long()

        # Summe der Embeddings über alle Feature-Spalten -> [num_nodes, hidden_channels]
        emb = 0
        for i, emb_layer in enumerate(self.embeddings):
            emb = emb + emb_layer(x[:, i])

        x = F.relu(emb)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

model = GNNModel(hidden_channels=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 200

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch.batch)
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
            preds = model(batch.x, batch.edge_index, batch.batch)

            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = torch.tensor(y_true).numpy()
    y_pred = torch.tensor(y_pred).numpy()
    r2 = r2_score(y_true, y_pred)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoche {epoch:03d} | Train Loss (MSE): {train_loss:.4f} | Val R²: {r2:.3f}")

model.eval()
y_test_true, y_test_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.batch)

        y_test_true.extend(batch.y.cpu().numpy())
        y_test_pred.extend(preds.cpu().numpy())

y_test_true = torch.tensor(y_test_true).numpy()
y_test_pred = torch.tensor(y_test_pred).numpy()
test_r2 = r2_score(y_test_true, y_test_pred)
print(f"\n---> Finaler Test R²-Score: {test_r2:.3f} <---")