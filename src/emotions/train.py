# train.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data import SpacioTemporalDataset
from emotions.model import SpatioTemporalHeteroGNN


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        # Simple regression target: predict mean pupil size
        # You can modify this to use actual labels from your data
        target = (data['node'].x[:, 3] + data['node'].x[:, 4]) / 2  # avg of left+right pupil
        target_graph = torch.zeros(out.size(0), device=device)
        
        # Aggregate target per graph
        for i in range(out.size(0)):
            mask = data['node'].batch == i
            target_graph[i] = target[mask].mean()
        
        loss = F.mse_loss(out.squeeze(), target_graph)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            # Same target as training
            target = (data['node'].x[:, 3] + data['node'].x[:, 4]) / 2
            target_graph = torch.zeros(out.size(0), device=device)
            
            for i in range(out.size(0)):
                mask = data['node'].batch == i
                target_graph[i] = target[mask].mean()
            
            loss = F.mse_loss(out.squeeze(), target_graph)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    data_dir = "./data/processed/cog-load"
    dest_dir = "./results/SpatioTemporalHeteroGNN"
    os.makedirs(dest_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    std = SpacioTemporalDataset(
        root_dir=data_dir,
        file_list=["s_001.csv"],
        recursive=True,
        window_length=10,
        kt=2,
        ks=2,
    )
    print(f"Total graphs: {len(std)}")
    
    # Split dataset
    indices = list(range(len(std)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    
    train_dataset = [std[i] for i in train_idx]
    test_dataset = [std[i] for i in test_idx]
    
    print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = SpatioTemporalHeteroGNN(
        in_channels=5,      # time, x, y, pupil_left, pupil_right
        hidden_channels=64,
        out_channels=1      # predicting single value (e.g., cognitive load)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    num_epochs = 50
    best_test_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            dest_path = os.path.join(dest_dir, 'best_model.pt')
            torch.save(model.state_dict(), dest_path)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    print(f"\nTraining complete! Best test loss: {best_test_loss:.4f}")
    print(f"Model saved to '{dest_path}'")


if __name__ == "__main__":
    main()

