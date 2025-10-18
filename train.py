import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from data_loader import SRDataset
from model import SRCNN
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = SRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset1 = SRDataset('dataset/train/low_res', 'dataset/train/high_res', transform=transform)
    train_dataset2 = SRDataset('dataset/Raw Data/low_res', 'dataset/Raw Data/high_res', transform=transform)
    train_dataset = ConcatDataset([train_dataset1, train_dataset2])

    val_dataset = SRDataset('dataset/val/low_res', 'dataset/val/high_res', transform=transform)

    print(f'Train dataset size: {len(train_dataset)} images')
    print(f'Val dataset size: {len(val_dataset)} images')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    num_epochs = 50
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for low, high in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Train'):
            low = low.to(device)
            high = high.to(device)
            pred = model(low)
            loss = loss_fn(pred, high)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low, high in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Val'):
                low = low.to(device)
                high = high.to(device)
                pred = model(low)
                loss = loss_fn(pred, high)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Performance: Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print('Loss plot saved to loss_plot.png')

    torch.save(model.state_dict(), 'model.pth')
    print('Model saved to model.pth')

if __name__ == '__main__':
    main()