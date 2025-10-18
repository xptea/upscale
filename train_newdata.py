import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from data_loader_multiscale import SRDatasetMultiScale
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

    # Load new dataset with multi-scale pairs
    new_dataset = SRDatasetMultiScale('newdata\\low', 'newdata\\high', transform=transform)

    print(f'New dataset size: {len(new_dataset)} images')

    train_loader = DataLoader(new_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    num_epochs = 15
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for low, high in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
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

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Performance: Loss over Epochs (New Dataset)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_newdata.png')
    print('Loss plot saved to loss_plot_newdata.png')

    torch.save(model.state_dict(), 'model_newdata.pth')
    print('Model saved to model_newdata.pth')

if __name__ == '__main__':
    main()