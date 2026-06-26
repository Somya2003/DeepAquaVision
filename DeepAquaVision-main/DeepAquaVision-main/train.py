import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from models import Generator, Discriminator


class UnderwaterDataset(Dataset):
    def __init__(self, low_folder, high_folder):
        self.low_images = sorted(os.listdir(low_folder))
        self.high_images = sorted(os.listdir(high_folder))
        self.low_folder = low_folder
        self.high_folder = high_folder

    def __getitem__(self, index):
        low_path = os.path.join(self.low_folder, self.low_images[index])
        high_path = os.path.join(self.high_folder, self.high_images[index])

        low_image = Image.open(low_path).convert("RGB")
        high_image = Image.open(high_path).convert("RGB")

        transform = transforms.ToTensor()
        return transform(low_image), transform(high_image)

    def __len__(self):
        return len(self.low_images)


def psnr(target, prediction):
    target = torch.clamp(target, 0.0, 1.0)
    prediction = torch.clamp(prediction, 0.0, 1.0)
    mse = F.mse_loss(target, prediction)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataset = UnderwaterDataset('low', 'high')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    num_epochs = 10

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        for i, (inputs, ground_truth) in enumerate(data_loader):
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            batch_size = inputs.size(0)

            # ---- Train Discriminator ----
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            real_predictions = discriminator(ground_truth)
            d_loss_real = criterion(real_predictions, real_labels)

            fake_images = generator(inputs)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_predictions = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_predictions, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ---- Train Generator ----
            optimizer_G.zero_grad()
            targets = torch.ones(batch_size, 1).to(device)
            fake_predictions = discriminator(fake_images)
            g_loss = criterion(fake_predictions, targets)
            g_loss.backward()
            optimizer_G.step()

            psnr_value = psnr(ground_truth, fake_images)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"D Loss: {d_loss.item():.4f} "
              f"G Loss: {g_loss.item():.4f} "
              f"PSNR: {psnr_value.item():.2f}")

        # Save checkpoint every epoch
        torch.save(generator.state_dict(), f"checkpoints/generator.pth")

    print("Training complete! Model saved to checkpoints/generator.pth")


if __name__ == "__main__":
    train()