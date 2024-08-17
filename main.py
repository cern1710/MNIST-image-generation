import model
import data_loader
from torch import optim
import torch

def train_VAE(model, train_loader, num_epochs, beta=1.0):
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss, total_rec_loss, total_reg_loss, total_bpd = 0, 0, 0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            x, _ = batch

            # Forward pass
            L_rec, L_reg, bpd = model(x)
            loss = L_rec + L_reg

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_rec_loss += L_rec.item()
            total_reg_loss += L_reg.item()
            total_bpd += bpd.item()

        avg_loss = total_loss / len(train_loader.dataset)
        avg_rec_loss = total_rec_loss / len(train_loader.dataset)
        avg_reg_loss = total_reg_loss / len(train_loader.dataset)
        avg_bpd = total_bpd / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {avg_loss:.4f}, "
              f"Reconstruction Loss: {avg_rec_loss:.4f}, "
              f"KL Divergence: {avg_reg_loss:.4f}, "
              f"BPD: {avg_bpd:.4f}")

if __name__ == "__main__":
    batch_size = 128
    num_filters = 32
    z_dim = 20
    learning_rate = 1e-3
    num_epochs = 10

    torch.manual_seed(42)

    train_loader, val_loader, test_loader = data_loader.load_mnist(batch_size=batch_size)
    VAE_model = model.VAE(num_filters, z_dim, learning_rate)
    train_VAE(VAE_model, train_loader, num_epochs)