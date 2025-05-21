import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

# === Create pretrained base + classifier ===
def create_model(model_name, num_classes=14, pretrained=True):
    if model_name == 'vgg16':
        base = models.vgg16(pretrained=pretrained)
        in_features = base.classifier[6].in_features
        base.classifier[6] = nn.Identity()
    elif model_name == 'vgg19':
        base = models.vgg19(pretrained=pretrained)
        in_features = base.classifier[6].in_features
        base.classifier[6] = nn.Identity()
    elif model_name == 'efficientnet_b0':
        base = models.efficientnet_b0(pretrained=pretrained)
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Identity()
    else:
        raise ValueError("Unsupported model")

    classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LeakyReLU(0.1),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.1),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return base, classifier


# === Train and evaluate ===
def train_model(model, classifier, loaders, device, epochs=10, lr=1e-4, patience=3):
    model.to(device)
    classifier.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        classifier.train()
        running_loss, correct, total = 0, 0, 0

        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = classifier(model(x))
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, classifier, loaders['val'], device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = {
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model before returning
    if best_model:
        model.load_state_dict(best_model['model_state_dict'])
        classifier.load_state_dict(best_model['classifier_state_dict'])

    return train_losses, val_losses, val_accs


def evaluate(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = classifier(model(x))
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

def plot_metrics(train_losses, val_losses, val_accs):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.show()

def save_model(model, classifier, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict()
    }, path)

def load_model(model_name, path=None, num_classes=14, use_pretrained=True, load_classifier=True, strict=True):
    model, classifier = create_model(model_name, num_classes=num_classes, pretrained=use_pretrained)

    if path:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if load_classifier and 'classifier_state_dict' in checkpoint:
            try:
                classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=strict)
            except RuntimeError:
                print("Warning: classifier weights mismatch. Using randomly initialized classifier.")
        else:
            print("Using fresh classifier (not loaded from checkpoint).")
    else:
        print("Initialized model without loading checkpoint.")

    return model, classifier


def predict_image(model, classifier, image_tensor, device):
    model.eval()
    classifier.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = classifier(model(image_tensor))
        return torch.argmax(output, dim=1).item()
