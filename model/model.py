import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models

# === Create pretrained base + classifier ===
def create_model(model_name, num_classes=14):
    if model_name == 'vgg16':
        base = models.vgg16(pretrained=True)
        in_features = base.classifier[6].in_features
        base.classifier[6] = nn.Identity()
    elif model_name == 'vgg19':
        base = models.vgg19(pretrained=True)
        in_features = base.classifier[6].in_features
        base.classifier[6] = nn.Identity()
    elif model_name == 'efficientnet_b0':
        base = models.efficientnet_b0(pretrained=True)
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Identity()
    else:
        raise ValueError("Unsupported model")

    classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return base, classifier

# === Train and evaluate ===
def train_model(model, classifier, loaders, device, epochs=10, lr=1e-4):
    model.to(device)
    classifier.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []

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

    plot_metrics(train_losses, val_losses, val_accs)

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

def load_model(model_name, path, num_classes=14):
    model, classifier = create_model(model_name, num_classes)
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    return model, classifier

# === Voting Ensemble ===
def predict_ensemble(models, classifiers, dataloader, device):
    for m, c in zip(models, classifiers):
        m.eval()
        c.eval()

    all_preds = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs = [clf(model(x)) for model, clf in zip(models, classifiers)]
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            preds = torch.argmax(avg_output, dim=1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds)
