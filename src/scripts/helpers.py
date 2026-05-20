import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

from IPython.display import display
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def show_augmented_images(dataloader, num_images=8):
    class_names = {0: 'Artifact (0)', 1: 'Real (1)'}

    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    images = images[:num_images]
    labels = labels[:num_images]

    rows = 2
    cols = num_images // rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx, ax in enumerate(axes.flat):
        img = images[idx].numpy().transpose((1, 2, 0))

        img = std * img + mean

        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(class_names[labels[idx].item()], fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0

            all_preds = []
            all_labels = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                all_preds.append(preds.detach())
                all_labels.append(labels.detach())

            all_preds = torch.cat(all_preds).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='micro')

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Micro F1: {epoch_f1:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_f1'].append(epoch_f1)

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model_temp.pth')

    model.load_state_dict(torch.load('best_model_temp.pth'))
    return model, history


def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss dependency on Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'b-', label='Train Micro F1')
    plt.plot(epochs, history['val_f1'], 'r-', label='Validation Micro F1')
    plt.title('Micro F1-score dependency on Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_confusion_matrix(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0 (Real)', 'Class 1 (Artifact)'],
                yticklabels=['Class 0 (Real)', 'Class 1 (Artifact)'])
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.show()

    print("\n--- DETAILED ANALYSIS ---")
    print(f"Total images checked: {len(all_labels)}")
    print(f"True Negatives (TN): {tn} - Real photos that the model correctly identified as real.")
    print(f"False Positives (FP): {fp} - WARNING! Real photos that the model incorrectly identified as artifacts (fakes).")
    print(f"False Negatives (FN): {fn} - Generated photos (artifacts), which the model missed.")
    print(f"True Positives (TP): {tp} - Generated photos that the model successfully detected.")

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSpecificity (Ability to not block real photos): {specificity:.4f}")


def compare_models(models_dict, test_loader, device='cuda'):
    """
    Get accuracy, micro F1, specificity, sensitivity, and FP count for each model on the test set.
    models_dict: format {'model_1': model_1, 'model_2': model_2, ...}
    """
    results = []

    for name, model in models_dict.items():
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='micro')

        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append({
            'Model': name,
            'Micro F1': round(f1, 4),
            'Accuracy': round(acc, 4),
            'Specificity (TN Rate)': round(specificity, 4),
            'Sensitivity (TP Rate)': round(sensitivity, 4),
            'False Positives': fp
        })

    df_results = pd.DataFrame(results)
    
    df_results = df_results.sort_values(by='Micro F1', ascending=False).reset_index(drop=True)
    
    print("\nComparison Results (Test Set):")
    display(df_results)
    return df_results


def random_search(model_builder_fn, train_dataset, val_dataset, param_space, sampler, num_trials=5, device='cpu'):
    """
    model_builder_fn: function that returns a new instance of the model
    param_space: dictionary with lists of possible parameters
    sampler: sampler for the data loader
    num_trials: how many random combinations we want to test
    """
    best_f1 = 0.0
    best_params = None
    best_model_state = None

    print(f"Starting Random Search. Total trials: {num_trials}")

    for idx in range(num_trials):
        lr = random.choice(param_space['lr'])
        batch_size = random.choice(param_space['batch_size'])
        weight_decay = random.choice(param_space.get('weight_decay', [0.0]))

        current_params = {'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay}
        
        print(f"\n--- Experiment {idx+1}/{num_trials} | LR: {lr}, Batch Size: {batch_size}, Weight Decay: {weight_decay} ---")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        model = model_builder_fn().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        epochs_for_search = 3

        for epoch in range(epochs_for_search):
            model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True) 
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        current_f1 = f1_score(all_labels, all_preds, average='micro')
        print(f"Result of combination {idx+1}: Micro F1 = {current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_params = current_params
            best_model_state = model.state_dict()

    print("\n=========================================")
    print(f"🏆 (peremoga) Best parameters: {best_params} (F1: {best_f1:.4f})")

    return best_params