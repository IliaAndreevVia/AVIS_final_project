import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

def imshow(loader):
    dataiter = iter(loader)
    images, _ = next(dataiter)
    img = torchvision.utils.make_grid(images)/2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize = (8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_images_with_predictions(loader, model, classes, num_images=100, device='cpu', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    model.to(device)
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    axs = axs.flatten()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= num_images:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            inputs = denormalize (inputs,mean,std)
            probs = nn.functional.softmax(outputs, dim=1)
            top_probs, top_preds = torch.topk(probs, 3, dim=1)

            image = inputs.cpu().squeeze().numpy().transpose((1, 2, 0))
            label = labels.cpu().squeeze().numpy()
            axs[i].imshow(image)
            title = f'gt: {classes[label]}\n'
            title += "\n".join([f'{classes[pred.item()]}: {prob:.2f}' for pred, prob in zip(top_preds[0], top_probs[0])])
            axs[i].set_title(title)
            axs[i].axis('off')
    plt.tight_layout()
    plt.show()