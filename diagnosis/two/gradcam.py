import os
import argparse
import torch
import torch.nn as nn
from model import get_model
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision
from dataset import LUMINA, MultiImagesDataset, get_fold
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import copy

def load_data(args):
    train_ds = []
    test_ds = []
    
    transform_train = v2.Compose([
        # v2.ToImage(), 
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.RandomHorizontalFlip(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    transform_test = v2.Compose([
        # v2.ToImage(), 
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    image_list, label_list = LUMINA(root = args.data_path, view = 2)
    train_image, train_label, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)
    train_ds = MultiImagesDataset(train_image, train_label, transform=transform_train)
    test_ds = MultiImagesDataset(test_image, test_label, transform=transform_test)
    print(f"There are {len(train_image)} training and {len(test_image)} testing images.")

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_dataloader, test_dataloader, test_image


def apply_gradcam2(model, target_layer, input_image, target_class=None):
    """
    Applies Grad-CAM to a given model and input image.

    Args:
        model (nn.Module): The PyTorch model.
        target_layer (nn.Module): The target convolutional layer for Grad-CAM.
        input_image (torch.Tensor): The preprocessed input image tensor.
        target_class (int, optional): The index of the target class. 
                                      If None, the predicted class is used.

    Returns:
        np.ndarray: The Grad-CAM heatmap.
    """
    # Store activations and gradients
    activations1 = None
    gradients1 = None

    def save_activations1(module, input, output):
        nonlocal activations1
        activations1 = output

    def save_gradients1(module, grad_input, grad_output):
        nonlocal gradients1
        gradients1 = grad_output[0]

    # Register hooks
    hook_handle_forward1 = target_layer[0].register_forward_hook(save_activations1)
    hook_handle_backward1 = target_layer[0].register_full_backward_hook(save_gradients1)
    
    # Store activations and gradients
    activations2 = None
    gradients2 = None

    def save_activations2(module, input, output):
        nonlocal activations2
        activations2 = output

    def save_gradients2(module, grad_input, grad_output):
        nonlocal gradients2
        gradients2 = grad_output[0]

    # Register hooks
    hook_handle_forward2 = target_layer[1].register_forward_hook(save_activations2)
    hook_handle_backward2 = target_layer[1].register_full_backward_hook(save_gradients2)

    # Perform forward pass
    output = model(input_image)

    # Get target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero gradients and perform backward pass for the target class
    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output, retain_graph=True)

    # Detach hooks
    hook_handle_forward1.remove()
    hook_handle_backward1.remove()
    
    hook_handle_forward2.remove()
    hook_handle_backward2.remove()

    # Compute Grad-CAM
    pooled_gradients1 = torch.mean(gradients1, dim=[0, 2, 3])
    for i in range(activations1.shape[1]):
        activations1[:, i, :, :] *= pooled_gradients1[i]

    heatmap1 = torch.sum(activations1, dim=1).squeeze()
    heatmap1 = nn.functional.relu(heatmap1)

    # Compute Grad-CAM
    pooled_gradients2 = torch.mean(gradients2, dim=[0, 2, 3])
    for i in range(activations2.shape[1]):
        activations2[:, i, :, :] *= pooled_gradients2[i]

    heatmap2 = torch.sum(activations2, dim=1).squeeze()
    heatmap2 = nn.functional.relu(heatmap2)
    
    heatmap_max = torch.max(torch.maximum(heatmap1, heatmap2))
    heatmap1 = heatmap1/heatmap_max # Normalize to [0, 1]
    heatmap2 = heatmap2/heatmap_max # Normalize to [0, 1]
    return [heatmap1.cpu().detach().numpy(), heatmap2.cpu().detach().numpy()]

class get_model2(nn.Module):
    def __init__(self, model):
        super(get_model2, self).__init__()
        self.backbones1 = copy.deepcopy(model.backbones)
        self.backbones2 = copy.deepcopy(model.backbones)
        self.classifier = copy.deepcopy(model.classifier)

    def forward(self, x:list):
        f = torch.cat([self.backbones1(x[0]), self.backbones2(x[1])], dim=1)
        return self.classifier(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MammoFL")
    parser.add_argument("--data-path", default="/dataset/Mammogram/LUMINA_PNG", type=str, help="dataset path")
    parser.add_argument("--model", default="efficientnet_b0", type=str, help="model name")
    parser.add_argument("-o", "--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("-r", "--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of train data loading workers")
    parser.add_argument("--input-size", default=512, type=int, help="input size")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.input_size))
    
    cam_root = os.path.join('./cam', args.model+'_'+str(args.input_size))
    if not os.path.exists(cam_root):
        os.makedirs(cam_root)
    if not os.path.exists(os.path.join(cam_root, 'Benign')):
        os.makedirs(os.path.join(cam_root, 'Benign'))
    if not os.path.exists(os.path.join(cam_root, 'Malign')):
        os.makedirs(os.path.join(cam_root, 'Malign'))
    
    device = torch.device(args.device)
    n_fold = 5
    model = get_model(name = args.model, num_classes = 1, view = 2)
    model.to(device)
    model.eval() 
    
    preprocess = v2.Compose([
        v2.ToImage(), 
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    for fold in range(n_fold):
        args.fold = fold
        _, _, test_image = load_data(args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        model2 = get_model2(model)
        # target_layer = model.backbones.features[-1]
        target_layer = [model2.backbones1.features[-1], model2.backbones2.features[-1]]
        for image_path in tqdm(test_image):
            image = [Image.open(i) for i in image_path]
            input_tensor = [preprocess(torchvision.tv_tensors.Image(i)).unsqueeze(0).to(device) for i in image] # Add batch dimension
            image = [i.point(lambda ii: ii * (1/256)).convert('RGB') for i in image]
            heatmap = apply_gradcam2(model2, target_layer, input_tensor)
            
            for i in range(2):        
                original_image_np = np.array(image[i].resize((args.input_size, args.input_size)))
                heatmap_resized = cv2.resize(heatmap[i], (original_image_np.shape[1], original_image_np.shape[0]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                overlayed_image = cv2.addWeighted(original_image_np, 0.6, heatmap_colored, 0.4, 0)
                
                cv2.imwrite(os.path.join(cam_root, os.path.split(os.path.split(image_path[i])[0])[-1], os.path.basename(image_path[i])), overlayed_image)
