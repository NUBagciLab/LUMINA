import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from model import get_model
from train import load_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patches as mpatches
    

def test_fn(dataloader, model, loss_fn, device):
    model.eval()
    input_to_classifier = {}
    # Hook function to capture input
    def hook_fn(module, input, output):
        input_to_classifier['classifier_input'] = input[0].detach()
    
    hook = model.classifier[-1].register_forward_hook(hook_fn)
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    y_all = []
    pred_all = []
    hidden_state = []
    with torch.no_grad(): 
        progress_bar = tqdm(dataloader, desc="Testing")
        for X, y in progress_bar:
            y_all.extend(y)
            X, y = X.to(device), y.to(device).unsqueeze(1)
            pred = model(X)
            pred_all.extend(torch.sigmoid(pred).cpu().numpy())
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct = ((pred>0) == (y>0.5)).type(torch.float).sum().item()
            total_correct += correct
            batch_count += 1
            sample_count += len(X)
            hidden_state.extend(input_to_classifier['classifier_input'].cpu().numpy())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")
    return {'true': y_all, 'pred': pred_all}, hidden_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MammoFL")
    parser.add_argument("--data-path", default="/dataset/Mammogram/LUMINA_PNG", type=str, help="dataset path")
    parser.add_argument("--model", default="efficientnet_b0", type=str, help="model name")
    parser.add_argument("-o", "--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("-r", "--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of train data loading workers")
    parser.add_argument("--input-size", default=224, type=int, help="input size")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.input_size))
        
    device = torch.device(args.device)
    n_fold = 5
    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    y_all = []
    hidden_state_all = []
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        
        epoch_y, hidden_state = test_fn(test_dataloader, model, loss_fn, device)
        y_output = [y>0.5 for y in epoch_y['pred']]
        y_all.extend(epoch_y['true'])
        hidden_state_all.extend(hidden_state)
    
    hidden_state_all = np.stack(hidden_state_all, axis=0)
    y_all = np.array([y.cpu().numpy() for y in y_all], dtype=int)
    X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(hidden_state_all)

    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_all, palette='tab10', legend='full', s=60)
    # Create custom legend
    plt.legend(handles=[
        mpatches.Patch(color='C0', label='Benign'),
        mpatches.Patch(color='C1', label='Malignant')
    ])
    plt.grid()
    # plt.show()
    plt.savefig(args.model+'_'+str(args.input_size)+"tsne.pdf", format="pdf", bbox_inches='tight')