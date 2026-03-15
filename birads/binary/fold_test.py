import os
import argparse
import torch
import torch.nn as nn
from model import get_model
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, confusion_matrix
from train import load_data, test_fn
import matplotlib.pyplot as plt
import numpy as np

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
    model = get_model(name = args.model, num_classes = 1, view = 2)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    results = {'loss':[], 'acc': [], 'auc': [], 'precision': [], 'recall': [], 'f1': [], 'specificity':[]}
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        
        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
        for metrics in ['loss', 'acc', 'auc']:
            results[metrics].append(epoch_log[metrics])
        y_output = [y>0.5 for y in epoch_y['pred']]
        results['precision'].append(precision_score(epoch_y['true'], y_output))
        results['recall'].append(recall_score(epoch_y['true'], y_output))
        results['f1'].append(f1_score(epoch_y['true'], y_output))
        tn, fp, fn, tp = confusion_matrix(epoch_y['true'], y_output).ravel()
        results['specificity'].append(tn / (tn + fp))
        fpr, tpr, thresholds = roc_curve(epoch_y['true'], epoch_y['pred'])
        roc_auc = epoch_log['auc']
        
        # Interpolate TPRs
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} ROC (AUC={roc_auc:.4f})')
        
    
    for fold in range(n_fold):
        st = f"Fold {fold} test"
        for metrics in ['loss', 'acc', 'auc', 'precision', 'recall', 'f1', 'specificity']:
            st += f" {metrics} {results[metrics][fold]:.4f}"
        print(st)
    st = "Cross validation"
    for metrics in ['acc', 'auc', 'precision', 'recall', 'f1', 'specificity']:
        st += f" {metrics} {np.mean(results[metrics]):.4f}±{np.std(results[metrics]):.4f}"
    print(st)
    st = "Latex:"
    for metrics in ['acc', 'auc', 'precision', 'recall', 'f1', 'specificity']:
        st += f" & {np.mean(results[metrics])*100:.2f}$\\pm${np.std(results[metrics])*100:.2f}"
    print(st)
    
    # Plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(results['auc'])
    std_auc = np.std(results['auc'])
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=f'Mean ROC (AUC={mean_auc:.4f}±{std_auc:.4f})',
             lw=2, alpha=0.8)
    
    # Plot std deviation
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                     label='±1 std. dev.')
    
    # Add plot details
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance', alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(os.path.join(args.output_dir, "roc.pdf"), format="pdf", bbox_inches='tight')