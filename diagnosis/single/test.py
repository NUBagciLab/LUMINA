import os
import argparse
import json
import torch
import torch.nn as nn
from model import get_model
from sklearn.metrics import roc_curve
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
    parser.add_argument("-f", "--fold", default=0, type=int, help="fold id for cross testidation  (must be 0 to 4)")
    parser.add_argument("--input-size", default=224, type=int, help="input size")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.input_size), 'fold'+str(args.fold))
        
    device = torch.device(args.device)
    _, test_dataloader = load_data(args)

    model = get_model(name = args.model, num_classes = 1)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.resume), map_location='cpu', weights_only=True))
    
    epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
    print(f"Test loss {epoch_log['loss']:.4f} acc {epoch_log['acc']:.4f} auc {epoch_log['auc']:.4f}")
    plt.figure(figsize=(8,6))
    fpr, tpr, thresholds = roc_curve(epoch_y['true'], epoch_y['pred'])
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.savefig(os.path.join(args.output_dir, "roc_curve.pdf"), format="pdf", bbox_inches='tight')
    
    with open(os.path.join(args.output_dir, "log.json")) as json_file:
        log_hist = json.load(json_file)
    
    for metric in ['loss', 'acc']:
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(1, 1+len(log_hist['test_'+metric])), log_hist['train_'+metric], lw=2, label='Train')
        plt.plot(np.arange(1, 1+len(log_hist['test_'+metric])), log_hist['test_'+metric], lw=2, label='Test')
        plt.xlim([0, len(log_hist['test_'+metric])])
        #plt.ylim([0, 1])
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.output_dir, metric+".pdf"), format="pdf", bbox_inches='tight')