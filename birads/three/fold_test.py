import os
import argparse
import torch
import torch.nn as nn
from model import get_model
from train import load_data, test_fn
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
    parser.add_argument("-c", "--classes", default=3, type=int, help="class (must be 3 or 7)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model+'_'+str(args.input_size)+'_'+str(args.classes))
        
    device = torch.device(args.device)
    n_fold = 5
    model = get_model(name = args.model, num_classes = args.classes)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    results = {'loss':[], 'acc': [], 'auc': [], 'f1': []}
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader = load_data(args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        
        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
        for metrics in ['loss', 'acc', 'auc', 'f1']:
            results[metrics].append(epoch_log[metrics])
    
    for fold in range(n_fold):
        st = f"Fold {fold} test"
        for metrics in ['loss', 'acc', 'auc', 'f1']:
            st += f" {metrics} {results[metrics][fold]:.4f}"
        print(st)
    st = "Cross validation"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" {metrics} {np.mean(results[metrics]):.4f}±{np.std(results[metrics]):.4f}"
    print(st)
    st = "Latex:"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" & {np.mean(results[metrics])*100:.2f}$\\pm${np.std(results[metrics])*100:.2f}"
    print(st)