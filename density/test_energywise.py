import os
import argparse
import torch
import torch.nn as nn
from model import get_model
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from dataset import LUMINA_Density, MultiImagesDataset, get_fold
from train import test_fn
import pandas as pd

def get_energy(root='/dataset/Mammogram/LUMINA_PNG'):
    low = {'Benign':[], 'Malign':[]}
    high = {'Benign':[], 'Malign':[]}
    for label in ['Benign', 'Malign']:
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        for i in range(0, len(df), 2):
            if df['MANUFACTOR'][i] in ['IMS GIOTTO S.p.A.', 'IMS s.r.l.', 'GE MEDICAL SYSTEMS']:
                if df['ID'][i] not in high[label]:
                    high[label].append(int(df['ID'][i]))
            else:
                if df['ID'][i] not in low[label]:
                    low[label].append(int(df['ID'][i]))
    return low, high

def load_data(args):
    train_ds = []
    test_ds = []
    
    transform_train = v2.Compose([
        # v2.ToImage(), 
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(),
        v2.Grayscale(num_output_channels=3),
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
    
    image_list, label_list = LUMINA_Density(root = args.data_path, view = 2)
    train_image, train_label, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)
    train_ds = MultiImagesDataset(train_image, train_label, transform=transform_train)
    test_ds = MultiImagesDataset(test_image, test_label, transform=transform_test)
    print(f"USHI has {len(train_image)} training and {len(test_image)} testing images.")

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return train_dataloader, test_dataloader, test_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MammoFL")
    parser.add_argument("--data-path", default="/dataset/Mammogram/LUMINA_PNG", type=str, help="dataset path")
    parser.add_argument("--model", default="swin_t", type=str, help="model name")
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
    model = get_model(name = args.model, num_classes = 4)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    results = {'loss':[], 'acc': [], 'auc': [], 'precision': [], 'recall': [], 'f1': [], 'specificity':[]}
    y_label = []
    y_predit = []
    x_name = []
    low, high = get_energy(args.data_path)
    for fold in range(n_fold):
        args.fold = fold
        _, test_dataloader, test_image = load_data(args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'fold'+str(args.fold), args.resume), map_location='cpu', weights_only=True))
        
        epoch_log, epoch_y = test_fn(test_dataloader, model, loss_fn, device)
        for metrics in ['loss', 'acc', 'auc']:
            results[metrics].append(epoch_log[metrics])
        y_output = [y.argmax() for y in epoch_y['pred']]
        
        x_name += test_image
        y_label += epoch_y['true']
        y_predit += epoch_y['pred']
    
    energy_list = []
    for i in range(len(x_name)):
        if 'Benign' in x_name[i][0]:
            if int(os.path.basename(x_name[i][0]).replace('L_MLO.png', '').replace('L_CC.png', '').replace('R_MLO.png', '').replace('R_CC.png', '')) in low['Benign']:
                energy_list.append(0)
            else:
                energy_list.append(1)
        else:
            if int(os.path.basename(x_name[i][0]).replace('L_MLO.png', '').replace('L_CC.png', '').replace('R_MLO.png', '').replace('R_CC.png', '')) in low['Malign']:
                energy_list.append(0)
            else:
                energy_list.append(1)
    y_output = [y.argmax() for y in y_predit]
    y_label_low = []
    y_label_high = []
    y_predit_low = []
    y_predit_high = []
    y_output_low = []
    y_output_high = []
    for i in range(len(energy_list)):
        if energy_list[i] == 0:
            y_label_low.append(y_label[i])
            y_predit_low.append(y_predit[i])
            y_output_low.append(y_output[i])
        else:
            y_label_high.append(y_label[i])
            y_predit_high.append(y_predit[i])
            y_output_high.append(y_output[i])
    low_result = {'acc':0, 'auc':0, 'f1':0}     
    high_result = {'acc':0, 'auc':0, 'f1':0}    
    global_result = {'acc':0, 'auc':0, 'f1':0}        
    low_result['acc'] = accuracy_score(y_label_low, y_output_low)
    high_result['acc'] = accuracy_score(y_label_high, y_output_high)
    global_result['acc'] = accuracy_score(y_label, y_output)
    low_result['auc'] = roc_auc_score(y_label_low, y_predit_low, multi_class='ovr')
    high_result['auc'] = roc_auc_score(y_label_high, y_predit_high, multi_class='ovr')
    global_result['auc'] = roc_auc_score(y_label, y_predit, multi_class='ovr')
    low_result['f1'] = f1_score(y_label_low, y_output_low, average='macro')
    high_result['f1'] = f1_score(y_label_high, y_output_high, average='macro')
    global_result['f1'] = f1_score(y_label, y_output, average='macro')
    
    st = "Low:"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" {metrics} {low_result[metrics]:.4f}"
    st += " High:"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" {metrics} {high_result[metrics]:.4f}"
    st += " Global:"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" {metrics} {global_result[metrics]:.4f}"
    print(st)
    st = "Latex:"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" & {low_result[metrics]*100:.2f}"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" & {high_result[metrics]*100:.2f}"
    for metrics in ['acc', 'auc', 'f1']:
        st += f" & {global_result[metrics]*100:.2f}"
    print(st)