import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import torchvision
import numpy as np

def get_view(root='/dataset/Mammogram/LUMINA_PNG'):
    for label in ['Benign', 'Malign']:
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        files = os.listdir(os.path.join(root, label))
        view1 = []
        view2 = []
        view4 = []
        for i in range(0, len(df), 2):
            if df['BIRADS'][i] == '0' or df['BIRADS'][i] == '-' or df['BIRADS'][i+1] == '0' or df['BIRADS'][i+1] == '-':
                continue
            if df['RIGHT_OR_LEFT'][i] == 'BILATERAL':
                for file in ['L_MLO', 'L_CC', 'R_MLO', 'R_CC']:
                    view1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))
                
                for file in ['L', 'R']:
                    view2.append([os.path.join(root, label, df['ID'][i]+file+'_MLO.png'),
                                  os.path.join(root, label, df['ID'][i]+file+'_CC.png')])
                    
                view4.append([os.path.join(root, label, df['ID'][i]+file+'.png') for file in ['L_MLO', 'L_CC', 'R_MLO', 'R_CC']])
                
            elif df['RIGHT_OR_LEFT'][i] == 'LEFT':
                for file in ['L_MLO', 'L_CC']:
                    view1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))
                    
                view2.append([os.path.join(root, label, df['ID'][i]+'L_MLO.png'),
                              os.path.join(root, label, df['ID'][i]+'L_CC.png')])
                
                if df['ID'][i]+'R_CC.png' in files and df['ID'][i]+'R_MLO.png' in files:
                    view4.append([os.path.join(root, label, df['ID'][i]+file+'.png') for file in ['L_MLO', 'L_CC', 'R_MLO', 'R_CC']])
             
            elif df['RIGHT_OR_LEFT'][i] == 'RIGHT':
                for file in ['R_MLO', 'R_CC']:
                    view1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))
                    
                view2.append([os.path.join(root, label, df['ID'][i]+'R_MLO.png'),
                                os.path.join(root, label, df['ID'][i]+'R_CC.png')])
                
                if df['ID'][i]+'L_CC.png' in files and df['ID'][i]+'L_MLO.png' in files:
                    view4.append([os.path.join(root, label, df['ID'][i]+file+'.png') for file in ['L_MLO', 'L_CC', 'R_MLO', 'R_CC']])
            
        if label=='Benign':
            benign1 = view1.copy()
            benign2 = view2.copy()
            benign4 = view4.copy()
        else:
            malign1 = view1.copy()
            malign2 = view2.copy()
            malign4 = view4.copy()
    
    return benign1, malign1, benign2, malign2, benign4, malign4


def LUMINA(root='/dataset/Mammogram/LUMINA_PNG', view = 1):
    image_list = []
    label_list = []
    benign, malign, benign2, malign2, benign4, malign4 = get_view(root)
    
    if view == 1:
        image_list = benign+malign
        label_list = [0.0 for i in range(len(benign))]+[1.0 for i in range(len(malign))]
    elif view == 2:
        image_list = benign2+malign2
        label_list = [0.0 for i in range(len(benign2))]+[1.0 for i in range(len(malign2))]
    elif view == 4:
        image_list = benign4+malign4
        label_list = [0.0 for i in range(len(benign4))]+[1.0 for i in range(len(malign4))]
    else:
        raise ValueError(f"View should be 1, 2, or 4 but got {view}!")
    
    return image_list, label_list

def LUMINA_BIRADS(root='/dataset/Mammogram/LUMINA_PNG', classes = 2):
    if classes not in [2, 3, 7]:
        raise ValueError(f"Classes must be 2, 3, or 7 but get {classes}!")
    
    images = []
    labels = []
    for label in ['Benign', 'Malign']:
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        for i in range(0, len(df), 2):     
            if df['BIRADS'][i] == '-' or df['BIRADS'][i+1] == '-':
                continue                       
            if df['BIRADS'][i] != df['BIRADS'][i+1]:
                raise ValueError("BIRADS not match!")
            birads = df['BIRADS'][i]
            if classes != 7 and birads == '0':
                continue
            if '4' in birads:
                birads = '4' 
            birads = np.int64(birads)                   
            if classes == 2:
                birads = np.float32(birads>3)
            if classes == 3:
                birads = np.int64((birads-1)//2)
            if df['RIGHT_OR_LEFT'][i] == 'BILATERAL':             
                for file in ['L', 'R']:
                    images.append([os.path.join(root, label, df['ID'][i]+file+'_MLO.png'),
                                    os.path.join(root, label, df['ID'][i]+file+'_CC.png')])                          
                    labels.append(birads)               
                    
            elif df['RIGHT_OR_LEFT'][i] == 'LEFT':
                images.append([os.path.join(root, label, df['ID'][i]+'L_MLO.png'),
                                os.path.join(root, label, df['ID'][i]+'L_CC.png')])
                labels.append(birads)  
                         
            elif df['RIGHT_OR_LEFT'][i] == 'RIGHT':                    
                images.append([os.path.join(root, label, df['ID'][i]+'R_MLO.png'),
                                os.path.join(root, label, df['ID'][i]+'R_CC.png')])
                labels.append(birads)  
    return images, labels          

def LUMINA_Density(root='/dataset/Mammogram/LUMINA_PNG', view=2):
    image1 = []
    image2 = []
    label1 = []
    label2 = []
    for label in ['Benign', 'Malign']:
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        for i in range(0, len(df), 2):     
            if df['BREAST COMPOSITION'][i] == '-':
                if df['RIGHT_OR_LEFT'][i+1] == 'BILATERAL':
                    for file in ['L_CC', 'R_CC']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.dcm'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i+1])-1)
                        
                elif df['RIGHT_OR_LEFT'][i+1] == 'LEFT':
                    for file in ['L_CC']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i+1])-1)  
                        
                elif df['RIGHT_OR_LEFT'][i+1] == 'RIGHT':
                    for file in ['R_CC']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i+1])-1)           
                
            elif df['BREAST COMPOSITION'][i+1] == '-':
                if df['RIGHT_OR_LEFT'][i] == 'BILATERAL':
                    for file in ['L_MLO', 'R_MLO']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i])-1)
                        
                elif df['RIGHT_OR_LEFT'][i] == 'LEFT':
                    for file in ['L_MLO']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i])-1)
                             
                elif df['RIGHT_OR_LEFT'][i] == 'RIGHT':
                    for file in ['R_MLO']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i])-1)
                        
            else:                               
                if df['RIGHT_OR_LEFT'][i] == 'BILATERAL':
                    for file in ['L_MLO', 'L_CC', 'R_MLO', 'R_CC']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i])-1)               
                    for file in ['L', 'R']:
                        image2.append([os.path.join(root, label, df['ID'][i]+file+'_MLO.png'),
                                        os.path.join(root, label, df['ID'][i]+file+'_CC.png')])                          
                        label2.append(np.int64(df['BREAST COMPOSITION'][i])-1)               
                        
                elif df['RIGHT_OR_LEFT'][i] == 'LEFT':
                    for file in ['L_MLO', 'L_CC']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i])-1)         
                        
                    image2.append([os.path.join(root, label, df['ID'][i]+'L_MLO.png'),
                                    os.path.join(root, label, df['ID'][i]+'L_CC.png')])
                    label2.append(np.int64(df['BREAST COMPOSITION'][i])-1)  
                             
                elif df['RIGHT_OR_LEFT'][i] == 'RIGHT':
                    for file in ['R_MLO', 'R_CC']:
                        image1.append(os.path.join(root, label, df['ID'][i]+file+'.png'))       
                        label1.append(np.int64(df['BREAST COMPOSITION'][i])-1)                         
                    image2.append([os.path.join(root, label, df['ID'][i]+'R_MLO.png'),
                                    os.path.join(root, label, df['ID'][i]+'R_CC.png')])
                    label2.append(np.int64(df['BREAST COMPOSITION'][i])-1)  
                    
    if view == 1:
        return image1, label1
    elif view == 2:
        return image2, label2
    else:
        raise ValueError(f"View must be 1 or 2 but get {view}!")

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of paths to image files.
            labels (list or None): Optional labels corresponding to each file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
class MultiImagesDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of paths to image files.
            labels (list or None): Optional labels corresponding to each file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = [torchvision.tv_tensors.Image(Image.open(i)) for i in self.image_paths[idx]]
        if self.transform:
            image = self.transform(image)
            # image = [self.transform(i) for i in image]
        label = self.labels[idx]
        return image, label
    
def get_fold(image:list, label:list, n_splits = 5, fold = 0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    skf.get_n_splits(image, label)
    for i, (train_index, test_index) in enumerate(skf.split(image, label)):
        if i == fold:
            train_image = [image[j] for j in train_index]
            train_label = [label[j] for j in train_index]
            test_image = [image[j] for j in test_index]
            test_label = [label[j] for j in test_index]
            return train_image, train_label, test_image, test_label

if __name__ == "__main__":
    image_list, label_list = LUMINA(view = 2)
    transforms = v2.Compose([
        # v2.ToImage(), 
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    ds = MultiImagesDataset(image_list, label_list, transform = transforms)
    