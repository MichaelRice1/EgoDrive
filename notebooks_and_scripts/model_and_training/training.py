from utils import AriaMultimodalTrainDataset, dict_transform, calculate_class_weights, load_data, train_single_phase, ModelWrapper
from EgoDriveMax import EgoDriveMultimodalTransformer
from torchvision import transforms
import torch
import numpy as np
import random
from torch.utils.data import DataLoader


class Train():

    def __init__(self, light = True, ablation=False):
        self.light = light
        self.ablation = ablation
        torch.manual_seed(13)
        np.random.seed(13)
        random.seed(13)
        torch.backends.cudnn.deterministic = True


    def build_data(self, ablation):
        dataset = AriaMultimodalTrainDataset(
            root_dir='/Users/michaelrice/Desktop/data backup/224_reduced_obj',
            transform=transforms.Compose([
                transforms.Lambda(dict_transform)
            ])
        )

        loss_weight = calculate_class_weights(dataset)
        loss_weight = loss_weight 
        print(f"Loss weights: {loss_weight}")

        train_set, val_set, test_set, class_weights = load_data(dataset)
        print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}, Test set size: {len(test_set)}")

        if self.ablation:
            ablation_train = []
            ablation_val = []
            ablation_test = []


            for t in train_set:
                tc = t.copy()
                tc.pop('frames')
                ablation_train.append(tc)

            for v in val_set:
                vc = v.copy()
                vc.pop('frames')
                ablation_val.append(vc)

            for s in test_set:
                sc = s.copy()
                sc.pop('frames')
                ablation_test.append(sc)
            
            return ablation_train, ablation_val, ablation_test, class_weights
        
        else:
            return train_set, val_set, test_set, class_weights
        
    def train(self):

        train,val,test,class_weights = self.build_data(ablation=self.ablation)


        wrapped_model = ModelWrapper(
            EgoDriveMultimodalTransformer(
            dim_feat=32,
            dropout=0.1,
            num_classes=6,
            num_frames=32,
            transformer_depth=1,
            transformer_heads=2,
            light = self.light
        )
    )



        train_single_phase(
            model=wrapped_model,
            train_loader=DataLoader(train, batch_size=8, shuffle=True),
            val_loader=DataLoader(val, batch_size=8, shuffle=False),
            test_loader=DataLoader(test, batch_size=8, shuffle=False),
            phase='multimodal',
            epochs=30,
            project_name="egodrive-multimodal",
            run_name="full_light_model",
            class_weights=class_weights
        )



    
if __name__ == "__main__":
    trainer = Train(light=True, ablation=False)
    trainer.train()






    