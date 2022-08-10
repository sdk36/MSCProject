import os
import random
import binvox_rw
import torch
import random

class ShapeNetDataset(torch.utils.data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader"""

    def __init__(self, root):
        """Set the path for Data.
        Args:
            root: image directory.
            transform: Tensor transformer.
        """
        self.root = root
        self.listdir = os.listdir(self.root)

    def __getitem__(self, index):
        selected_cat = self.listdir[index]
        objdir = os.listdir(os.path.join(self.root,selected_cat))
        selected_obj = objdir[random.randrange(len(objdir))]
        modeldir = os.path.join(self.root, selected_cat, selected_obj, 'models','model_normalized.solid.binvox')
        with open(modeldir, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        tmodel = torch.FloatTensor(model.data)
        return tmodel

    def __len__(self):
        return len(self.listdir)