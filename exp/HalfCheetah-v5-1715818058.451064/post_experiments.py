import torch
import os

path = './exp/'
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print(dir_list)

model_path = './exp/PostExperiments/model.py'
# Load saved checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Load best actor weights
actor.load_state_dict(checkpoint['best_actor'])