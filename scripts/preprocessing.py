import torch
from os import path
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

# load efficientnet-b0 - ~5.3 million params
# model export docs - https://docs.pytorch.org/executorch/stable/using-executorch-export.html
project_root = path.dirname(path.dirname(path.abspath(__file__))) + "/data/" # for writing files to data/
model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open(project_root + "model.pte", "wb") as f:
    f.write(et_program.buffer)

# process input image into tensor - will be used by libtorch for inference
image_path = project_root + "test.jpeg"
image = Image.open(image_path).convert("RGB") # read as rgb
totensor = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor()])
image_tensor = totensor(image).unsqueeze(0) # shape: (1, 3, 224, 224)
torch.save(image_tensor, project_root+"image_tensor.pt")
