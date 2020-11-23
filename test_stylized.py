import torch
from wct import stylization
from photo_wct import PhotoWCT


# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load("model.pth"))

stylization(
    stylization_module=p_wct,
    content_image_path="grid_m.png",
    style_image_path="fire.png",
    output_image_path="result.png",
    cuda=True,
)
