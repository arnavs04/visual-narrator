"""
Contains various utility functions for PyTorch model building, training and saving.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import math
import random
import spacy

spacy_eng = spacy.load("en_core_web_sm")


def save_model(model: nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def tokenize(text):
    return [token.text for token in spacy_eng.tokenizer(text)]

def is_torch_available():
    """
    Checks if the PyTorch library is available in the current environment.

    This function attempts to import the PyTorch library and returns True if the import is successful,
    indicating that PyTorch is available. If the import fails (ImportError is raised), it returns False.

    Returns:
        bool: True if PyTorch is available, False otherwise.
    """
    try:
        import torch
        return True
    except ImportError:
        return False

    
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def count_parameters(model: nn.Module):
    """
    Calculate the total number of trainable parameters in a PyTorch model.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters in the model.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super(MyModel, self).__init__()
        ...         self.linear = nn.Linear(10, 5)
        ...         self.conv = nn.Conv2d(3, 6, 3)
        ...     def forward(self, x):
        ...         return self.conv(self.linear(x))
        >>> model = MyModel()
        >>> count_parameters(model)
        66
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    # To Evaluation Mode
    model.eval()
    
    # First Image
    test_img1 = transform(Image.open("visual-narrator/data/flickr8k/images/1001773457_577c3a7d70.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: A black dog and a spotted dog are fighting")
    print("Example 1 OUTPUT: " + " ".join(model.caption_image(test_img1.to(device), dataset.vocab)))
    print("\n")
    
    # Second Image
    test_img2 = transform(Image.open("visual-narrator/data/flickr8k/images/102351840_323e3de834.jpg").convert("RGB")).unsqueeze(0)
    
    print("Example 2 CORRECT: A man is drilling through the frozen ice of a pond .")
    print("Example 2 OUTPUT: " + " ".join(model.caption_image(test_img2.to(device), dataset.vocab)))
    print("\n")
    
    # Third Image
    test_img3 = transform(Image.open("visual-narrator/data/flickr8k/images/1007320043_627395c3d8.jpg").convert("RGB")).unsqueeze(0)
    
    print("Example 3 CORRECT: A little girl climbing on red roping .")
    print("Example 3 OUTPUT: " + " ".join(model.caption_image(test_img3.to(device), dataset.vocab)))
    print("\n")
    
    # Fourth Image
    test_img4 = transform(Image.open("visual-narrator/data/flickr8k/images/1015118661_980735411b.jpg").convert("RGB")).unsqueeze(0)
    
    print("Example 4 CORRECT: A young boy runs aross the street .")
    print("Example 4 OUTPUT: "+ " ".join(model.caption_image(test_img4.to(device), dataset.vocab)))
    print("\n")
    
    # Fifth Image
    test_img5 = transform(Image.open("visual-narrator/data/flickr8k/images/1052358063_eae6744153.jpg").convert("RGB")).unsqueeze(0)
    print("Example 5 CORRECT: A boy takes a jump on his skateboard while another boy with a skateboard watches .")
    print("Example 5 OUTPUT: "+ " ".join(model.caption_image(test_img5.to(device), dataset.vocab)))
    print("\n")
    
    # Back to Training Mode
    model.train()
   