from pathlib import Path
import torch


def save_model(model:torch.nn.Module,base_dir:str,model_name:str):

    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True,parents=True)

    assert model_name.endswith('pth') or model_name.endswith('pt'), 'The format should be .pt or .pth'
    model_path = base_dir / model_name

    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')