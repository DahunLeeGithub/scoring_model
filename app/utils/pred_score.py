import torch
from app.models.ml import RiskClassifier

def load_model(data_size, dict_path):
    model = RiskClassifier(data_size)
    model.load_state_dict(torch.load(dict_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def pred_score(data: torch.Tensor):
    model = load_model(data.size(), 'app/models/scoring_model')
    with torch.no_grad():
        output = model(data)
        return output.numpy().tolist()
