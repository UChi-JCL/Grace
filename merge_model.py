import torch
from dvc.net import VideoCompressor

def load_weights(model_a_path, model_b_path):
    # Load the entire models
    compressor = VideoCompressor()

    model_a = torch.load(model_a_path)
    model_b = torch.load(model_b_path)

    # List of submodules to load from model_a
    model_a_modules = ['opticFlow', 'mvEncoder', 'mvDecoder', 'warpnet', 'bitEstimator_mv']

    # Iterate over the submodules in model_a and model_b
    breakpoint()
    for name, module in compressor.named_children():
        if name in model_a_modules:
            # Load the weights from model_a
            module.load_state_dict(model_a[name])
        else:
            # Load the weights from model_b
            module.load_state_dict(model_b[name])
    
    return compressor

if __name__ == "__main__":
    DECODER_ONLY_MODEL = "/dataheart/autoencoder_dataset/datamirror/autoencoder_dataset/dvc-decoder-only/"
    mv_model = "models/pretrained/256.model"
    res_models = {
            "64": f"{DECODER_ONLY_MODEL}/64.model",
            "128": f"{DECODER_ONLY_MODEL}/128.model",
            "256": f"{DECODER_ONLY_MODEL}/256.model",
            "512": f"{DECODER_ONLY_MODEL}/512.model",
            "1024": f"{DECODER_ONLY_MODEL}/1024.model",
            "2048": f"{DECODER_ONLY_MODEL}/2048.model",
            "4096": f"{DECODER_ONLY_MODEL}/4096.model",
            "8192": f"{DECODER_ONLY_MODEL}/8192.model",
            }

    for key in res_models.keys():
        res_path = res_models[key]
        model = load_weights(mv_model, res_path)
        break
