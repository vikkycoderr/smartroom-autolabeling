import torch


def load_slowfast_model(device="cpu"):
    model = torch.hub.load(
        "facebookresearch/pytorchvideo:main",
        model="slowfast_r50",
        pretrained=True
    )

    model = model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    model = load_slowfast_model(device=device)

    print("SlowFast loaded successfully")