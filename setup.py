# setup.py
import os
from pathlib import Path

def download_model():
    model_dir = Path("./lfm-instruct-dynamic")
    
    if model_dir.exists() and any(model_dir.iterdir()):
        print("✓ Model already exists at ./lfm-instruct-dynamic")
        return
    
    print("Downloading LFM2.5-1.2B-Instruct from HuggingFace...")
    print("This is ~2.5GB — may take a few minutes.\n")
    
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
        local_dir="./lfm-instruct-dynamic",
    )
    
    # Auto-create .env if it doesn't exist
    env_path = Path(".env")
    if not env_path.exists():
        abs_path = str(model_dir.resolve())
        env_path.write_text(
            f"MODEL_PATH={abs_path}\n"
            f"ADAPTER_PATH=./data/lora_adapter\n"
        )
        print(f"\n✓ Created .env with MODEL_PATH={abs_path}")
    
    print("\n✓ Model ready. Run: python main.py")

if __name__ == "__main__":
    download_model()
