echo "!!!DO NOT run this script directly. Pick the right configuration and run each line manually!!!"
exit(0)

# Common Packages
pip install transformers datasets accelerate tqdm peft protobuf sentencepiece matplotlib scikit-learn

# For Nvidia Driver Version: 570.124.06
#   torch==2.7.0+cu126 transformers==4.55.2 peft==0.17.0 numpy==2.2.6

# For Nvidia Driver Version: 570.133.20
#   torch==2.7.1+cu126 transformers==4.55.2 peft==0.17.0 numpy==2.3.2

# To install torch with a specific CUDA version:
# pip install torch==2.7.1 torchvision==0.22.1 \
#   --index-url https://download.pytorch.org/whl/cu126  # or 128
# pip install torch==2.7.0 torchvision==0.22.0 \
#   --index-url https://download.pytorch.org/whl/cu126  # or 128

# Common Tools
sudo apt update
sudo apt install nano
sudo apt install gh
sudo apt install unzip sqlite3 git-lfs

# Authentication for GitHub and HuggingFace.
gh auth login
huggingface-cli login
