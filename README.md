## ExecuTorch Inference using EfficientNet
### Clone the repo into a new directory:
`git clone https://github.com/orangewolf79/executorch_tests`
### For macOS development, make sure Xcode Command Line Tools are installed.
### CMake and Ninja are also dependencies:
`brew install cmake ninja`
### Full documentation is provided at https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html
### For this project, only the commands below are required:
```bash
cd third_party
git clone -b release/0.6 https://github.com/pytorch/executorch.git && cd executorch 
```
### This project uses pyenv to manage python versions. The exact version can be found in '.python-version'
`python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip`
### Run from within the ExecuTorch directory once the virtual environment is activated
```bash
git submodule sync
git submodule update --init --recursive
./install_executorch.sh
```
### CMake Requirement >= 3.15 (Check CMakeLists.txt)
### From the root of the directory:
`pip install -r requirements.txt`
### Run preprocessing.py to export the model and the image tensor
`python scripts/preprocessing.py`
### model.pte and image_tensor.pt should be saved to data/
## 1. Running Locally:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./main
```
### Presently, "cmake --build ." causes failure, citing undefined symbols for architecture arm_64
## 2. Running Docker Image:
### Check Dockerfile for more details
### From the root of the project:
### Also causes failure - buck2 installation fails even though it is not required for running locally
`docker build -t executorch_inference .`
### For testing with limited memory (adjustable):
`docker run --rm -it --memory=1g executorch_inference`