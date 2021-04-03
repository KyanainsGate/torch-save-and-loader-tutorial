# torch-save-and-loader-tutorial
## Overview
- Tutorial for following standard functions
    - Visualization by [tensorboard](https://www.tensorflow.org/tensorboard?hl=ja)
    - Save the model and restart training 
    - Support multi device (GPU or NOT)
- Now only MNIST training is available

---
![Visuzlization of training suspension and restart](/readme-doc/tensorboard_img_50p.png)
---

## Details
- Log directory handling
    - All log files(`events.out.tfevent ... `) are saved separated directory
        - In order to avoid overwriting of scalars on tensorboard
    - The directory to storage above is automatically generated by timestamp (e.g. `<path_to_log>/20210224/234590`)
- The checkpoint(`e.g. checkpoint_0005_.pth`) can be load by the method `torch.load`, and that format is following;
    - `epoch`: Last epoch when the weight was saved
    - `state_dict`: Weight, which can be gotten like `<your_model>.state_dict()`
    - `optimizer`: Optimizer information. which can be gotten like `<your_optimizer.state_dict()>`.
- Seamless use of `argparse` between python and ipython

## Requirements
- Hardware setup
    - Windows10
        - Only `device = "cpu"` is confirmed 
    - Ubuntu 18.04 LTS
        - CUDA: 10.0
        - GPU: NVIDIA Tesra t4
- Anaconda installation
    - python 3.6 went well, so creating that environment is recommended
    - Set `<vir_env>` as you like
        ```bash
        conda create -n <vir_env> python=3.6
        ```
- Python packages
    - torchvision: 0.4.1
    - torch: 1.3.0 (CUDA10.0, cp36-cp36m-linux_x86_64)
    - tensorflow-gpu: 1.15.0
    
## Setup
Activate your environment
```bash
conda activate <vir_env>
```
- Download torch wheel files (**CUDA 10.0**) from [here](https://download.pytorch.org/whl/cu100/torch_stable.html)
- Followings are example for `torchvisio==0.4.1` and `torch==1.3.0`
    ```bash
    mkdir tmp && cd tmp
    wget https://download.pytorch.org/whl/cu100/torchvision-0.4.1%2Bcu100-cp36-cp36m-linux_x86_64.whl
    wget https://download.pytorch.org/whl/cu100/torch-1.3.0%2Bcu100-cp36-cp36m-linux_x86_64.whl
    cd ../
    ```
- And then, install these requirements as followings;
    - If you hope, set `tensorflow-gpu==1.15.0` instead of  `tensorflow-gpu`
    ```bash
    pip install -r requirements.txt
    ```
## Start
### Training
- Run training from scratch
    - Default configuration;
        ```bash
        python train.py
        ```
    -  Change training configurations (defined on `parser()` implemented in [iter_classification.py](src/iter_classification.py))
        ```bash
        python train.py -log ./tmp -b 1024  
        ```
- Restart training from loading checkpoint
    - Run restart script, following is example
        ```bash
        python restart.py --restore ./log/20190401/125759/checkpoint_0004_.pth
        ```
    - When you want to same above, you can set laoding model path by coding `TrainingIter.get_load_weight(20190401, 125759, 4)` wihout command line arguments (`start.ipynb` show you example) .
- Or, open `start.ipynb` by jupyter notebook
    ```bash
    jupyter notebook --port 5900 # Set port as you like
    ```

### Visualization
`Tensorboard` will show the transitions of accuracy in all scripts
```bash
cd <log_path_you_set(defaault:log)>/
tensorboard --logdir=./ --port 8181 # Set port as you like
```

## Set virtual interpreter to jupyter
- Jupyter notebook **CANNOT** detect conda environment `<vir_env>` that you made except `(base)` by default
- Thus, type following commands after activated your target environment `<vir_env>`;
    ```bash
    conda activate <vir_env>
    ipython kernel install --user --name <vir_env>
    ```
- In jupyter GUI, set the kernel by `[Kernel]` -> `[Change kernel]` -> `[<vir_env>]`

- Other commands
    - Check your kernel
    ```bash
    jupyter kernelspec list
    ```
    - Remove non-usable kernel
    ```bash
    jupyter kernelspec uninstall KERNEL_NAME
    ```

## Tips
### Why score on training dataset looks a little better when load once saved model and restart training.
- Since the summation of accuracy `epoch_train_percent` and parameter update `loss.backward()` conducted in each batch, the peformance of "optimizing (=ongoing) model" often worse than "optimized (=restored) model".
- That's why the score difference between "ongoing" and "restored" model don`t come from loading failure and is not fixed in this project.
---
![mismatch detail](/readme-doc/why_trainig_curve_mismatch_50p.png)
---
## References
- Argparse handling : https://qiita.com/uenonuenon/items/09fa620426b4c5d4acf9
- Basic implementation (1) : https://github.com/YutaroOgawa/pytorch_advanced
- Basic implementation (2) : https://qiita.com/fukuit/items/215ef75113d97560e599 (Downloading MNIST goes well)
- Tqdm : https://qiita.com/tamahassam/items/6aa9d367e8c15bbd15a3