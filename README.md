# Environment Setup

Create the environment with the required dependencies:

```
conda env create -f environment.yaml
```

Activate the environment:
```
conda activate EE274-proj
```


# Training
We train models remotely and save the trained checkpoints to the `models` directory. From your remote machine, after setting up the env, run 

```
python train.py
```

to train and save all the model-dataset-compression configurations given in `config.py`.
