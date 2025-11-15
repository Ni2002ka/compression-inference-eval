# Environment Setup


Create the environment with the required dependencies:

```
conda env create -f environment.yaml
```

Activate the environment:
```
conda activate EE274-proj
```

## Python Venv
Alternatively, we can just create a python venv

```
python3 -m venv EE274-proj
source EE274-proj/bin/activate
pip install -r requirements.txt
```

## Using Myth machines (Stanford)
Stanford myth machines have externally managed python environments, which we need to bypass :)

```
python3 -m venv EE274-proj --without-pip
source EE274-proj/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install -r requirements.txt
```

# Training
We train models remotely and save the trained checkpoints to the `models` directory. From your remote machine, after setting up the env, run 

```
python train.py
```

to train and save all the model-dataset-compression configurations given in `config.py`.
