## Train a model

The `experiments` directory contains helper scripts to facilitate training.

First `cd` to `experiments`. A training script is usually invoked like this:
```bash
bash <train_script> <model> <tag> <arg1> <arg2> ...
```
where `<model>` is the directory that contains the model specification (`model_spec.yaml`), and `tag` can be used
to distinguish models if you override the default parameters by passing in additional arguments `<arg1> <arg2> ...` 


For example
```bash
bash train_species_new_dataset_2019.sh resnet_finetune/species default
bash train_species_new_dataset_2019.sh resnet_finetune/species batchsize64 --batch_size=64
```

A training script simply calls actual python scripts for training, but it fills in most of the arguments related to
a particular model or dataset so that you don't have to do it repeatedly.


## Visualize training

We use pytorch's tensorboard tool for visualization. By default logs are stored under `experiments/runs`.

After you have installed tensorboard (e.g., `pip install tensorboard`), cd to `experiments`. Then run
```bash
tensorboard --logdir=runs
```

You should see the visualizer URL. Open it in your browser and you should be able to see the visualization logs. 
