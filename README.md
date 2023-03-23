# Augmented-Notes-GNNs

Welcome to the repo! Here, we'll provide detailed instructions to duplicate the results from this blog post - https://medium.com/@arjunkaranam10/augmenting-your-notes-using-graph-neural-networks-e61f0898033a. 

## Downloading the repo 
Go to your terminal, and clone the git repo by running <code>git clone https://github.com/QuantumArjun/Augmented-Notes-GNNs.git</code>
After running this command, you should have a copy of the git repo on your computer.

## Setting up your environment

To ensure that your package versions aling with ours, run the following command inside Augmented-Notes-GNNs <code>pip install -r requirements.txt<.code>
This should install the necessary packages onto your computer.

## Accessing the Dataset
The repo comes with a compressed version of the dataset (in the form of graph.pickle under dataset/model

Instructions will come shortly on how to use our wiki extractor in order to create a dataset of you're own. In the meantime, you can run through our logic at this colab: https://colab.research.google.com/drive/1EojTdUDdM-NuFIjveF-6XeADb1msxqLO?usp=sharing

## Training the Model

To train the model, navigate over to /experiments/linkprediction

Once you're inside this folder, run the following command
<code> python train_vgae.py --dataset ../../dataset/model/graph.pickle</code>

The dataset flag tells the code where it should get the dataset from. If you've created your own, point it towards this direction. 

Other flags you can control are:
* --epochs, default = 200

* --val_freq, default = 20 (how often validation results are printed)

* --runs, default = 3

* --test, default = false (whether you want to run in validation mode)

* --save_dir, default = "../../results/data/vgae/", where you want to save your model 

While you're running the model, you can run the following command <code>tensorboard --logdir="results/logs" </code> in order to see the results ive on tensorboard.

And there you go! You're able to train and test the GNN locally on your computer. Unfortunately, in order to try it out on your own notes, you need access to the full dataset, which is currently 25gb. We are currently working on extracting just the title so that the prediction doesn't require accessing the full graph.
