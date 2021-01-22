<h1 style='font-size: 1.6em'>Space-Time Correspondence as a Contrastive Random Walk</h1>

<!-- ![](https://github.com/ajabri/videowalk/raw/master/figs/teaser_animation.gif) -->
<p align="center">
<img src="https://github.com/ajabri/videowalk/raw/master/figs/teaser_animation.gif" width="600">
</p>

This is the repository for *Space-Time Correspondence as a Contrastive Random Walk* by Allan Jabri, Andrew Owens, and Alexei A. Efros. 

This work is to be presented at NeurIPS 2020. Here's the [project page](http://ajabri.github.io/videowalk).

> **NOTE**: Part of this code is still subject to a few more cleaning updates -- I am still wrapping up sanity checks after refactoring and fixing a bug. For more details, please see [this](#fixed-a-bug). I just wanted to share the code before the CVPR deadline, in case anyone would like to use the code for evaluation and fair comparison.

##  Requirements
- pytorch (>1.3)
- torchvision (0.6.0)
- cv2
- matplotlib
- skimage
- imageio

For visualization (`--visualize`):
- wandb
- visdom
- sklearn



## Train
An example training command is:
```
python -W ignore train.py --data-path /path/to/kinetics/ \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type scratch --workers 16 --batch-size 20  \
--cache-dataset --data-parallel --visualize --lr 0.0001
```

This yields a model with performance on DAVIS as follows (see below for evaluation instructions), provided as `pretrained.pth`:
```
 J&F-Mean    J-Mean  J-Recall  J-Decay    F-Mean  F-Recall   F-Decay
  0.67606  0.645902  0.758043   0.2031  0.706219   0.83221  0.246789
```

Arguments of interest:

* `--dropout`: The rate of edge dropout (default `0.1`).
* `--clip-len`: Length of video sequence.
* `--temp`: Softmax temperature.
* `--model-type`: Type of encoder. Use `scratch` or `scratch_zeropad` if training from scratch. Use `imagenet18` to load an Imagenet-pretrained network. Use `scratch` with `--resume` if reloading a checkpoint.
* `--batch-size`: I've managed to train models with batch sizes between 6 and 24. If you have can afford a larger batch size, consider increasing the `--lr` from 0.0001 to 0.0003.
* `--frame-aug`: `grid` samples a grid of patches to get nodes; `none` will just use a single image and use embeddings in the feature map as nodes.
* `--visualize`: Log diagonistics to `wandb` and data visualizations to `visdom`.

### Data

We use the official `torchvision.datasets.Kinetics400` class for training. You can find directions for downloading Kinetics [here](https://github.com/pytorch/vision/tree/master/references/video_classification). In particular, the code expects the path given for kinetics to contain a `train_256` subdirectory.

You can also provide `--data-path` with a file with a list of directories of images, or a path to a directory of directory of images. In this case, clips are randomly subsampled from the directory.


### Visualization
By default, the training script will log diagnostics to `wandb` and data visualizations to `visdom`.


### Fixed a bug 
We found a bug in the original training code while refactoring, which affected the way in which transition matrices are multiplied (only relevant to training). The bug essentially contributes additional noise to the transition probabilities of the walk. It is present in experiments reported in the paper, though it does not significantly change the main results. After the fix, performance slightly improved for our best models (with edge dropout), though performance of models with no edge dropout suffers slightly on DAVIS. We want to include a discussion for transparency.

An outcome of the bug is that transitions between nodes of the space-time graph were occasionally shuffled. The fact that models without edge dropout did slightly worse on the VOS task makes sense, as this shuffling (which is removed with the fix) can be seen as having provided a similar effect as edge dropout.


**The main ramification is that the correct version seems to require a lower softmax temperature.** While we originally reported a temperature of 0.07, we found that we needed smaller temperatures, i.e. 0.01 - 0.05, to train the model; this is reflected in the command provided above. 
You can train a model using `--flip` with commensurate hyper-param changes, run:

```
python -W ignore train.py --data-path /path/to/kinetics/ \
--frame-aug grid --dropout 0.1 --clip-len 6 --temp 0.07 --flip \
--model-type scratch --workers 16 --batch-size 20  \
--cache-dataset --data-parallel --visualize --lr 0.0001
```

I am in the process of re-running ablations (edge dropout and path length) and will be updating the results reported in the paper accordingly.

### Pretrained Model
You can find the model resulting from the training command above at `pretrained.pth`.
We are still training updated ablation models and will post them when ready.

---

## Evaluation: Label Propagation
The label propagation algorithm is described in `test.py`.  The output of `test.py` (predicted label maps) must be post-processed for evaluation.

### DAVIS
To evaluate a trained model on the DAVIS task, clone the [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) repository, and prepare the data by downloading the [2017 dataset](https://davischallenge.org/davis2017/code.html) and modifying the paths provided in `eval/davis_vallist.txt`. Then, run:


**Label Propagation:**
```
python test.py --filelist /path/to/davis/vallist.txt \
--model-type scratch --resume ../pretrained.pth --save-path /save/path \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1
```
Though `test.py` expects a model file created with `train.py`, it can easily be modified to be used with other networks. Note that we simply use the same temperature used at training time.

You can also run the ImageNet baseline with the command below.
```
python test.py --filelist /path/to/davis/vallist.txt \
--model-type imagenet18 --save-path /save/path \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1
```


**Post-Process:**  
```
# Convert
python eval/convert_davis.py --in_folder /save/path/ --out_folder /converted/path --dataset /davis/path/

# Compute metrics
python /path/to/davis2017-evaluation/evaluation_method.py \
--task semi-supervised   --results_path /converted/path --set val \
--davis_path /path/to/davis/
```

You can generate the above commands with the script below, where removing `--dryrun` will actually run them in sequence.
```
python eval/run_test.py --model-path /path/to/model --L 20 --K 10  --T 0.05 --cropSize -1 --dryrun
```


## Test-time Adaptation
Coming soon.


## Reference
Please consider citing our work if you found this repository to be helpful.
```
@inproceedings{jabri2020walk,
    Author = {Allan Jabri and Andrew Owens and Alexei A. Efros},
    Title = {Space-Time Correspondence as a Contrastive Random Walk},
    Booktitle = {Advances in Neural Information Processing Systems},
    Year = {2020},
}
```
