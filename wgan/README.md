## Prerequisite

```
aiohttp
pytorch
numpy
pillow
opencv
psd-tools
tqdm
```

## Generate the dataset:

Frist open ./utils/preproceess.py, then edit the `PATH_TO_PSD` and `OUT_PATH` (at the end section of that file) to include the psd folder that 
you want to process and the folder you want save the output png files, respectively. 

This file generally contains two steps (please uncomment the code under each step as your need):
the first step is convert psd file into aligned png files, and the second step is to refine the flat the shadow results. 

Then run:
```
cd utils
python preprocess.py
```
Feel free the adjust this script if you need different output.

## Train
After you get the training data, put them at ./dataset/img/, then run the `train.py` to start the training. 

```
usage: train.py [-h] [-e E] [-m] [-w] [-c C] [-b [B]] [-l [LR]] [-f LOAD] [-r RESIZE] [-i IMGS] [--log]

ShadowMagic Ver 0.1

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 90000)
  -m, --multi-gpu
  -w, --weighted-loss   use mask to weight the loss computation (default: False)
  -c C, --crop-size C   the size of random cropping (default: 512)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.0001)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -r RESIZE, --resize RESIZE
                        resize the shorter edge of the training image (default: 1024)
  -i IMGS, --imgs IMGS  the path to training set (default: None)
  --log                 enable wandb log (default: False)
```

For example, you can run:
```
python train.py -i .\dataset
```
to test if the code works on your computer, this will start a training with batch size 1 by default.

## Misc tools
Here also has a fake backend server (will always return the same result) for the system debug, to use the fake server, simply run:
```
cd utils
python fake_backend.py
```
And here is another tool that could gerenate the hint line layer from given flat layer png files and label json file. To use this function, include the file then call the function:
```
from misc import to_hint_layer
# flat, numpy array as flat layer image
# label, json region label 
# hint, numpy array as hint layer image
hint = to_hint_layer(flat, label)
```
For refining the flat layer, you can call the function in preprocess.py, here is a simple example:
```
from preprocess import flat_refine
from PIL import Image
import numpy as np

flat = np.array(Image.open(PATH_TO_FLAT_LAYER))
line = np.array(Image.open(PATH_TO_LINE_LAYER)) # this is necessary for the refinement, we need the line drawing to align the falt region's boundary
flat, _ = flat_refine(flat, line)
Image.fromarray(flat).save(PATH_TO_REFINED_FLAT_LAYER)
```
