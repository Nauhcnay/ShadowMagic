## Prerequisite

```
numpy
pillow
opencv
psd-tools
tqdm
```

## Generate the dataset:

Frist open ./utils/preproceess.py, then edit the PATH_TO_PSD and OUT_PATH to include the psd file that 
you want to process and the folder you want save the output png files, respectively. 

This file generally contains two steps (please uncomment the code under each step as your need):
the first step is convert psd file into aligned png files, and the second step is to refine the flat the shadow results. 

Then run:
```
cd utils
python preprocess.py
```
Feel free the adjust this script if you need different output. For example:

if you want to change the naming format, please look at line 58, change the varibale `png` to the new format you want.

or if you want ot change the refinement degree of the flat png output, please look at line 193, change the iteration times will make the refinment stronger (more iters) or weaker (less iters).
