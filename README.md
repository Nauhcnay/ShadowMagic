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
Feel free the adjust this script if you need different output (like, different naming format or the refined flat png with differernt mask threshold and so on).
