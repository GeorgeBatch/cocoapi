# Milti-label Binary Classification of Coco Images

This repository shoes my work on addressing a question of whether objects of selected categories are present in an image. For example, the categories can be "bird", "cat", "dog", "person". In this case, an image containing a woman holding a cat with a dog by her side (top image below) will have a label [0, 1, 1, 1] since instances of "cat", "dog" and "person" categories are present, while there are no instances of the "bird" category.

![Input Output Example](https://github.com/GeorgeBatch/cocoapi/blob/multilabel-classsification/illustrations/input-output-example.png)

* Top: https://cocodataset.org/#explore?id=42169
* Bottom: https://cocodataset.org/#explore?id=120703

## Motivation

The same workflow can be used when dealing with medical images for which presence or absence of certain morphological patterns can be of utmost importance. For example, presence of certain patterns can signifiy cancer.

## Project Details

- **Title:** Milti-label Binary Classification of Coco Images
- **Aim:** Simulate the workflow for a medical imaging problem while waiting for the data
- **Author:** George Batchkala, george.batchkala@gmail.com
- **Data:** [COCO dataset](http://cocodataset.org/)
- **GitHub repository:** https://github.com/GeorgeBatch/cocoapi


### Contents

```
- annotations (not on GitHub)
- common
- history
- illustrations
- images (not on GitHub)
  |
  ---- train2017
  |    |
  |    ---- 000000000009.jpg
  |    |
  |    ---- …
  |
  ---- val2017
       |
       ---- 000000000139.jpg
       |
       ---- …
- learning_curves
- my_annotations  (not on GitHub)
- my_images  (not on GitHub)
  |
  ---- train1
  |    |
  |    ---- 000000001515.jpg
  |    |
  |    ---- …
  |
  ---- dev1
       |
       ---- 000000000813.jpg
       |
       ---- …
- my_splits  (not on GitHub)
- notebooks
- results
- weights
```

#### In this GitHub repository

* **commmon**: scripts from the original cocoapi. Mainly used for pre-processing.

* **history**: histories of losses, per-class, and average accuracies for each of the training epochs, best epochs and average accuracies. Saved as .json files at the end of each execution of `notebooks/5-plot-and-save-learning-curves.ipynb`.

* **illustrations**: images for README.md.

* **learning_curves**: saved learning curves. Produced in `notebooks/5-plot-and-save-learning-curves.ipynb`.

* **notebooks**: Jupyter notebooks with all the code. Should be ran in order given by numbers.

* **results**: summary of results.

* **weights** contains a README file with links to download pre-trained weights.


#### Not in GitHub repository (saving space)

##### Need to create and populate

Follow the instructions from the original readme of Coco API (at the bottom of the page) to download annotations and images.

* annotations
* images

##### Created in the process while executing the notebooks

* **my_annotations**: image-to-label mappings, produced in `my_annotations`.

* **my_images**: copies of a small subset (~1300) of the images from `images`. `my_images` were uploaded to Google Drive and training was performed in Google Colab. Produced in `notebooks/2.5_copy_train1_and_dev1_images_into_separate_folders_for_colab.ipynb`.

* **my_splits**: .txt files containing ids of train, dev, and test images. Produced in `notebooks/2-splits.ipynb`.


# README.txt (original)

COCO API - http://cocodataset.org/

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
-Please download, unzip, and place the images in: coco/images/
-Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see http://cocodataset.org/#download.

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To install:
-For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
-For Python, run "make" under coco/PythonAPI
-For Lua, run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/
