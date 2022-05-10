GRAM-TRANCOS (TRaffic ANd COngestionS dataset)
==============================================


Version 
=======
2015.3


Dataset details
===============
This dataset release contains the following folders and files:

  1. images: folder that contains all the images of the TRANCOS dataset, with their corresponding annotations (see next section below for more information).
  2. image_sets: this folder contains the following files:
    -training.txt: text file with all the image names that must be used for training.
    -validation.txt: text file that holds all the image names that must be used for validation, i.e. images used to tune up training parameters.
    -trainval.txt: text file with all the image names included in training.txt and validation.txt. It is useful to perform a final training with the maximum amount of data.
    -test.txt: it is a text file with all the image names that must be used for testing the system, and reporting the final results.

  3. code: folder that contains the code to run the experiments described in our paper.

  4. LICENSE.txt: text file with the license terms.


Annotation format
=================
Image names have the format "image-X-XXXXXX.jpg". All the annotations of an image are provided in the images folder. We provide three types of annotations:

1. Text files that include the pixel coordinates (x,y) for each of the vehicles annotated in the image. These files have the same name as the image but finished with the extension ".txt", i.e. for an image named "image-X-XXXXXX.jpg" the corresponding text annotations file is named "image-X-XXXXXX.txt". Format of the annotation provided:
    x0 y0
    x1 y1
    x2 y2
    .
    .
    .
    xn yn
2. We also provide image files, with the name "image-X-XXXXXXdots.png". These are basically black images with red dots, which indicate the localization of each of the cars in the image.
3. Finally, for each image we provide a mask or region of interest, in the file "image-X-XXXXXXmask.mat", which defined the region of the image that is going to be used during the evaluation
of the system, especially during testing.

Prerequisites
=============
In order to run our codes, it is necessary to download and install the fantastic VLFeat library for Matlab.  http://www.vlfeat.org/

How to run an experiment?
=========================

Within the code folder we provide a Matlab script to reproduce the results reported in our paper:

  1- Open Matlab and set your working directory to "TRANCOS/code".
  2- Run "setup.m". This will compile and setup all the libraries needed.
  3- Run "do_example_experiment.m".

Database Rights
===============

All the images included in the TRANCOS dataset have been captured using the publicly available
surveillance cameras provided by the Dirección General de Tráfico (DGT) of Spain. TRANCOS users
must respect the terms of use defined by the DGT. These images must be strictly used for research purposes.

Check the file "LICENSE.txt" included in the root folder of the database, to know
the license applied to the source code provided.


Acknowledgements
================

This work is supported by projects CCG2014/EXP-054, IPT-2012-0808-370000, TEC2013-45183-R and SPIP2014-1468 of the Dirección General de Tráfico of Spain.


Contact
=======

Please, direct all your comments, questions, bugs at robertoj.lopez@uah.es or daniel.onoro@edu.uah.es


How to cite GRAM-TRANCOS
========================

If you make use of this data and software, please cite the following reference in any publications:

@InProceedings{TRANCOSdataset_IbPRIA2015,
  Title                    = {Extremely Overlapping Vehicle Counting},
  Author                   = {Ricardo Guerrero-Gómez-Olmedo, Beatriz Torre-Jiménez, Roberto López-Sastre, Saturnino Maldonado Bascón, and Daniel Oñoro-Rubio},
  Booktitle                = {Iberian Conference on Pattern Recognition and Image Analysis (IbPRIA)},
  Year                     = {2015}
}

Changelog
=========
2015.3 - Second public release: Review of the documentation, some scripts have been improved.
2015.2 - First public release.
2015.2 - First internal release.
