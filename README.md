# 3DTransformation
### Evaluation of a Spectral Data Transformation Method for Meaningful Mesh Segmentation

**Disclaimer**: the objective of this study is not the development of an efficient 3D Segmentation method, but instead the evaluation of a data transformation method prior to segmentation. For 3D Segmentation repositories, see [this page](https://github.com/topics/3d-segmentation). This project was conducted as part of CS5228 - Knowledge Discovery and Data Mining for the National University of Singapore.

### Abstract

We define a data transformation method based on existing literature, and use well-known clustering algorithms to generate a simple 3D segmentation. A human-generated ground truth segmentation is used to evaluate the segmentation results. Consequently, we are able to measure the improvement brought by the data transformation. Experiments are then performed to determine in which conditions this spectral transformation yields the best results. 

Please read the full paper [here](https://github.com/CorentinDumery/3DTransformation/blob/master/Evaluation%20of%20a%20Spectral%20Data%20Transformation%20Method%20for%20Meaningful%20Mesh%20Segmentation.pdf) for more detail.

### Implementation

The implementation is divided in three files:
 * ```SpectralTransform.py```: Computes transformed points and stores them as numpy array in the ```/points``` folder.
 * ```3DClusters.py```: Defines functions used in our experiments. In particular :
   - ```comparison()``` will run the main experiment and store graph results in the ```/graph``` folder.
   - ```visualizeC(name)``` will store STL data in ```/out``` that can be visualized in an appropriate software, Blender for example. Replace name with the name of the input, "hand" or "glasses" for example.
 * ```ioFunctions.py``` contains essential I/O functions used in the other two files.
 
The inputs are located in:
 * ```/off```: 3D data
 * ```/seg```: ground truth segmentation
 
 ### Examples
 
 ![My planet](https://github.com/CorentinDumery/3DTransformation/blob/master/examples/glassesC.png)
  
 ![My planet](https://github.com/CorentinDumery/3DTransformation/blob/master/examples/handC.png)
 


