# HMC_DropOut

## In this repo you can find, c++ code of:
* Hamiltonian Monte Carlo (hmc)
* Multivariable Hamiltonian Monte Carlo (mhmc)
* DropOut Hamiltonian Monte Carlo (dhmc)
* Multivariable DropOut Hamiltonian Monte Carlo (dmhmc)
* Logistic Regression (lr)
* Softmax Regression (sr)
* Utils
* Others

## Excecution:

unzip datasets:

~~~bash

cat data.zip* > data.zip
unzip data.zip

~~~

compile programs:

~~~bash

mkdir build
cd build/
cmake ..
make

~~~


run programs:

~~~bash

cd build/
./<program>

~~~


## Keras VGG_Face Age Dataset Creation:

For the creation of the features through Keras VGG_Face, first it is necessary to download the ADIENCE faces database from the following links:

https://drive.google.com/drive/folders/1A0EDo0oYH3pBEZyq6zfk_jVg8ZvYM2cE?usp=sharing

or

https://www.openu.ac.il/home/hassner/Adience/data.html

Download "aligned.tar.gz" archive, then:

~~~bash

mv download_path/aligned.tar.gz dropout-hmc_cpp/data/

cd dropout-hmc_cpp/data/

tar -xvf aligned.tar.gz 

~~~


Run python scripts:

~~~bash

cd dropout-hmc_python/python/

python keras_vgg_face_features.py

~~~

## Requirements:

* CMake
* C++ 4.8 or later, C++ 11
* OpenCV 3.4 or later
* Eigen 3.3 or later 
* Python 2.7

* Tensorflow 1.3 or later, Tensorflow-gpu (alternative)
* Edward 1.3 or later
* Keras 2.1 or later, keras-vggface
* SKlearn, Numpy, Scipy, Pandas, Seaborn, Matplotlib (according to dependence on previous packages)
