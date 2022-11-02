# Fisher Information Matrix 

This repository produce a library to calculate the Fisher Information Matrix and the effective dimension of a statistical model.
The library is based on the article [The power of quantum neural networks](https://doi.org/10.1038/s43588-021-00084-1).
The library works for any [Pytorch](https://pytorch.org/)(c++) module.
The module function parameters() represent the $\theta\in\Theta\subset[-1,1]^d$ in the previous reference.
In the constructor of the module one has to take care of the proper initialization of the module parameters 


## Dependencies 

* libtorch from [Pytorch](https://pytorch.org/)
* Intel oneAPI DPC++ Library ([oneDPL](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-dpcpp-library-onedpl.html))
* Intel oneAPI Threading Building Blocks ([oneTBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html#gs.hiug8u)) 
