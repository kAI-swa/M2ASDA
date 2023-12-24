# scRNA-AnomalyDetection
**Anomaly Detection for scRNA-seq Data through GAN-based Model**
![image](https://github.com/Kainan-Liu/scRNA-AnomalyDetection/assets/146005327/523a3b3c-d2c0-4982-8be3-160b9effc05e)


## Notice
**Forked from CatchXu(https://github.com/Catchxu/ACsleuth)**

## Dependencies
- torch==1.13.0
- torchvision==0.14.1
- anndata==0.10.3
- numpy==1.19.2
- scanpy==1.9.6
- scipy==1.9.3
- pandas==1.5.2
- setuptools==59.5.0

## Architecture
```lua
Model/
|-- Net/ #模型结构
|   |-- __init__.py/
|   |-- _net.py   # SCNetAE model
|   |-- _unit.py   # memory_unit, style_unit
|   |--classifier.py   # moduleⅣ subtyping的模型
|   |--discriminator.py   # 判别器
|   |--generator.py   # scNetAE + memory_unit =  Memory_G...以此类推， (不管是哪个module, 基本上可以理解为以scNetAE model为基础，加上memory_unit, style_unit)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|-- __init__.py
|--  _pretrain.py   #预训练
|-- _utils.py        # 一些helpful function
|-- align.py         # ModuleⅡ, align pairs的训练过程, Align_G找pair
|-- correct.py     # ModuleⅢ, 找到pair之后，用Batch_G去除批次效应
|-- detect.py      # Module Ⅰ，anomaly detection
|-- Model.py      # 整个代码的运行入口
|-- subtyping.py # Module Ⅳ, subtyping
|-- LICENSE

## Tested environment
- **CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz**
- **Memory: 256 GB**
- **System: Ubuntu 20.04.5 LTS**
- **Python: 3.9.15**
