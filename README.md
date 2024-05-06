## Approach
__To address the aforementioned challenges and limitations, we propose an innovative Generative Adverserial Network (GAN)-based framework named Multimodal and Multi-sample Anomalous Single-cell Detection and Annotation (M2ASDA). M2ASDA pioneers in using annotation-free, normal scRNA-seq dataset as reference to detect and subtype ASCs across multiple modalities and target samples. This approach integrates three essential tasks of DAASC(anomaly detection, alignment, and annotation) into a cohesive, three-phase pipeline__
![M2ASDA](https://github.com/kAI-swa/M2ASDA/assets/146005327/47b49595-a954-45a0-a47f-7909e7b93da8)

## Architecture
```lua
m2asda/
|-- Net/
|   |-- __init__.py/
|   |-- _net.py
|   |-- _unit.py
|   |--classifier.py
|   |--discriminator.py
|   |--generator.py
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
|-- __init__.py
|--  _pretrain.py
|-- _utils.py
|-- align.py
|-- correct.py
|-- detect.py
|-- m2asda.py
|-- subtyping.py
|-- LICENSE
```

## Tested environment
- **CPU: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz**
- **Memory: 256 GB**
- **System: Ubuntu 20.04.5 LTS**
- **Python: 3.9.15**
