# VAE
* Optimization of packing structures: VAE + ES
* Estimation of packing property: Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules

## Python Scripts

- training.py -> 训练模型
- testing.py -> 测试模型
- models.py -> 模型定义
- utils.py -> 自定义组件

## Jupyter Notebook

- preprocess.ipynb -> 数据预处理
- analysis.ipynb -> 数据后处理
- end2end.ipynb -> 端到端模型
- experiment.ipynb -> 实验

问题：
1. models（Autoencoder）和network（PointnetAutoencoder）分别是什么功能？为什么有两个模型类？
2. PointnetAutoencoder类里通过call函数添加了损失函数，但在compile时又指定了损失函数，这是为什么？
3. 