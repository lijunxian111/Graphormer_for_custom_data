## 这是项目：Graphormer用户数据适配 的代码，主要改动在于数据集的导入方式，训练的方式等  

# 环境配置：  
先创建一个文件夹，名为Graph_hormer, 将所有压缩包内的文件放进去，然后在该文件夹下打开命令行/bash:  
`
conda create -n Graphormer python=3.9  
conda activate Graphormer  
`  
按照之前文档中已经有的环境配置方法，下载对应的库，然后再下载一个：  
`
pip install pytorch_lightning==2.2.1
`  

#数据集  
采用自己的node.csv和edge.csv，构建了一批用于预测的图，将（不算virtual token的）第一个token视为要预测节点的信息，拿它用于分类：  
训练：验证：测试的比例为7：1：2。也尝试了6：2：2，效果仍然可以。

#预测  
已经给出了一个训练好的权重可以直接调用，位置在`.\graphormer\lightning_logs\version_0\checkpoints\epoch=3-step=1779.ckpt`  
预测模式在文件`.\graphormer\model_cora.py`中，找到418行的`add_model_specific_args`函数，将下面的test这里的default改为True
然后：
`cd graphormer
python entry.py
`  
即可，注意改变路径为自己的路径  

#训练  
上文提到的test这个参数保持False，然后同样运行
`cd graphormer
python entry.py
`  
`add_model_specific_args`还有一些超参数可以调节，建议尝试学习率，weight decay等等  

#目前的结果  
10个epoch之后，能在测试集上面达到95%以上的准确率  


