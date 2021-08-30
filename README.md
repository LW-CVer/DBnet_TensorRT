# DBnet_TensorRT

[参考的原始版本](https://github.com/BaofengZan/DBNet-TensorRT)

[模型训练代码](https://github.com/BaofengZan/DBNet.pytorch)

[模型地址-百度网盘](https://pan.baidu.com/s/1MYm099HuIbUsG4uub_to9g)

提取码：zzl3

[icdar2015训练的pth模型文件](https://pan.baidu.com/s/1zzjZfKFRfSrbEfvBFV03Ag)

提取码：spv2

# 修改

1.训练代码将上采样层的最邻近插值(nearest)修改为双线性插值(bilinear)

2.去掉了减均值除方差前处理（训练也去掉），感觉没用，主要是会导致trt的前处理耗时，至少占上面代码的整个流程的40%，取决于输入图片大小

3.原始tensorrt代码后处理没有分数，box扩大代码有bug，不能按照长宽进行等比例缩放，导致box不准

4.处理性能提高，640x640输入，整个流程耗时17ms

5.训练的模型，recall：90%，precision：77%，hmean(f1):82.7%，resnet18+fpn
