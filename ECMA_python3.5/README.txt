# Self-Attentive-Tensorflow
#张思原 2018.6.22
$ tree
./ECMA
└──
    ├── noLstm_WH_model  (此模块为ECMA-Attention1相关代码，用于论文4.5.4模型对比实验)
    ├── myMedicalModel  (本文主要代码实现)
    ├── data  （实验数据）
    ├── medicalVector  （预训练好的药向量，但是这个药向量质量不好）
```
数据部分说明：
./data
└──
    ├── aprioriData  (此模块为ECMA-Attention1相关代码，用于论文4.54模型对比实验)
    ├── trainTCM  (各功效训练数据)
    ├── evalData （各功效标准配伍集，用于dice评估）
    ├── profession  （用于论文4.5.3，专家对比实验数据）
    ├── testData（用于论文4.5.4模型对比实验数据）

主要功能代码说明（可以直接运行）：
./myMedicalModel
└──
    ├── loadModel.py  (论文4.5.1和4.5.2章节实验代码，每次跑10次求对应注意力均值得到药组进行dice评估，请在myMedicalModel—>AprioriResults下查看得到的药组项集，然后转用Apriori算法得到一组配伍。)
    ├── modelVSmodel.py  (论文4.5.4代码)
    ├── profession_eval.py （论文4.5.3代码）
    ├── eval_attention.py  （论文4.4.4，设置自注意力阈值,绘图代码）
    ├── vision_4.3.4.py（自注意力以及药物相互作用可视化）
