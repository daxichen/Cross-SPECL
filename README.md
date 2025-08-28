# Cross-SPECL
# Harnessing Spectral Low-Frequency Stability and Causal Invariance for Cross-Scene Hyperspectral Image Classification（Under Review）
![DSPLTnet Framework](figure/DSPLTnet.png)
Cross-SPECL integrates three synergistic components for robust cross-scene HSI classification. **1.** First, the spectral patch low-frequency transformation network acts as a feature stabilizer. By learning representations that are invariant to low-frequency perturbations, it effectively filters domain-specific spectral shifts, providing a cleaner and more stable input for the subsequent causal analysis. **2.** Building upon this stabilized feature foundation, a novel domain-agnostic DAG is constructed to learn the underlying causal structure. It leverages this cleaner input to more effectively distinguish and isolate true domain-invariant features while pruning detrimental domain-specific and spurious ones. **3.** To guide the stable discovery of this DAG, a progressive contrastive learning framework is employed to obtain highly representative class-conditional prototypes, which are essential for an effective search of the causal structure. This synergy—where feature stabilization provides a robust basis for causal discovery—is key to the model's enhanced generalization. Notably, during inference, the frequency domain filtering branch, the learned domain-agnostic DAG, and the classifier are utilized for classification.

# Requirements：
```
1. torch==1.11.0+cu113
2. python==3.8.3
3. ptflops==0.6.9
4. timm==0.5.4
```
# Dataset:
The dataset can be downloaded from here: [HSI datasets](https://github.com/YuxiangZhang-BIT/Data-CSHSI). We greatly appreciate their outstanding contributions.

The dataset directory should look like this:
```
datasets
  Houston
  ├── Houston13.mat
  ├── Houston13_7gt.mat
  ├── Houston18.mat
  └── Houston18_7gt.mat
```

# Usage:
Houston datasets:
```
python inference.py --save_path ./results/ --data_path ./datasets/Houston/ --target_name Houston18 --patch_size 8
```
