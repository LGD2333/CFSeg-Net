# CFSeg-Net

Image segmentation is critical in medical image analysis, enabling accurate identification and localization of key regions within images. This accuracy is indispensable for reliable diagnosis and effective treatment planning. To achieve an outstanding model performance with moderate parameters, a context feature extraction network for medical image segmentation (abbreviated as CFSeg-Net) has been proposed. The proposed CFSeg-Net is designed to enhance the understanding of the context features. The CFSeg-Net primarily comprises two components: the context feature extraction (CFE) module and feature enhancement (FE) module. 
The CFE module leverages a multi-range perception scheme, channel shuffle, and multi-feature integration scheme to achieve a broader receptive field. A broader receptive field leads to a better understanding of context features and better segmentation performance. Notably, model performance can be further improved when the FE module is inserted between two CFE modules in the encoder.
Extensive experiments have been conducted on four public medical image datasets, which are ISIC2018, Kvasir-SEG, BUSI, and CVC-ClinicDB, respectively. Experimental results demonstrate that the proposed CFSeg-Net achieves outstanding model performance. These results highlight CFSeg-Net’s potential as a robust and efficient tool for segmentation in medical imaging. 


## Experiment
In the experimental section, four publicly available and widely utilized datasets are employed for testing purposes. These datasets are:<br> 
ISIC-2018 (dermoscopy, with 2,594 images)<br>
Kvasir-SEG (endoscopy, with 1,000 images)<br> 
BUSI (breast ultrasound, with 437 benign and 210 malignant images)<br> 
CVC-ClinicDB (colonoscopy, with 612 images)<br>  

For each dataset, the images are randomly split into training, validation, and test sets with a ratio of 6:2:2.<br>
The dataset path may look like:
```bash
/The Dataset Path/
├── ISIC-2018/
    ├── Train_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Val_Folder/
    │   ├── img
    │   ├── labelcol
    │
    ├── Test_Folder/
        ├── img
        ├── labelcol
```


## Usage

---

### **Installation**
```bash
git clone git@github.com:LDG2333/CFSeg-Net.git
conda create -n cfseg python=3.8
conda activate cfseg
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
``` 


### **Training**
```bash
python Experiment/train_model.py
```
To run on different setting or different datasets, please modify:

batch_size, model_name, and task_name in Experiment/Config.py .


### **Evaluation**
```bash
python Experiment/test_model.py
``` 


## Citation

If you find our repo useful for your research, please consider citing our article. <br>
This article has been submitted for peer-review in the journal called *The visual computer*.<br>
```bibtex
@ARTICLE{40046799,
  author  = {Guodong Li, Shiren Li, Yaoxue Lin, Sihua Tang, Wenguang Xu, Kangxian Chen, Guangguang Yang},
  journal = {The Viusal Computer}
  title   = {CFSeg-Net: Context Feature Extraction Network for Medical Image Segmentation},
  year    = {2025}
}
``` 


## Contact

For technical questions, please contact guodong0012@gmail.com .
