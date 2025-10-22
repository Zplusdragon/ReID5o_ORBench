# OM-ReID
<div align="center"><img src="Figures/logo.png" width="900"></div>

We investigate a new challenging problem called **Omni Multi-modal Person Re-identification** (OM-ReID), which aims to achieve effective retrieval with varying multi-modal queries. 

To address dataset scarcity, we construct **ORBench**, the first high-quality multi-modal dataset comprising 1,000 unique identities across five modalities: RGB, infrared, color pencil, sketch, and textual description. Moreover, we propose **ReID5o**, a novel multi-modal learning framework for person ReID. It enables synergistic fusion and cross-modal alignment of arbitrary modality combinations in a single model. 

More details can be found at our paper [ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model](http://arxiv.org/abs/2506.09385).

## News
* 🔥[2025.10.22] The ReID5o model and ORBench dataset are released. welcome to use!
* 👏[2025.10.17] The OM-ReID challenge (PRCV2025) has concluded. Congratulations to all the winning teams!
* 👍[2025.9.18] Our paper is accepted at NeurIPS2025!
* 🤞[2025.6.13] We are hosting the [OM-ReID challenge](http://2025.prcv.cn/CN/Competitions2/) at PRCV2025 using ORBench. Everyone is welcome to participate! 
* ❤[2025.6.13] The paper is released at [ArXiv](http://arxiv.org/abs/2506.09385).
  
## ORBench
<div align="center"><img src="Figures/ORBench.png" width="900"></div>

We have publicly released our complete ORBench dataset on [Google Drive](https://drive.google.com/drive/folders/1PSFXdc4Iid7RK3DicTZ3Tfygu00ntrFn?usp=sharing) to facilitate researchers in downloading and using them. Users are requested to voluntarily sign the relevant agreements and send them to our email address.

The ORBench dataset has the following format, you should place it in your data root:

```
|-- data/
|   |-- ORBench/
|   |   |-- cp/
|   |       |-- 0001
|   |       |-- 0002
|   |       |-- ...
|   |       |-- 0999
|   |       |-- 1000
|   |   |-- nir/
|   |   |-- sk/
|   |   |-- vis/
|   |   |-- train_annos.json
|   |   |-- test_gallery_and_queries.json
```

## ReID5o
The ReID5o code and model will be made publicly available after the paper is accepted. Stay tuned!

## Reference
If you use this work in your research, please cite it by the following BibTeX entry:
```
@inproceedings{zuo2025reid5o,
      title={ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model}, 
      author={Jialong Zuo and Yongtai Deng and Mengdan Tan and Rui Jin and Dongyue Wu and Nong Sang and Liang Pan and Changxin Gao},
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
      year={2025}
}
