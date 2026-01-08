# OM-ReID
<div align="center"><img src="Figures/logo.png" width="900"></div>

We investigate a new challenging problem called **Omni Multi-modal Person Re-identification** (OM-ReID), which aims to achieve effective retrieval with varying multi-modal queries. 

To address dataset scarcity, we construct **ORBench**, the first high-quality multi-modal dataset comprising 1,000 unique identities across five modalities: RGB, infrared, color pencil, sketch, and textual description. Moreover, we propose **ReID5o**, a novel multi-modal learning framework for person ReID. It enables synergistic fusion and cross-modal alignment of arbitrary modality combinations in a single model. 

Details can be found at our paper [ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model](http://arxiv.org/abs/2506.09385).

## News
* 🔥[2026.1.8] We have fixed the relevant bugs in the code and retrained the model. The performance results of the new model are comparable to the previous one, and it is now publicly available.
* 🔥[2025.12.24] We have discovered a bug in the code related to the classification head and are currently fixing it and retraining the model.
* 👏[2025.10.22] The ReID5o model and ORBench dataset are released. Welcome to use!
* 👏[2025.10.17] The OM-ReID challenge (PRCV2025) has concluded. Congratulations to all the winning teams!
* 👍[2025.9.18] Our paper is accepted at NeurIPS2025!
* 🤞[2025.6.13] We are hosting the [OM-ReID challenge](http://2025.prcv.cn/CN/Competitions2/) at PRCV2025 using ORBench. Everyone is welcome to participate! 
* ❤[2025.6.13] The paper is released at [ArXiv](http://arxiv.org/abs/2506.09385).
  
## ORBench
<div align="center"><img src="Figures/ORBench.png" width="900"></div>

We have publicly released our complete ORBench dataset on [Google Drive](https://drive.google.com/file/d/1kfqAY9xfRguV-A_ntZqVtER-fz4uf-jq/view?usp=sharing) and [Baidu Cloud](https://pan.baidu.com/s/1zbfHwXU66KTeVDGzxx2VBg?pwd=1037) to facilitate researchers in downloading and using them. Users are requested to sign the `Data_Use_License_Agreement.pdf` and send it to our email address.

The ORBench dataset has the following format, you should place it in your data root:

```
|-- data_root/
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
ReID5o achieves superior performance by enabling retrieval of the RGB modality from any combination of four modalities (infrared, color painting, sketch, text) within a single model.

### Checkpoint Prepare
[2025/12/24 updated] We have discovered a bug in the code related to the classification head and are currently fixing it and retraining the model. The retrained model checkpoint and corresponding evalutation results will be released recently. To avoid causing any confusion, the previous model checkpoint have been temporarily hidden and will not be made public. Please stay tuned!

**[2026/1/8 updated] We have fixed the relevant bugs in the code and retrained the model. The weights of the new model can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1lbD68fBwVS4wepX3A_V3Pg?pwd=1037).**
```
|-- ReID5o/
|   |-- data/
|   |-- datasets/
|   |-- logs/
|   |   |-- reid5o_ckpt/
|   |   |   |-- best.pth
|   |   |   |-- configs.yaml
|   |-- model/
|   |-- solver/
|   |-- utils/
|   |-- infer.py
|   |-- test.py
```

**The evaluation results of the retrained new model:**
```
+----------------+--------+--------+--------+--------+--------+
|      task      |   R1   |   R5   |  R10   |  mAP   |  mINP  |
+----------------+--------+--------+--------+--------+--------+
|      NIR       | 57.554 | 73.659 | 80.086 | 48.793 | 19.970 |
|       CP       | 85.750 | 93.208 | 95.319 | 75.991 | 41.753 |
|       SK       | 71.292 | 84.417 | 89.264 | 61.149 | 27.740 |
|      TEXT      | 63.535 | 78.310 | 83.463 | 55.224 | 20.681 |
|     NIR+CP     | 94.235 | 98.849 | 99.511 | 85.443 | 48.215 |
|     CP+NIR     | 92.764 | 98.458 | 99.458 | 83.893 | 48.547 |
|     NIR+SK     | 86.014 | 95.607 | 97.583 | 75.431 | 35.977 |
|     SK+NIR     | 84.278 | 95.167 | 97.361 | 74.175 | 36.977 |
|    NIR+TEXT    | 80.374 | 91.904 | 94.878 | 70.196 | 33.007 |
|    TEXT+NIR    | 82.055 | 92.715 | 95.319 | 71.935 | 32.375 |
|     CP+SK      | 85.431 | 94.000 | 95.986 | 74.579 | 38.780 |
|     SK+CP      | 85.111 | 93.792 | 96.000 | 74.429 | 38.549 |
|    CP+TEXT     | 85.597 | 94.181 | 96.222 | 75.296 | 39.321 |
|    TEXT+CP     | 89.717 | 96.582 | 97.895 | 81.021 | 41.056 |
|    SK+TEXT     | 78.083 | 89.556 | 93.153 | 68.010 | 31.869 |
|    TEXT+SK     | 83.651 | 93.645 | 96.094 | 74.749 | 33.788 |
|   NIR+CP+SK    | 95.472 | 99.108 | 99.693 | 86.168 | 48.565 |
|   CP+NIR+SK    | 94.056 | 98.792 | 99.583 | 84.870 | 48.891 |
|   SK+NIR+CP    | 94.333 | 98.903 | 99.611 | 84.737 | 48.931 |
|  NIR+CP+TEXT   | 95.376 | 99.079 | 99.635 | 86.712 | 49.485 |
|  CP+NIR+TEXT   | 94.181 | 98.722 | 99.472 | 85.426 | 49.950 |
|  TEXT+NIR+CP   | 95.867 | 99.263 | 99.695 | 87.765 | 48.311 |
|  NIR+SK+TEXT   | 92.758 | 98.197 | 99.137 | 82.922 | 43.315 |
|  SK+NIR+TEXT   | 91.431 | 97.833 | 98.931 | 81.425 | 43.823 |
|  TEXT+NIR+SK   | 93.690 | 98.521 | 99.319 | 84.076 | 42.181 |
|   CP+SK+TEXT   | 88.486 | 95.750 | 97.444 | 77.786 | 41.397 |
|   SK+CP+TEXT   | 88.417 | 95.778 | 97.403 | 77.926 | 41.316 |
|   TEXT+CP+SK   | 92.499 | 97.673 | 98.693 | 83.174 | 43.269 |
| NIR+CP+SK+TEXT | 95.990 | 99.175 | 99.703 | 87.410 | 50.227 |
| CP+NIR+SK+TEXT | 95.347 | 99.014 | 99.639 | 86.179 | 50.358 |
| SK+NIR+CP+TEXT | 95.444 | 99.125 | 99.639 | 86.208 | 50.203 |
| TEXT+NIR+CP+SK | 96.560 | 99.474 | 99.789 | 88.429 | 49.229 |
|    ONE_AVER    | 69.533 | 82.399 | 87.033 | 60.289 | 27.536 |
|    TWO_AVER    | 85.609 | 94.538 | 96.622 | 75.763 | 38.205 |
|   THREE_AVER   | 93.047 | 98.135 | 99.051 | 83.582 | 45.786 |
|   FOUR_AVER    | 95.835 | 99.197 | 99.692 | 87.056 | 50.004 |
+----------------+--------+--------+--------+--------+--------+
```

### Evaluation
First, you should modify `root_dir` in `configs.yaml` to your data root. Then run the following command:

```
python test.py
```

Once this program finishes (approximately a few minutes), it will provide the evaluation results of ReID5o on the entire ORBench.

### Inference
Run the following command:

```
python infer.py
```

This code uses several specified query examples to generate visual retrieval results from the gallery, and you can freely modify these examples.

Some visual retrieval results:
<div align="center"><img src="Figures/examples.png" width="900"></div>

## Acknowledgements
We thank these great works and open-source repositories: [SYSU-MM01](https://github.com/wuancong/SYSU-MM01), [LLCM](https://github.com/ZYK100/LLCM), [IRRA](https://github.com/anosorae/IRRA), [RDE](https://github.com/QinYang79/RDE).

## Reference
If you use this work in your research, please cite it by the following BibTeX entry:
```
@inproceedings{zuo2025reid5o,
      title={ReID5o: Achieving Omni Multi-modal Person Re-identification in a Single Model}, 
      author={Jialong Zuo and Yongtai Deng and Mengdan Tan and Rui Jin and Dongyue Wu and Nong Sang and Liang Pan and Changxin Gao},
      booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
      year={2025}
}
