# Dynamic Hand Gesture Recognition Using Multi-Branch Attention Based Graph and General Deep Learning Model

If you used this resource please cited the following paper  "Dynamic Hand Gesture Recognition Using Multi-Branch Attention Based Graph and General Deep Learning Model, https://www.researchgate.net/publication/366980156_Dynamic_Hand_Gesture_Recognition_using_Multi-Branch_Attention_Based_Graph_and_General_Deep_Learning_Model" and "https://arxiv.org/abs/1907.08871"

***Abstract***
The dynamic hand skeleton data have become increasingly attractive to widely studied for the recognition of hand gestures that contain 3D coordinates of hand joints. Many researchers have been working to develop skeleton-based hand gesture recognition systems using various discriminative spatial-temporal attention features by calculating the dependencies between the joints. However, these methods may face difficulties in achieving high performance and owing generalizability due to their inefficient features. To overcome these challenges, we proposed a Multi-branch attention-based graph and a general deep-learning model to recognize hand gestures by extracting all possible types of skeleton-based features. We used two graph-based neural network channels in our multi-branch architectures and one general neural network channel. In graph-based neural network channels, one channel first uses the spatial attention module and then the temporal attention module to produce the spatial-temporal features. In contrast, we produced temporal-spatial features in the second channel using the reverse sequence of the first branch. In the last branch, general deep learning-based features are extracted using a general deep neural network module. The final feature vector was constructed by concatenating the spatial-temporal, temporal-spatial, and general features and feeding them into the fully connected layer. We included position embedding and mask operation for both spatial and temporal attention modules to track the sequence of the node and reduce the computational complexity of the system. Our model achieved 94.12%, 92.00%, and 97.01% accuracy after evaluation with MSRA, DHG, and SHREC’17 benchmark datasets, respectively. The high-performance accuracy and low computational cost proved that the proposed method outperformed the existing state-of-the-art methods.
![Graphical_Abstract](https://user-images.githubusercontent.com/2803163/231707385-5d163d21-694f-49a2-ae91-adb587fd7f77.jpg)

## Dataset:

We evaluated the model with Three Dataset
1. MSRA Dataset https://jimmysuen.github.io/
2. Shrec Dataset  http://www-rech.telecom-lille.fr/shrec2017-hand/
3. DhG dataset  http://www-rech.telecom-lille.fr/DHGdataset/

We develop this project by taking idea from "https://arxiv.org/abs/1907.08871", thanks them to open their project code and everything in public."https://github.com/yuxiaochen1103/DG-STA.git"


## Citation
If the resource is useful please cite the following paper:
```
@article{miah2023dynamic,
  title={Dynamic Hand Gesture Recognition using Multi-Branch Attention Based Graph and General Deep Learning Model},
  author={Miah, Abu Saleh Musa and Hasan, Md Al Mehedi and Shin, Jungpil},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}

}
@article{shin2023korean,
  title={Korean Sign Language Recognition Using Transformer-Based Deep Neural Network},
  author={Shin, Jungpil and Musa Miah, Abu Saleh and Hasan, Md Al Mehedi and Hirooka, Koki and Suzuki, Kota and Lee, Hyoun-Sup and Jang, Si-Woong},
  journal={Applied Sciences},
  volume={13},
  number={5},
  pages={3029},
  year={2023},
  publisher={MDPI}
}

@article{miah2023multistage,
  title={Multistage Spatial Attention-Based Neural Network for Hand Gesture Recognition},
  author={Miah, Abu Saleh Musa and Hasan, Md Al Mehedi and Shin, Jungpil and Okuyama, Yuichi and Tomioka, Yoichi},
  journal={Computers},
  volume={12},
  number={1},
  pages={13},
  year={2023},
  publisher={MDPI}
}
@article{miah2022bensignnet,
  title={BenSignNet: Bengali Sign Language Alphabet Recognition Using Concatenated Segmentation and Convolutional Neural Network},
  author={Miah, Abu Saleh Musa and Shin, Jungpil and Hasan, Md Al Mehedi and Rahim, Md Abdur},
  journal={Applied Sciences},
  volume={12},
  number={8},
  pages={3933},
  year={2022},
  publisher={MDPI}
}

@article{miahrotation,
  title={Rotation, Translation and Scale Invariant Sign Word Recognition Using Deep Learning},
  author={Miah, Abu Saleh Musa and Shin, Jungpil and Hasan, Md Al Mehedi and Rahim, Md Abdur and Okuyama, Yuichi},
journal={ Computer Systems Science and Engineering },
  volume={44},
  number={3},
  pages={2521–2536},
  year={2023},
  publisher={TechSchince}

}

@inproceedings{chenBMVC19dynamic,
  author    = {Chen, Yuxiao and Zhao, Long and Peng, Xi and Yuan, Jianbo and Metaxas, Dimitris N.},
  title     = {Construct Dynamic Graphs for Hand Gesture Recognition via Spatial-Temporal Attention},
  booktitle = {BMVC},
  year      = {2019}
}
```
## Acknowledgement

Part of our code is borrowed from the (https://arxiv.org/abs/1907.08871). We thank to the authors for releasing their codes.
