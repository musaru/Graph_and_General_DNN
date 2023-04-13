# Dynamic Hand Gesture Recognition Using Multi-Branch Attention Based Graph and General Deep Learning Model

If you used this resource please cited the following paper  "Dynamic Hand Gesture Recognition Using Multi-Branch Attention Based Graph and General Deep Learning Model, https://www.researchgate.net/publication/366980156_Dynamic_Hand_Gesture_Recognition_using_Multi-Branch_Attention_Based_Graph_and_General_Deep_Learning_Model" and "https://arxiv.org/abs/1907.08871"

***Abstract***
The dynamic hand skeleton data have become increasingly attractive to widely studied for the recognition of hand gestures that contain 3D coordinates of hand joints. Many researchers have been working to develop skeleton-based hand gesture recognition systems using various discriminative spatial-temporal attention features by calculating the dependencies between the joints. However, these methods may face difficulties in achieving high performance and owing generalizability due to their inefficient features. To overcome these challenges, we proposed a Multi-branch attention-based graph and a general deep-learning model to recognize hand gestures by extracting all possible types of skeleton-based features. We used two graph-based neural network channels in our multi-branch architectures and one general neural network channel. In graph-based neural network channels, one channel first uses the spatial attention module and then the temporal attention module to produce the spatial-temporal features. In contrast, we produced temporal-spatial features in the second channel using the reverse sequence of the first branch. In the last branch, general deep learning-based features are extracted using a general deep neural network module. The final feature vector was constructed by concatenating the spatial-temporal, temporal-spatial, and general features and feeding them into the fully connected layer. We included position embedding and mask operation for both spatial and temporal attention modules to track the sequence of the node and reduce the computational complexity of the system. Our model achieved 94.12%, 92.00%, and 97.01% accuracy after evaluation with MSRA, DHG, and SHREC’17 benchmark datasets, respectively. The high-performance accuracy and low computational cost proved that the proposed method outperformed the existing state-of-the-art methods.

**Dataset:**
We evaluated the model with Three Dataset
1. MSRA Dataset https://jimmysuen.github.io/
2. Shrec Dataset  http://www-rech.telecom-lille.fr/shrec2017-hand/
3. DhG dataset  http://www-rech.telecom-lille.fr/DHGdataset/

We develop this project by taking idea from "https://arxiv.org/abs/1907.08871", thanks them to open their project code and everything in public."https://github.com/yuxiaochen1103/DG-STA.git"
