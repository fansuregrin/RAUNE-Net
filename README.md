# RAUNE-Net
A Residual and Attention Powered Underwater Image Enhancement Method.

## 🤗Abstract
> Underwater image enhancement (UIE) poses challenges due to distinctive properties of the underwater environment, including low contrast, high turbidity, visual blurriness, and color distortion. In recent years, the application of deep learning has quietly revolutionized various areas of scientific research, including UIE. However, existing deep learning-based UIE methods generally suffer from issues of weak robustness and limited adaptability. In this paper, inspired by residual and attention mechanisms, we propose a more reliable and reasonable UIE network called *RAUNE-Net* by employing residual learning of high-level features at the network's bottle-neck and two aspects of attention manipulations in the down-sampling procedure. Furthermore, we have collected and created two datasets specifically designed for evaluating UIE methods, which contains different types of underwater distortions and degradations. The experimental validation demonstrate that our method obtains promising objective performance and consistent visual results across various underwater image test-sets compared to other eight UIE methods.

## 🛖Datasets
We used 1 dataset for training and 4 full-reference datasets and 3 non-reference datasets for testing. You can download these datasets by clicking links below.

### Training Datasets
- LSUI3879
    - 3879 pairs of images randomly selected from LSUI [13].
    - [🔗GoogleDrive](https://drive.google.com/file/d/19UGfKpgqNiue3SD765xwy-P5dsqrAk8X/view?usp=sharing)

### Test Datasets
#### Full-reference Datasets
- LSUI400
    - 400 pairs of images remained in LSUI [13].
    - [🔗GoogleDrive](https://drive.google.com/file/d/1p_WSipuOnsW_HyKU1IZoI-iv1BPqbHN_/view?usp=sharing)
- UIEB100
    - 100 pairs of images randomly selected from UIEB [8].
    - [🔗GoogleDrive](https://drive.google.com/file/d/1QcVZbfGiNq3qCU2SrciBizQdCHm3L7ed/view?usp=sharing)
- EUVP_Test515
    - 515 pairs of testing images from EUVP [6].
    - [🔗GoogleDrive](https://drive.google.com/file/d/1Jy9AGn5MgPyyZsQgtbD1voEx7IsYlWpx/view?usp=sharing)
- OceanEx
    - We gathered 40 high-quality underwater images from the official website of [NOAA Ocean Exploration](https://oceanexplorer.noaa.gov/image-gallery/welcome.html). Then, we applied CycleGAN to add underwater distortions and degradation styles to these images, making them the samples to be enhanced, while keeping the original high-quality images as reference images.
    - [🔗GoogleDrive](https://drive.google.com/file/d/1I9u8lUvPTsk9OXOcqGTKkWu0uFfjKnQM/view?usp=sharing)
#### Non-reference Datasets
- U45
    - 45 real-world images from [10].
    - [🔗GoogleDrive](https://drive.google.com/file/d/1AUBv8gQZGvd8YDWuEexpXq9dOWJ0jCAT/view?usp=sharing)
- RUIE_Color90
    - 90 real-world images from RUIE [12], which contains 3 types of color cast (i.e., greenish, blue-greenish, and bluish).
    - [🔗GoogleDrive](https://drive.google.com/file/d/1VX7Y7PjqVN3a32i4O8OSD1LprJrtwl71/view?usp=sharing)
- UPoor200
    - UPoor200 is a dataset we collected and created, which consists of 200 real-world underwater images with poor visual quality. It includes distortions such as blue-green color cast, low lighting, blurriness, noise, and other types of distortions.
    - [🔗GoogleDrive](https://drive.google.com/file/d/1BxUMLfx0VVX2odrjfcTHJ5P-OBsyvvLf/view?usp=sharing)

## 🎲Experimental Results

### Experiments on Different UIE Methods
we compared RUNE-Net with eight other UIE methods (i.e., UT-UIE [13], SyreaNet [17], WaterNet [8], UGAN [3], FUnIE-GAN [6], UWCNN [7], SGUIE-Net [14], Cycle-GAN [9]) using the best model from RUNE-Net.

#### Objective Evaluation Results
![Table 2. Objective evaluation results of different UIE methods](./paper_figures/objective_evaluation_different_UIE.png)
#### Subjective Evaluation Results
![Subjective evaluation results of different methods on U45](./paper_figures/different_methods_comparison_subj_eval_U45.png)
![Subjective evaluation results of different methods on RUIE_Color90](./paper_figures/different_methods_comparison_subj_eval_RUIE_Color90.png)
![Subjective evaluation results of different methods on UPoor200](./paper_figures/different_methods_comparison_subj_eval_UPoor200.png)
## 🎯Notice
**The whole code will come soon. Please wait :)**

## 🤔Q&A
If you have any question about this project, please contact `fansuregrin` through **pwz113436@gmail.com**! Btw, any pull request is welcome if you are interested in this project:)

## 📔References
[3] Fabbri, C., Islam, M.J., Sattar, J.: Enhancing underwater imagery using generative adversarial networks. In: 2018 IEEE International Conference on Robotics and Automation (ICRA). pp. 7159–7165. IEEE (2018)

[6] Islam, M.J., Xia, Y., Sattar, J.: Fast underwater image enhancement for improved visual perception. IEEE Robotics and Automation Letters 5(2), 3227–3234 (2020)

[7] Li, C., Anwar, S., Porikli, F.: Underwater scene prior inspired deep underwater image and video enhancement. Pattern Recognition 98, 107038 (2020)

[8] Li, C., Guo, C., Ren, W., Cong, R., Hou, J., Kwong, S., Tao, D.: An underwater image enhancement benchmark dataset and beyond. IEEE Transactions on Image Processing 29, 4376–4389 (2019)

[9] Li, C., Guo, J., Guo, C.: Emerging from water: Underwater image color correction based on weakly supervised color transfer. IEEE Signal processing letters 25(3), 323–327 (2018)

[10] Li, H., Li, J., Wang, W.: A fusion adversarial underwater image enhancement network with a public test dataset. arXiv preprint arXiv:1906.06819 (2019)

[12] Liu, R., Fan, X., Zhu, M., Hou, M., Luo, Z.: Real-world underwater enhancement: Challenges, benchmarks, and solutions under natural light. IEEE Transactions on Circuits and Systems for Video Technology 30(12), 4861–4875 (2020)

[13] Peng, L., Zhu, C., Bian, L.: U-shape transformer for underwater image enhancement. IEEE Transactions on Image Processing (2023)

[14] Qi, Q., Li, K., Zheng, H., Gao, X., Hou, G., Sun, K.: Sguie-net: Semantic attention guided underwater image enhancement with multi-scale perception. IEEE Transactions on Image Processing 31, 6816–6830 (2022)

[17] Wen, J., Cui, J., Zhao, Z., Yan, R., Gao, Z., Dou, L., Chen, B.M.: Syreanet: A physically guided underwater image enhancement framework integrating synthetic and real images. arXiv preprint arXiv:2302.08269 (2023)

