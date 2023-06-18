# RUNE-Net
A refined and reliable Underwater Image Enhancement (UIE) method.

## 🤗Abstract
> Underwater image enhancement (UIE) poses challenges due to distinctive properties of the underwater environment, including low contrast, blurriness, and color distortion. The application of deep learning has quietly revolutionized various areas of scientific research, including UIE, in recent years. However, existing deep learning-based UIE methods generally suffer from issues of weak robustness and limited adaptability. In this paper, we investigated different network architectures extended from attention and residual mechanisms to explore their robustness and effectiveness in enhancing real-world underwater images, and refined beneficial architectural components to propose a more reliable and reasonable UIE network called *RUNE-Net*. Furthermore, we have collected and created two datasets specifically designed for evaluating UIE methods, which contains different types of underwater distortions and degradations. Through experimental validation, our model has demonstrated promising objective performance and consistent visual results across various underwater image test sets compared to other UIE methods.

## Experimental Results

### Experiments on Different UIE Methods
we compared RUNE-Net with eight other UIE methods (i.e., UT-UIE [13], SyreaNet [17], WaterNet [8], UGAN [3], FUnIE-GAN [6], UWCNN [7], SGUIE-Net [14], Cycle-GAN [9]) using the best model from RUNE-Net.

#### Full-reference Evaluation Results
![Table 2. Full-reference evaluation results of different UIE methods](./paper_figures/full-ref_different_UIE.png)
#### Non-reference Evaluation Results
![Non-reference evaluation results of different methods on U45](./paper_figures/different_methods_comparison_noref_U45.png)
![Non-reference evaluation results of different methods on RUIE_Color90](./paper_figures/different_methods_comparison_noref_RUIE_Color90.png)
![Non-reference evaluation results of different methods on UPoor200](./paper_figures/different_methods_comparison_noref_UPoor200.png)
## 🎯Notice
**The whole code will come soon. Please wait :)**

## References
[3] Fabbri, C., Islam, M.J., Sattar, J.: Enhancing underwater imagery using generative adversarial networks. In: 2018 IEEE International Conference on Robotics and Automation (ICRA). pp. 7159–7165. IEEE (2018)
[6] Islam, M.J., Xia, Y., Sattar, J.: Fast underwater image enhancement for improved visual perception. IEEE Robotics and Automation Letters 5(2), 3227–3234 (2020)
[7] Li, C., Anwar, S., Porikli, F.: Underwater scene prior inspired deep underwater image and video enhancement. Pattern Recognition 98, 107038 (2020)
[8] Li, C., Guo, C., Ren, W., Cong, R., Hou, J., Kwong, S., Tao, D.: An underwater image enhancement benchmark dataset and beyond. IEEE Transactions on Image Processing 29, 4376–4389 (2019)
[9] Li, C., Guo, J., Guo, C.: Emerging from water: Underwater image color correction based on weakly supervised color transfer. IEEE Signal processing letters 25(3), 323–327 (2018)
[13] Peng, L., Zhu, C., Bian, L.: U-shape transformer for underwater image enhancement. IEEE Transactions on Image Processing (2023)
[14] Qi, Q., Li, K., Zheng, H., Gao, X., Hou, G., Sun, K.: Sguie-net: Semantic attention guided underwater image enhancement with multi-scale perception. IEEE Transactions on Image Processing 31, 6816–6830 (2022)
[17] Wen, J., Cui, J., Zhao, Z., Yan, R., Gao, Z., Dou, L., Chen, B.M.: Syreanet: A physically guided underwater image enhancement framework integrating synthetic and real images. arXiv preprint arXiv:2302.08269 (2023)

