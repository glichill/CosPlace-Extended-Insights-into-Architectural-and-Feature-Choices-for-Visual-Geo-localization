# CosPlace Extended: Insights into Visual Geo-localization

This project builds upon [CosPlace](https://github.com/gmberton/CosPlace.git), originally created by [Gabriele Berton](https://github.com/gmberton). For installation and usage details, please refer to the [original repository](https://github.com/gmberton/CosPlace.git).

## Abstract

This research extends the CosPlace framework for Visual Geo-localization (VG) to explore various architectural and feature choices. Our aim is to gain insights into the effectiveness of specific features and design decisions within the VG pipeline. We establish a systematic evaluation protocol for method comparison and conduct experiments to benchmark parameters while assessing the impact of engineering techniques on model performance.

## Implementations

### Data Augmentation

We've implemented several data augmentation techniques, including:
- Horizontal flipping
- Blurring
- Color jittering

Each technique offers different parameter configurations. Refer to the accompanying table for results.

|                  |     SF_XS     |     Tokyo_XS     |
|------------------|:-------------:|:----------------:|
|                  |  R@1  |  R@5  |  R@10  |  R@20  |  R@1  |  R@5  |  R@10  |  R@20  |

| Baseline         | 16.3  | 28.1  |  34.0  |  40.1  | 28.9  | 46.0  | 59.0  | 71.1  |
| Random Horizontal Flip | 15.1  | 27.1  | 32.6  | 37.9  | 27.6  | 51.7  | 61.9  | **72.1** |
| Gaussian Blur (kernel_size=5, sigma=(0.5,1)) | 14.5  | 25.3  | 32.1  | 38.3  | 26.1  | 49.8  | 60.0  | 70.1  |
| Color-Jitter with contrast [1.0, 1.5] | **19.7** | **33.0** | 37.9  | 43.6  | **37.8** | **53.7** | 59.0  | 70.2  |
| Color-Jitter with contrast [1.5, 2.0] | 19.5  | 32.1  | **38.3** | **43.8** | 36.5  | 52.7  | **62.2** | 70.8  |
| Color-Jitter with contrast [3.0, 4.0] | 16.9  | 30.1  | 35.8  | 41.9  | 30.8  | 49.2  | 54.0  | 66.0  |

Table: Augmentation results

### Aggregation Layer

We explore three final aggregation layers:
1. **GeM Pooling Layer:** The default layer in the original 'CosPlace.'
2. **MixVPR Pooling Layer:** Inspired by the paper "MixVPR: Feature Mixing for Visual Place Recognition (2023)," with customizations to suit our needs.
3. **NetVLAD Layer:** Initialized using an unsupervised learning approach, with the number of centroids determined via k-means clustering. We found an optimal number of 15 clusters.

Table results demonstrate that MixVPR consistently improves model performance across various recall metrics.

### Domain Adaptation (ADDA)

To address domain shift, we've incorporated Adversarial Domain Adaptation (ADDA) with Tokyo_XS and St_Lucia as target domains, while testing on SF_XS. We used the NetVLAD layer as the aggregation layer. Domain labels were assigned to both source and target data, and a domain discriminator network was trained adversarially to minimize domain distribution differences.

Table results show improved model generalization when Tokyo is chosen as the target domain.

### Optimizers

We conducted experiments using Adam and AdamW optimizers with both NetVLAD and MixVPR aggregators. While the results did not significantly enhance performance, you can find detailed results in the accompanying Report.pdf file.

For comprehensive details, please refer to the original CosPlace repository and the associated report.






