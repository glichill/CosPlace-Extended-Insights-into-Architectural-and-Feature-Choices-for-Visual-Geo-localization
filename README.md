# CosPlace Extended: Insights into Visual Geo-localization

This project builds upon [CosPlace](https://github.com/gmberton/CosPlace.git), originally created by [Gabriele Berton](https://github.com/gmberton). For installation and usage details, please refer to the [original repository](https://github.com/gmberton/CosPlace.git).

## Abstract

Our research aims to enhance the CosPlace framework for Visual Geo-localization (VG) by exploring various architectural and feature choices. We focus on constructing, training, and evaluating different models to gain insights into the effectiveness of specific design decisions in the VG pipeline. Our objective is to establish a systematic evaluation protocol for method comparison. Leveraging our framework, we conduct extensive experiments to benchmark and optimize model parameters while assessing the impact of engineering techniques on model performance.

## Implementations

### Data Augmentation

We've implemented several data augmentation techniques, including:
- Horizontal flipping
- Blurring
- Color jittering

Each technique offers different parameter configurations. Refer to the accompanying table for results.

|              | SF_XS R@1 | SF_XS R@5 | SF_XS R@10 | SF_XS R@20 | Tokyo_XS R@1 | Tokyo_XS R@5 | Tokyo_XS R@10 | Tokyo_XS R@20 |
|--------------|:---------:|:---------:|:----------:|:----------:|:------------:|:------------:|:-------------:|:-------------:|
| Baseline     |   16.3    |   28.1    |    34.0    |    40.1    |     28.9     |     46.0     |      59.0     |      71.1     |
| Random Horizontal Flip |   15.1    |   27.1    |    32.6    |    37.9    |     27.6     |     51.7     |      61.9     |     72.1  |
| Gaussian Blur (kernel_size=5, sigma=(0.5,1)) |   14.5    |   25.3    |    32.1    |    38.3    |     26.1     |     49.8     |      60.0     |     70.1  |
| Color-Jitter with contrast [1.0, 1.5] |   19.7    |   33.0    |    37.9    |    43.6    |     37.8     |     53.7     |      59.0     |     70.2  |
| Color-Jitter with contrast [1.5, 2.0] |   19.5    |   32.1    |    38.3    |    43.8    |     36.5     |     52.7     |     62.2  |     70.8  |
| Color-Jitter with contrast [3.0, 4.0] |   16.9    |   30.1    |    35.8    |    41.9    |     30.8     |     49.2     |      54.0     |     66.0  |


Table: Augmentation results

### Aggregation Layer

We explore three final aggregation layers:
1. **GeM Pooling Layer:** The default layer in the original 'CosPlace.'
2. **MixVPR Pooling Layer:** Inspired by the paper "Brahim Chaib-draa Amar Ali-bey and Philippe Giguere, Mixvpr: Feature Mixing for Visual Place Recognition (2023)." We made customizations to suit our specific requirements.
3. **NetVLAD Layer:** This layer is initialized using an unsupervised learning approach to determine cluster centroids. The optimal number of clusters (15) is established through an elbow search using the k-means algorithm.

Table results demonstrate that MixVPR consistently improves model performance across various recall metrics.

| Aggregator | SF_XS R@1 | SF_XS R@5 | SF_XS R@10 | SF_XS R@20 | Tokyo_XS R@1 | Tokyo_XS R@5 | Tokyo_XS R@10 | Tokyo_XS R@20 |
|------------|-----------|-----------|------------|------------|--------------|--------------|---------------|---------------|
| GeM        | 19.1      | 30.4      | 36.2       | 43.1       | 38.1         | 55.9         | 63.5          | 71.4          |
| NetVLAD    | 22.8      | 39.0      | 44.7       | 51.0       | 36.2         | 55.2         | 64.4          | 72.7          |
| MixVPR     | **30.9**      | **44.4**      | **51.8**       | **57.4**       | **48.9**         | **68.9**         | **75.9**          | **81.6**          |

Table: Aggregation results

### Domain Adaptation (ADDA)

To address the domain shift problem, we incorporated Adversarial Domain Adaptation (ADDA) using Tokyo_XS and St_Lucia as target domains while testing on SF_XS. In this setup, we employed the NetVLAD aggregation layer and utilized a domain discriminator network during training. The goal was to minimize the domain distribution discrepancy between the source and target domains.

Table results show improved model generalization when Tokyo is chosen as the target domain.

| Source Domain | Target Domain | SF_XS Test      |
|---------------|---------------|-----------------|
|               |               | R@1: 24.7       |
| SF_XS         | Tokyo_XS      | R@5: 39.0       |
| train         | database      | R@10: 46.4      |
|               |               | R@20: 54.0      |
|               |               |                 |
|               |               | R@1: 24.8       |
| SF_XS         | St_Lucia      | R@5: 38.3       |
| train         | database      | R@10: 44.2      |
|               |               | R@20: 51.3      |

Table: Domain Adaptation results

### Optimizers

In our final set of experiments, we evaluated the performance of the Adam and AdamW optimizers using both the NetVLAD and MixVPR aggregation layers. However, the results did not demonstrate a significant improvement in performance. For detailed optimizer results, please refer to the Report.pdf file.

This research extends the CosPlace framework, providing valuable insights into architectural and feature choices for visual geo-localization. Our experiments and findings contribute to a better understanding of how different design decisions impact model performance in this domain.






