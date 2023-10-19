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






