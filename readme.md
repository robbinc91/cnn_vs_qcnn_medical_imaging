# Comparison of UNet and Quaternion-based UNet (QCNN) for Medical Image Segmentation

This repository contains an implementation and comparison of a standard Convolutional Neural Network (UNet) and a Quaternion-based Convolutional Neural Network (QCNN) for medical image segmentation. The models are applied to the task of **binary segmentation of the brainstem** from MRI scans, replicating the methodology from the original study.

### **Key Features**
- **Basic train-and-evaluate scripts** for both a standard UNet and a Quaternion-modified UNet (QCNN).
- Code designed for **binary segmentation**, specifically applied to brainstem segmentation.

## 🧠 QCNN Internals: How Quaternion-Based CNNs Work

Quaternion-based Convolutional Neural Networks (QCNNs) extend traditional CNNs by leveraging **quaternion algebra**, a type of hypercomplex number system. Here's how they work internally:

- **Quaternion Representation**: A quaternion \( \mathbf{q} \) is represented as \( q_R + q_I\hat{i} + q_J\hat{j} + q_K\hat{k} \), where \( q_R \) is the real part and \( q_I, q_J, q_K \) are three imaginary components. In image processing, this structure can naturally represent color channels (e.g., RGB) or multi-modal medical images as a single, cohesive entity.

- **Hamilton Product**: The core operation in a QCNN is the **Hamilton product**, which replaces the standard real-valued convolution. This product captures latent inter-dependencies between the input channels when convolving with quaternion-valued filters. This allows the network to model internal relationships within the data more effectively than real-valued networks.

- **Advantages**: QCNNs can achieve performance comparable to or better than real-valued CNNs while using **fewer parameters**. They are particularly well-suited for tasks involving multi-channel data, such as color images or multi-parametric MRIs, because they process the input as a unified whole rather than as separate channels.

## 🔗 Related Repositories

- **Original Study Repository**: [https://github.com/robbinc91/mipaim_unet](https://github.com/robbinc91/mipaim_unet)


## 📄 License
This project is for academic and research purposes. Please refer to the original study's repository for specific licensing details.

## ❓ FAQs and Common Issues
- **Q: Why is my QCNN training slower?**
  - A: Quaternion operations (Hamilton product) are computationally more complex than real-valued convolutions, which can increase training time.