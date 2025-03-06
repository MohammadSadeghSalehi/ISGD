# **Bilevel Learning with Inexact Stochastic Gradients**  

This repository contains the implementation of the inexact stochastic bilevel optimization method presented in the paper:  

**[Bilevel Learning with Inexact Stochastic Gradients](https://arxiv.org/abs/2412.12049)**  
Mohammad Sadegh Salehi, Subhadip Mukherjee, Lindon Roberts, Matthias J. Ehrhardt  

Bilevel learning plays a crucial role in machine learning, inverse problems, and imaging applications, such as hyperparameter optimization, learning data-adaptive regularizers, and optimizing forward operators. Our work introduces an inexact stochastic bilevel optimization framework with strongly convex lower-level problems and a nonconvex sum-of-functions in the upper level. We establish connections to stochastic optimization theory, ensuring convergence under mild assumptions. The proposed method improves efficiency and generalization in imaging tasks like image denoising and deblurring compared to existing adaptive deterministic bilevel methods.  

---

## **Paper Reference**  

[![Bilevel Learning with Inexact Stochastic Gradients](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2412.12049)  
**[Bilevel Learning with Inexact Stochastic Gradients](https://arxiv.org/pdf/2412.12049.pdf)**  

- [Mohammad Sadegh Salehi](https://scholar.google.com/citations?user=bunZmJsAAAAJ&hl=en)  
- [Subhadip Mukherjee](https://scholar.google.com/citations?user=subhadip)  
- [Lindon Roberts](https://scholar.google.com/citations?user=lindon)  
- [Matthias J. Ehrhardt](https://scholar.google.com/citations?user=matthias)  

---

## **Description**  

This repository implements the proposed inexact stochastic bilevel optimization framework, focusing on practical applications such as:  

- Learning data-adaptive regularizers  
- Hyperparameter optimization  
- Optimizing forward operators for imaging applications (denoising, deblurring)  

### **Key highlights of our approach:**  
✅ Inexact stochastic hypergradients for improved efficiency  
✅ Strongly convex lower-level with a nonconvex upper-level sum-of-functions  
✅ Theoretical convergence guarantees  
✅ Faster training and better generalization than deterministic bilevel methods  

---

## **Installation**  

### **1. Clone the repository**  
```bash
git clone https://github.com/MohammadSadeghSalehi/Inexact-Bilevel-Optimization.git
cd Inexact-Bilevel-Optimization
