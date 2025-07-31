
---

# 🛡️PrivacyBench  
**PPML-Benchmark: A Standardized Evaluation Framework for Privacy-Preserving Machine Learning (PPML)**

## 📌 Overview  

PrivacyBench is a benchmarking framework that evaluates **Privacy-Preserving Machine Learning (PPML)** techniques by quantifying their trade-offs in:

- ✅ Model Utility (Accuracy Loss)  
- ⏱️ Computational Cost (Training Time, Memory Usage)  
- ⚡ Energy Consumption (kWh, CO₂ Emissions)  

This study uses image-based deep learning models to analyze how privacy techniques affect **performance and sustainability**.

---

## 🧪 Dataset  

- **Dataset:** Alzheimer MRI Image Dataset  
- **Task:** Multi-class classification (4 classes)  
- **Baseline Models:** CNN (ResNet18) and ViT (Vision Transformer)

---

## 🧠 Research Theme  

### 1️⃣ Research Topic  
Benchmarking trade-offs in Privacy-Preserving Machine Learning (PPML) across utility, computational cost, and sustainability metrics.

### 2️⃣ Research Title  
**"PrivacyBench: Benchmarking Utility Loss, Computational Costs, and Energy Consumption in Privacy-Preserving Machine Learning"**

### 3️⃣ Research Questions  
#### 🔍 Main Question:  
How do different PPML techniques affect model utility, computational efficiency, and energy use?

#### 🛠 Sub-Questions:
1. **Privacy vs. Utility:**  
   - How much accuracy loss is introduced by FL, DP, HE, and SMPC?

2. **Computational Cost:**  
   - What’s the impact on training time and memory usage?

3. **Energy & Carbon Footprint:**  
   - How do CO₂ emissions and energy usage differ across techniques?

4. **Hybrid Privacy Techniques:**  
   - Which combinations (e.g., FL+DP, FL+SMPC) offer the best trade-offs?

---

## 🎯 Research Aim & Objectives  

### 📌 Aim  
To build a reproducible benchmark for evaluating privacy, performance, and sustainability in deep learning.

### 📌 Objectives  
- ✅ Develop a benchmarking framework  
- ✅ Measure accuracy loss vs. baseline  
- ✅ Track training time and memory  
- ✅ Log energy and CO₂ emissions using CodeCarbon  
- ✅ Evaluate hybrid privacy techniques

---

## 📊 Experiment Design  

### ✅ Techniques Evaluated:
- **Federated Learning (FL)**
- **Differential Privacy (DP)**
- **Homomorphic Encryption (HE)**
- **Secure Multi-Party Computation (SMPC)**
- **Combinations (e.g., FL+DP, FL+DP+SMPC)**

---

## 🔬 Finalized PPML Experiments  

| S/N | Experiment ID             | Model Type | PPML Technique           | Status         |
|-----|---------------------------|------------|--------------------------|----------------|
|  1  | CNN Baseline              | CNN        | No Privacy               | ✅ Done        |
|  2  | ViT Baseline              | ViT        | No Privacy               | ✅ Done        |
|  3  | Federated Learning (CNN)  | CNN        | FL                       | ✅ Done        |
|  4  | Federated Learning (ViT)  | ViT        | FL                       | ✅ Done |
|  5  | Differential Privacy (CNN)| CNN        | DP                       | 🔁 In Progress |
|  6  | Differential Privacy (ViT)| ViT        | DP                       | ✅ Done     |
|  7  | SMPC (CNN)                | CNN        | SMPC                     | ✅ Done     |
|  8  | SMPC (ViT)                | ViT        | SMPC                     | ❌ Pending     |
|  9  | FL + DP (CNN)             | CNN        | FL + DP                  | ❌ Pending     |
| 10  | FL + DP (ViT)             | ViT        | FL + DP                  | ❌ Pending     |
| 11  | FL + SMPC (CNN)           | CNN        | FL + SMPC                | ❌ Pending     |
| 12  | FL + SMPC (ViT)           | ViT        | FL + SMPC                | ❌ Pending     |
| 13  | DP + SMPC (CNN)           | CNN        | DP + SMPC                | ❌ Pending     |
| 14  | DP + SMPC (ViT)           | ViT        | DP + SMPC                | ❌ Pending     |
| 15  | FL + DP + SMPC (CNN)      | CNN        | FL + DP + SMPC           | ❌ Pending     |
| 16  | FL + DP + SMPC (ViT)      | ViT        | FL + DP + SMPC           | ❌ Pending     |

---

## 📍 Progress Tracker

To track the current progress of this project, check [PrivacyBench Experiment Tracking](https://docs.google.com/spreadsheets/d/1ZY0F4-PTpOnUXk4_5Udo9d7Qf5ZkiN33v1DMfP6qgew/edit?usp=sharing)


---

## 🗓 Timeline  
**February 2025 → May 2025**  
- Experimentation, Logging, Benchmarking  
- Target Venue: **NeurIPS 2025 Main Conference**

---
