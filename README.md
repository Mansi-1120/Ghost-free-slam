# Ghost-Free SLAM 🚀
### Recovering Static Scene Geometry in Crowded Dynamic Environments

Ghost-Free SLAM investigates how dynamic objects influence localization accuracy in visual SLAM systems operating in crowded indoor environments.

Feature-based SLAM systems assume that most observed features belong to static scene structure. In real environments, moving people introduce dynamic features that corrupt data association and increase trajectory drift.

This project studies whether **instance-level dynamic segmentation combined with lightweight geometric reprojection** improves localization stability without modifying the SLAM backend.

Rather than proposing a new SLAM algorithm, the project provides a **controlled empirical analysis** of how dynamic filtering influences **Absolute Trajectory Error (ATE)** in dense RGB-D sequences.

---

## 🔬 System Pipeline

```
TUM RGB-D Dataset
        │
        ▼
Instance Segmentation (YOLO-based)
        │
        ▼
Dynamic Pixel Masking
        │
        ▼
ORB-SLAM3
        │
        ▼
Trajectory Estimation
        │
        ▼
Evaluation using EVO (ATE)
```

This modular pipeline allows us to isolate how segmentation-based dynamic filtering affects SLAM localization accuracy.

---

## 📁 Repository Structure

```
Ghost-free-slam/
│
├── segmentation/       # YOLO-based instance segmentation
├── masking/            # Remove dynamic pixels using segmentation masks
├── slam/               # ORB-SLAM3 execution scripts
├── evaluation/         # Trajectory evaluation using EVO
├── examples/           # Example outputs and visualizations
├── demos/              # Demo videos and qualitative comparisons
├── configs/            # Experiment configuration files
├── docs/               # Proposal, reports, and slides
│
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 👥 Team Responsibilities

| Member | Responsibility |
|--------|----------------|
| **Mansi Singh** | ORB-SLAM3 integration and SLAM experiment design |
| **Tianqin Fu** | SLAM experiments and trajectory analysis |
| **Bhoomika Monthy Rajashekar** | Instance segmentation pipeline |
| **Devinn Chi** | YOLO model configuration and mask generation |
| **Brendan Coyne** | Dataset preparation, experiment automation, and visualization |

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/Mansi-1120/Ghost-free-slam.git
cd Ghost-free-slam
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Experiments use the TUM RGB-D Dataset, a widely used benchmark for visual SLAM research.

🔗 Dataset Link: https://vision.in.tum.de/data/datasets/rgbd-dataset

Example sequences used in this project:

- `freiburg3_walking_xyz`
- `freiburg3_sitting_xyz`
- `freiburg3_walking_static`

Due to size constraints, datasets are not stored in this repository and are maintained on the Boston University SCC cluster.

---

## 🧪 Experimental Configurations

We evaluate three configurations:

**1️⃣ Baseline ORB-SLAM3**
- Standard ORB-SLAM3 with no dynamic filtering applied.

**2️⃣ Segmentation Masking Only**
- Dynamic pixels identified by instance segmentation are masked before feature extraction.

**3️⃣ Segmentation + Geometric Reprojection**
- Dynamic pixels are masked, and previously observed static geometry is recovered via depth-consistent reprojection.

This controlled comparison isolates the effect of dynamic filtering on localization accuracy.

---

## 🧠 Geometric Recovery Strategy

- When masked regions correspond to static areas previously observed, geometry is recovered using depth-consistent reprojection.
- Previously observed 3D points are transformed into the current frame using camera poses and depth measurements.
- Only depth-consistent points are retained to prevent introducing inconsistent geometry.
- If a region has never been observed before, no reconstruction is performed and those pixels remain excluded from feature extraction.

---

## 📈 Evaluation Metric

Localization performance is evaluated using **Absolute Trajectory Error (ATE)**.

ATE measures the difference between the estimated trajectory and the ground truth trajectory provided by the dataset.
Trajectory evaluation is performed using the **EVO toolkit**.

---

## 🖥 SCC Experiment Structure

Experiments are executed on the Boston University Shared Computing Cluster (SCC).

```
dynamic_slam/
│
├── dataset/            # TUM RGB-D sequences
├── logs/               # Experiment logs
├── results/            # Experiment outputs
├── trajectories/       # SLAM trajectory files
├── trained_models/     # Segmentation weights
│
└── Ghost-free-slam/    # GitHub repository (after this project)
```

---

## 🎯 Research Objective

This project aims to answer the following research question:

> **Does segmentation-based dynamic removal combined with minimal geometric reprojection reduce trajectory drift in crowded indoor RGB-D sequences?**

The focus is on measuring localization behavior rather than proposing a new SLAM formulation.

---

## 📚 References

- Campos et al., 2021 — ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM
- Mur-Artal & Tardós, 2017 — ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras
- Bescos et al., 2018 — DynaSLAM: Tracking, Mapping and Inpainting in Dynamic Scenes
- Yu et al., 2018 — DS-SLAM: A Semantic Visual SLAM Towards Dynamic Environments

---

## 👩‍💻 Maintainer

**Mansi Singh**
Lead Researcher
MS Robotics, Boston University

🔗 GitHub: https://github.com/Mansi-1120
