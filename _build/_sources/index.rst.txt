.. _Home:

******************************
Hi 👋, I am Abhijeet Pendyala
******************************

.. image:: _static/profile_picture.jpg
   :align: right
   :height: 300px
   :alt: Abhijeet Pendyala at ECML conference


I am a machine learning engineer with a passion for learning, communication and execution. I like mindmaps, survey papers, and reading books.

This website serves multiple purposes:

* **Extended Curriculum Vitae**: A recap of my professional journey and accomplishments
* **ML Blog**: A collection of my bite-sized attempts at developing intuitive understanding of machine learning concepts
* **Personal Knowledge Base**: A centralized place for organizing my thoughts, insights, and continuous learning process

CV and Resume
*************

* Download my CV in PDF format: `Abhijeet Pendyala - CV.pdf <https://github.com/Pendu/Portfolio/blob/06d606a7d9821bce407e546517501a92042c7e3a/source/CV_aug_2025.pdf>`_

.. * Download my Resume in PDF format: `Abhijeet Pendyala - Resume.pdf <https://abhijeet-pendyala.github.io/cv/Abhijeet_Pendyala_Resume.pdf>`_ 

Interests
******************

* Reinforcement Learning
* Optimization
* Low code software development


Skills
******

.. note::

   This section provides a quick overview of my skills (not exhaustive) and tools that i have used.

.. list-table:: Leadership Skills
   :widths: 25 25 50
   :header-rows: 1

   * - Category
     - Leadership Skills
     - Description
   * - **Engineering Leadership**
     - Strategy, roadmapping, OKRs
     - Translate business goals into technical direction; create execution roadmaps and measurable outcomes
   * - **Team Management**
     - 1:1s, sprint rituals, performance reviews
     - Build high-trust teams, facilitate delivery ceremonies, remove blockers, and foster continuous improvement
   * - **Project Management**
     - Requirements, user story mapping, risk management
     - Drive projects from discovery to delivery with clear scope, milestones, and stakeholder alignment
   * - **Architecture & Quality**
     - Design docs, code reviews, testing strategy
     - Establish engineering standards, review designs, and ensure reliability via testing and observability


.. list-table:: Technical Skills (Depth & Scope)
   :widths: 20 35 15 30
   :header-rows: 1

   * - Area
     - Technologies
     - Proficiency
     - Notes (scope, examples)
   * - **Programming**
     - Python; C++
     - Expert; Advanced
     - Production-grade ML/DS code, CLI tools, data pipelines; performance-sensitive modules in C++
   * - **Reinforcement Learning**
     - PPO; curriculum learning; simulation
     - Advanced
     - Continuous-control problems; reward shaping; Monte Carlo based evaluation
   * - **Deep Learning**
     - PyTorch; TensorFlow/Keras
     - Expert; Advanced
     - Model design, training loops, mixed precision, experiment tracking, reproducibility
   * - **Classical ML**
     - scikit-learn; XGBoost; LightGBM
     - Expert; Expert; Advanced
     - Feature engineering, calibration, cross-validation, model diagnostics
   * - **Data & Analysis**
     - Pandas; NumPy; SciPy; Statsmodels
     - Expert
     - Data wrangling, statistical testing, time-series preprocessing
   * - **MLOps**
     - Weights & Biases; MLflow; DVC
     - Expert; Advanced; Advanced
     - Experiment tracking, model registry, data/model versioning, lineage
   * - **Packaging & CI/CD**
     - Poetry; pre-commit; GitHub Actions
     - Advanced
     - Build automation, quality gates, style/lint hooks, release workflows
   * - **Deployment & Runtime**
     - Docker; Kubernetes; FastAPI
     - Advanced
     - Containerized APIs, autoscaling, health checks, observability
   * - **Storage & Messaging**
     - PostgreSQL; MongoDB; (basics) Redis
     - Proficient
     - OLTP schemas, indexing, simple caching patterns
   * - **Visualization**
     - Matplotlib; Seaborn; Plotly; Panel-Holoviz
     - Proficient
     - Exploratory analysis dashboards and publication-ready figures
   * - **Documentation & Writing**
     - Sphinx; Markdown; LaTeX/Overleaf
     - Expert
     - API docs, architecture docs, technical reports and experiment write-ups


******************************************
Research & Publications
******************************************

My research focuses on developing robust and scalable machine learning solutions for complex industrial control problems. A primary area of my work has been applying advanced reinforcement learning (RL) techniques, such as curriculum learning and hybrid RL-planning architectures, to optimize resource allocation in real-world, high-throughput systems with stochastic dynamics and strict safety constraints.

.. container:: publication-card

    **Curriculum RL meets Monte Carlo Planning: Optimization of a Real World Container Management Problem**
    
    `A. Pendyala and T. Glasmachers, European conference for Machine Learning (ECML), 2025 (to appear) <https://arxiv.org/abs/2503.17194>`_
    
    This paper introduces a hybrid RL-planning method that uses a curriculum-learning trained agent and an offline-trained collision model to proactively avert safety-limit violations in a waste-sorting facility. The work provides actionable guidelines for designing safer and more efficient industrial systems.

.. container:: publication-card

    **Solving a Real-World Optimization Problem Using Proximal Policy Optimization with Curriculum Learning and Reward Engineering**
    
    `A. Pendyala, A. Atamna, and T. Glasmachers, European conference for Machine Learning (ECML), 2024 <https://arxiv.org/abs/2404.02577>`_
    
    We demonstrate a five-stage curriculum learning approach combined with meticulous reward engineering to train a PPO agent for a complex industrial control task. The method enables the agent to effectively balance competing objectives of operational safety, volume optimization, and resource usage, where a vanilla agent fails.

.. container:: publication-card

    **ContainerGym: A Real-World Reinforcement Learning Benchmark for Resource Allocation**
    
    `A. Pendyala, J. Dettmer, T. Glasmachers, and A. Atamna, International Conference on Machine Learning, Optimization, and Data Science (LOD), 2023 <https://arxiv.org/abs/2307.02991>`_
    
    This work introduces ContainerGym, an open-source RL benchmark environment derived directly from a real-world industrial resource allocation problem. It is designed to evaluate and highlight the limitations of state-of-the-art RL algorithms on challenges such as delayed rewards, stochastic dynamics, and resource scarcity.

.. container:: publication-card

    **Online Budgeted Stochastic Coordinate Ascent for Large-Scale Kernelized Dual Support Vector Machine Training**
    
    `S. Qaadan, A. Pendyala, M. Schüler, and T. Glasmachers, International Conference on Pattern Recognition Applications and Methods (ICPRAM), 2020 <https://link.springer.com/chapter/10.1007/978-3-030-41014-2_14>`_
    
    This paper presents a novel C++ algorithm that combines an adaptive coordinate frequency optimization with a budget maintenance strategy to accelerate the training of non-linear Support Vector Machines (SVMs) on large-scale datasets.

PhD Thesis
************

* **Title:** Optimizing Industrial Process through Reinforcement Learning
* **Supervisor:** Prof. Dr. Tobias Glasmachers
* **Abstract:**
  Optimizing complex industrial processes, particularly those involving resource allocation under uncertainty and strict constraints, presents significant challenges for traditional control methods. This thesis addresses the optimization of container management in a real-world, high-throughput waste sorting facility, a critical stage impacting overall efficiency and sustainability. The core task involves scheduling the emptying of multiple containers, which accumulate different materials at stochastic rates, into a limited number of shared processing units (PUs). This research proposes a progressively sophisticated set of techniques centered around Proximal Policy Optimization (PPO), including a multi-stage curriculum learning strategy and a novel hybrid RL-planning architecture that augments the agent with an offline-trained collision model. The findings demonstrate the potential of combining structured learning techniques with domain-specific models to develop safe, efficient, and scalable control solutions for challenging industrial applications.
* **Download full thesis draft:** `Phd_thesis_Aug_2025_draft.pdf <https://github.com/Pendu/Portfolio/blob/06d606a7d9821bce407e546517501a92042c7e3a/source/CV_aug_2025.pdf>`_


.. _contact:

Contact
*******


.. raw:: html

    <p>
        <a href="mailto:abhijeet.pendyala@gmail.com" target="_blank"> 
        abhijeet.pendyala@gmail.com
        </a>
        &nbsp;|&nbsp;
        <a href="https://www.linkedin.com/in/abhijeet-pendyala-3ba00942/" target="_blank"> 
        LinkedIn
        </a>
        &nbsp;|&nbsp;
        <a href="https://github.com/Pendu" target="_blank"> 
        GitHub
        </a>
        &nbsp;|&nbsp;
        <a href="https://www.researchgate.net/profile/Abhijeet_Pendyala" target="_blank"> 
        ResearchGate
        </a>
    </p>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Table of Contents

   ml_blog/ml_blog
   