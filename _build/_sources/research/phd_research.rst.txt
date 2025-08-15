.. _PhDResearch:


=============
PhD Research
============= 

Under the supervision of **Professor Tobias Glasmachers** at Ruhr University Bochum, my research focused on developing an end-to-end deep reinforcement learning solution for a complex industrial control system. I designed a hybrid approach that combines a Proximal Policy Optimization (PPO) agent, trained with a sophisticated curriculum learning strategy, with a predictive model for operational safety. This work culminated in the creation of 'ContainerGym,' a new open-source benchmark for industrial AI, and provided key insights into managing resource bottlenecks and preventing system failures.

.. figure:: plant_layout.png
   :alt: Industrial plant layout showing container management system
   :align: center
   :width: 90%
   :figclass: align-center

   **Industrial Plant Layout: Schematic diagram of the container management system with 12 containers, conveyor belts, and a central processing unit (PU)**

**************************************************
Foundational Work: From Data to Problem Formulation
**************************************************

Before building any models, my first task was to understand the real-world industrial data. The challenge was to transform raw, noisy sensor data into a clean and structured representation that could be used to formulate a formal reinforcement learning problem.

**Methodology and Contributions:**

* **Data Preprocessing and EDA:** I started by performing an extensive exploratory data analysis (EDA) on raw MySQL dump files. This involved converting the data into pandas DataFrames, normalizing features, and visualizing key trends in container volumes and fill cycles.
* **Time-Series Analysis:** A core part of this work was analyzing the time-series nature of the container filling process.
    * I applied various **smoothening techniques** (including rolling average, weighted moving average, and more advanced libraries like ``tssmoothie``) to filter out sensor noise and extract clean fill cycles.
    * I studied the **stationarity** of these fill cycles using tests like KPSS and ADFuller, and analyzed autocorrelation to understand underlying patterns.
    * I experimented with a range of **forecasting models** (such as ARIMA, Prophet, and LSTMs) to predict fill rates. This helped in accurately modeling the environment's stochastic dynamics for simulation.
* **Problem Abstraction:** This foundational work allowed me to identify the key operational challenges, such as:
    * The **stochasticity** and **non-linearity** of material inflow.
    * The existence of multiple optimal emptying volumes.
    * The crucial trade-offs between throughput, energy usage, and safety.

This comprehensive data-centric approach was essential. It enabled me to move beyond theoretical assumptions and ground the entire research in a robust, real-world understanding of the system, directly informing the design of the benchmark environment.

.. _ContainerGym:

**************************************************
The Benchmark: ContainerGym
**************************************************

**The Challenge:**

Many existing reinforcement learning benchmarks, such as those from the gaming or robotics domains, fail to capture the unique complexities of real-world industrial systems. Key challenges like stochastic dynamics, extremely delayed and sparse rewards, and strict safety constraints are often absent, making it difficult to properly evaluate and develop algorithms for industrial deployment.

**My Solution:**

To address this gap, I designed and released **ContainerGym**, an open-source RL benchmark environment. It is a direct, minimally simplified digital twin of the container management problem from a high-throughput waste sorting facility. The environment is highly customizable, allowing for a variable number of containers, processing units, and reward functions.

**Key Features of ContainerGym:**

* **Stochastic Dynamics:** The environment incorporates a random walk model with drift and noisy sensor readings to accurately simulate the unpredictable material inflow.
* **Resource Allocation:** It models a scarce resource bottleneck where multiple containers compete for a limited number of processing units (PUs).
* **Complex Rewards:** The reward function is designed with multiple Gaussian peaks, encoding a trade-off between higher throughput (emptying at high volumes) and safety (emptying at lower volumes).
* **Strict Safety Constraints:** The environment imposes a severe penalty for container overflows, forcing agents to learn risk-averse, proactive policies.

**The Outcome:**

By using ContainerGym to benchmark standard RL algorithms like PPO, TRPO, and DQN, I was able to empirically demonstrate their primary limitations. The results showed that vanilla agents struggle to handle the delayed rewards and strategic foresight required by the task, which motivated the need for my subsequent research into more advanced techniques.

.. figure:: example.gif
   :alt: ContainerGym environment visualization during evaluation
   :align: center
   :width: 85%
   :figclass: align-center

   **ContainerGym Environment: Dynamic visualization showing the container management system in action during reinforcement learning evaluation**

.. _Advanced_RL_Solutions:

**************************************************
Advanced Reinforcement Learning Solutions
**************************************************

The limitations of baseline algorithms on the ContainerGym environment motivated my core research contributions: the development of advanced reinforcement learning techniques tailored for complex industrial problems.

**Curriculum Learning & Reward Engineering**

**The Challenge:** Standard RL agents failed to learn a robust policy because of the environment's inherent complexities, such as extremely delayed and sparse rewards, and the need to balance multiple competing objectives. A naive agent would often converge on a "reward-hacking" strategy of frequent, low-value empties instead of learning a long-term optimal policy.

**My Solution:** I developed a novel, multi-stage **curriculum learning** strategy combined with custom **reward engineering**. The agent was progressively exposed to a series of tasks, starting with a simplified, deterministic environment and gradually introducing real-world complexities like stochastic dynamics and resource constraints. I also designed a custom multi-component reward function that provided a smooth, Gaussian-shaped signal to guide learning and a penalization mechanism to counteract reward hacking.

**The Outcome:** This methodical approach enabled the PPO agent to learn a robust and efficient policy that achieved near-zero safety violations, significantly outperformed a naive agent, and demonstrated a superior balance between throughput and energy efficiency.

**Hybrid RL-Planning for Safety**

**The Challenge:** Even with a sophisticated curriculum, a purely reactive RL agent can be myopic, failing to anticipate future conflicts for the shared processing units (PUs) that could lead to costly, safety-critical overflows.

**My Solution:** I proposed and validated a **hybrid control architecture** that augments the curriculum-trained RL policy with an offline-trained **collision model**. This model, an XGBoost classifier, was trained on a large dataset of simulated pairwise collision scenarios generated via Monte Carlo rollouts. During inference, this model acts as a "proactive safety layer," overriding the agent's "do nothing" action if it predicts a high probability of an imminent collision.

**The Outcome:** This hybrid approach significantly reduced collision events and safety-limit violations, particularly in high-contention scenarios (e.g., a single PU managing 7-12 containers). The framework not only yielded a safer controller but also provided a tool for system-level analysis, offering actionable guidelines for facility design regarding the optimal ratio of containers to PUs.

**************************************************
Technical & Implementation Contributions
**************************************************

Beyond the core algorithms, a significant part of my research involved the hands-on engineering and technical implementation of these systems.

* **High-Performance Training:** I optimized the RL training loop by using vectorized environments (`Dummyvecenv`, `Subprocvecenv`), employing `JAX` with `Stable-Baselines3` for faster PPO execution, and conducting extensive profiling to identify and resolve bottlenecks.
* **Robust Agent Design:** My work involved implementing and evaluating crucial RL components, including **action masking** to handle dynamic and constrained action spaces and performing hyperparameter tuning for PPO to ensure stable and optimal performance.
* **Advanced Modeling:** I explored and implemented alternative model architectures like **Mixture Density Networks** for the agent's policy and investigated **Behavioral Cloning** to deal with catastrophic forgetting during curriculum transitions.
* **Code Quality & Maintenance:** I was responsible for refactoring and maintaining the core simulation environment code, ensuring its robustness, reusability, and reproducibility for future research.

**************************************************
Mindmap
**************************************************

.. raw:: html

    <div style="max-width: 900px;">
      <iframe width="100%" height="500" style="border: 1px solid #d0d0d0; border-radius: 6px;" src="https://www.mindomo.com/mindmap/sutco-project-my-contributions-192efed34fa7252681d8849939341d47" frameborder="0" allowfullscreen></iframe>
    </div>


.. _Phd Thesis:

****************************
Thesis Abstract and Download
****************************

**Title:** Optimizing Industrial Process through Reinforcement Learning
**Supervisor:** Prof. Dr. Tobias Glasmachers

**Abstract:**

Optimizing complex industrial processes, particularly those involving resource allocation under uncertainty and strict constraints, presents significant challenges for traditional control methods. This thesis addresses the optimization of container management in a real-world, high-throughput waste sorting facility, a critical stage impacting overall efficiency and sustainability. The core task involves scheduling the emptying of multiple containers, which accumulate different materials at stochastic rates, into a limited number of shared processing units (PUs). This research proposes a progressively sophisticated set of techniques centered around Proximal Policy Optimization (PPO), including a multi-stage curriculum learning strategy and a novel hybrid RL-planning architecture that augments the agent with an offline-trained collision model. The findings demonstrate the potential of combining structured learning techniques with domain-specific models to develop safe, efficient, and scalable control solutions for challenging industrial applications.

**Download full thesis draft:** `Phd_thesis_Aug_2025_draft.pdf <https://github.com/Pendu/Portfolio/blob/06d606a7d9821bce407e546517501a92042c7e3a/source/CV_aug_2025.pdf>`_

