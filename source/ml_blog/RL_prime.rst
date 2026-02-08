.. _rl_primer:

===============================================================================
The Architecture of Agency: Unifying Planning and Reinforcement Learning
===============================================================================

.. contents:: Table of Contents
   :depth: 3
   :local:

*A theoretical deconstruction of optimal control, based on the principle that Planning and RL are not separate fields, but distinct regions on a continuous spectrum of information access.*

The problem of optimal decision-making under uncertainty is the dark matter of artificial intelligence. It pulls at the orbits of distinct fields — Control Theory, Operations Research, and Machine Learning — yet these disciplines often speak different languages. A control theorist might speak of "Hamilton-Jacobi-Bellman equations," while an RL researcher talks of "Q-learning," despite both solving the same underlying mathematical structure.

This post aims to strip away the terminological divergence and deconstruct the field into its first principles. We will look at the problem through the lens of the Markov Decision Process (MDP), not just as a formalism, but as a *map*. By classifying algorithms based on their *access* to this map and the *scope* of their solutions, we can build a unified taxonomy that connects classical Planning to modern Deep Reinforcement Learning.


The Substrate: The Markov Decision Process
==============================================

At the heart of the problem lies the **agent-environment loop** [Sutton & Barto, 2018]. At every timestep :math:`t`, an agent observes a state :math:`s_t`, selects an action :math:`a_t`, and receives a reward :math:`r_{t+1}` and a new state :math:`s_{t+1}`. The goal is simple but computationally explosive: construct a strategy (a **policy**, :math:`\pi`) that maximizes the expected sum of future rewards.

This structure relies on a single, powerful assumption: the **Markov Property**. A state :math:`s_t` is Markov if it is a "sufficient statistic" of history — meaning the future depends only on the present, not the past.

.. math::

   \Pr(S_{t+1} = s' \mid S_t, A_t) = \Pr(S_{t+1} = s' \mid H_t, A_t)

If this holds, we are operating within a **Markov Decision Process (MDP)**, defined by the tuple :math:`\langle \mathcal{S}, \mathcal{A}, p, r, \gamma \rangle`.

The Taxonomy of Access
----------------------

How do we solve an MDP? The answer depends entirely on *what information we are allowed to touch*. Following the framework proposed by Moerland et al. (2022), we can classify every algorithm along two axes.

**Key Insight: The Access-Scope Matrix**

* **Access** (Reversible vs. Irreversible): Can you "teleport" and query the model (*Planning*), or are you stuck in the arrow of time (*RL*)?
* **Scope** (Local vs. Global): Do you solve for *now* (*Search*), or for *everywhere* (*Learning*)?

We can map the entire field of Control onto these quadrants:

.. list-table:: The Access-Scope Matrix
   :widths: 25 35 35
   :header-rows: 1
   :stub-columns: 1

   * -
     - **Local Scope** (Solve for :math:`s_t`)
     - **Global Scope** (Solve for :math:`\forall s`)
   * - **Reversible Access** (Has Model)
     - **Planning** (e.g., A*, MCTS)
     - **Model-Based RL** (e.g., AlphaZero, Dyna)
   * - **Irreversible Access** (No Model)
     - N/A (Requires interaction)
     - **Model-Free RL** (e.g., PPO, DQN)

This matrix is the conceptual backbone of the entire field. Every algorithm you encounter lives in one of these quadrants.



The Objective and the Philosophy of Discounting
===================================================

The agent's goal is to maximize the expected return :math:`J(\pi)`. But defining "return" requires a nuance: the **Discount Factor**, :math:`\gamma \in [0, 1]`.

.. math::

   G_t \doteq \sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1}

While :math:`\gamma` is often treated as a mathematical trick to ensure infinite sums converge (for :math:`\gamma < 1`), it is actually a profound **behavioral lever**:

* :math:`\gamma \to 0` **(Myopia):** The agent cares only about the immediate moment.
* :math:`\gamma \to 1` **(Foresight):** The agent becomes far-sighted, willing to sacrifice short-term gains for long-term payoffs.

This introduces a fundamental trade-off: Higher :math:`\gamma` allows for complex, long-horizon strategies, but it drastically increases the **variance** of the learning signal, as the return now depends on highly uncertain future events.



The Mechanics: Value and Bellman Equations
=============================================

To optimize the policy, we need a way to keep score. We do this via **Value Functions** (:math:`V^\pi(s)` and :math:`Q^\pi(s,a)`), which measure the long-term desirability of states. These functions are governed by the recursive relationship known as **Bellman's Principle of Optimality**:

.. admonition:: Bellman's Principle

   *An optimal path is composed of optimal sub-paths.*

This recursion gives us two distinct equations.

The Bellman Expectation Equation
--------------------------------

The **Bellman Expectation Equation** evaluates a fixed policy:

.. math::

   V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \, V^\pi(s') \right]

This equation is linear — it is simply a weighted average that can be solved via matrix inversion.

The Bellman Optimality Equation
-------------------------------

However, optimal control requires us to find the *best* action. We replace the expectation over actions with a maximization, yielding the **Bellman Optimality Equation**:

.. math::

   V_*(s) = \max_{a} \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \, V_*(s') \right]

.. important::

   **The Crucial Insight:** This equation introduces two complexities:

   1. **Simultaneity:** Every state's value depends on every other state's value, creating a complex web of dependencies.
   2. **Non-Linearity:** The :math:`\max` operator breaks the linear structure. We can no longer simply "solve" the equation using linear algebra; we must find the solution **iteratively**.

The Engine: Generalized Policy Iteration (GPI)
==================================================

Almost all RL and Planning algorithms are instantiations of a single meta-algorithm: **Generalized Policy Iteration (GPI)** [Sutton & Barto, 2018]. This is the engine that drives learning, composed of two interacting processes:

1. **Policy Evaluation:** Estimate the value function :math:`V^\pi` for the current policy.
2. **Policy Improvement:** Generate a better policy :math:`\pi'` by acting greedily with respect to :math:`V^\pi`.

.. note::

   **The GPI Cycle:** The policy :math:`\pi` and the value function :math:`V` interact in a loop. The policy is *evaluated* to produce a value function, and the value function is used to *improve* the policy (by acting greedily with respect to :math:`V`). This cycle converges to the optimal policy :math:`\pi_*` and the optimal value function :math:`V_*`.

The **Policy Improvement Theorem** guarantees that if we act greedily with respect to the true values of our current policy, the new policy is strictly better (or equal, if optimal). The algorithms below are just different ways to turn this crank.



The Blockade: The Three Curses of Dynamic Programming
========================================================

If we have a perfect model (Reversible Access), we can use **Dynamic Programming** (DP) (e.g., Value Iteration) to run GPI directly. But in the real world, DP fails due to three fatal "Curses":

1. **The Curse of Knowledge:** We rarely have the transition model :math:`p(s'|s,a)`.
2. **The Curse of Dimensionality:** DP requires a "full-width backup" — summing over *all* possible future states [Bellman, 1957]. In high dimensions, this is computationally impossible.
3. **The Curse of Modeling:** DP uses tables. It cannot generalize. If it learns state A is bad, it knows *nothing* about state A', even if they are nearly identical.

.. list-table:: Breaking the Three Curses
   :widths: 30 35 35
   :header-rows: 1

   * - Curse
     - Problem
     - Solution
   * - **Knowledge**
     - No access to :math:`p(s'|s,a)`
     - **Sampling** (learn from interaction)
   * - **Dimensionality**
     - Full-width backup is intractable
     - **Sampled Backups** (follow single trajectories)
   * - **Modeling**
     - Tables cannot generalize
     - **Function Approximation** (neural networks)



The Evolution: Model-Free Learning
======================================

Reinforcement Learning is the paradigm designed to break these curses using **Sampling** (to kill the Knowledge Curse) and **Function Approximation** (to kill the Modeling Curse).

The Mechanics of Time: MC vs. TD
---------------------------------

How do we estimate value without a model? We shift from "Full Width" lookups to "Sampled" paths.

.. note::

   **DP vs. RL Backups:**

   * **Dynamic Programming (Full-Width):** From a state :math:`s`, the algorithm fans out to *all* possible successor states :math:`s'`, weighting each by its transition probability. This requires complete knowledge of the model.
   * **Reinforcement Learning (Sampled):** From a state :math:`s`, the algorithm follows a *single* sampled trajectory :math:`s \to s' \to s'' \to \ldots`, learning only from the path it actually experiences.

**Monte Carlo (MC):** Wait until the end of the episode, then use the actual total return :math:`G_t` to update the value.

* **Pros:** Unbiased (it uses real data).
* **Cons:** High Variance (outcomes vary wildly).

**Temporal Difference (TD):** Update the current guess based on *another guess* [Sutton, 1988].

.. math::

   V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma \, V(S_{t+1}) - V(S_t) \right]

This is **Bootstrapping**. It introduces bias (our guess might be wrong) but drastically reduces variance and allows for online learning.

This is not a binary choice but a **spectrum**. Methods like :math:`n`-step returns and :math:`\text{TD}(\lambda)` blend these approaches, balancing the bias of bootstrapping with the variance of full returns.

The Strategy: On-Policy vs. Off-Policy
---------------------------------------

How do we explore while trying to optimize?

* **On-Policy (SARSA):** The agent learns the value of the policy it *is actually executing*. It is "conservative," learning about the danger it actually faces.
* **Off-Policy (Q-Learning):** The agent learns the value of the *optimal* policy, regardless of what it is currently doing [Watkins, 1989].

.. math::

   Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]

This allows for learning from historical data, but is notoriously unstable when combined with approximation.



The Deep Synthesis
=====================

The final step is to scale these ideas to high-dimensional spaces (images, robots) using **Neural Networks**.

Deep Q-Networks (DQN)
---------------------

DQN [Mnih et al., 2015] combined Q-Learning with Deep Learning. However, this combination is notoriously unstable, known as **the Deadly Triad**:

1. **Bootstrapping** (TD updates)
2. **Off-Policy Learning** (Sampling from a buffer)
3. **Function Approximation** (Neural Networks)

When combined, these three elements can cause value estimates to **diverge**. DQN solved this with architectural hacks:

* **Experience Replay** — to break correlations in sequential data.
* **Target Networks** — to stabilize the moving target.

However, DQN eventually hits a wall. In **continuous action spaces** (like robotics), finding :math:`\arg\max_a Q(s,a)` is computationally intractable. In effect, the Curse of Modeling returns — this time prohibiting us from *representing* the optimal action.

The Actor-Critic Architecture
-----------------------------

To solve this, we return to **Policy-Based methods** (like REINFORCE [Williams, 1992]), which parameterize the policy directly. This handles continuous actions easily but suffers from extreme variance.

The solution is the **Actor-Critic architecture** [Konda & Tsitsiklis, 2000], which synthesizes the strengths of both paradigms:

* **The Actor** (:math:`\pi_\theta`): Represents the policy directly, handling continuous actions.
* **The Critic** (:math:`V_\omega`): Estimates value to provide a **Baseline** for the Actor.

By subtracting the baseline :math:`V(s)` from the action value :math:`Q(s,a)`, we calculate the **Advantage**:

.. math::

   A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)

This tells us not just *if* a state is good, but **how much better** a specific action is than the average behavior in that state.

The Final Hurdle: Stability and PPO
------------------------------------

While Actor-Critic solves the variance problem, it introduces a new one: **Step-Size Instability**. Because we learn on-policy, a single bad update that collapses the policy performance cannot be easily recovered from — the agent generates bad data, which leads to worse updates.

This motivates **Trust Region methods**, which constrain the update to ensure the new policy :math:`\pi_{\text{new}}` does not deviate too far from :math:`\pi_{\text{old}}`. The state-of-the-art implementation of this idea is **Proximal Policy Optimization (PPO)** [Schulman et al., 2017]. By "clipping" the objective function, PPO ensures safe, stable updates without the complexity of second-order optimization.

This architecture — an Actor-Critic setup guarded by PPO's clipped objective — represents the current culmination of the field. It is the robust answer to the three curses, balancing bias, variance, and stability to solve the problem of agency in complex worlds.



References
==========

* Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
* Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-Critic Algorithms.
* Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
* Moerland, T. M., et al. (2022). A Unifying Framework for Reinforcement Learning and Planning.
* Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv*.
* Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*.
* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
* Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis.
* Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*.
