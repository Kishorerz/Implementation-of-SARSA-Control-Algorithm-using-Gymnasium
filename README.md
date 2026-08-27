# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the **SARSA (State-Action-Reward-State-Action)** reinforcement learning control algorithm using the Gymnasium `FrozenLake-v1` environment.

The agent must learn an optimal policy by interacting with the environment and updating its Q-values based on the action actually selected in the next state.

The learned Q-table, state-value function, policy, average reward, and learning curve are used to evaluate the performance of the agent.

---

## Software Requirements
The environment contains a **4 × 4 grid**:

```text
S F F F
F H F H
F F F H
H F F G
```

Where:

| Symbol | Meaning             |
| ------ | ------------------- |
| `S`    | Starting state      |
| `F`    | Frozen/safe surface |
| `H`    | Hole                |
| `G`    | Goal                |

The environment contains:

* **16 states**
* **4 possible actions**

The actions are:

| Action | Meaning |
| ------ | ------- |
| `0`    | Left    |
| `1`    | Down    |
| `2`    | Right   |
| `3`    | Up      |

For this experiment, `is_slippery=False` is used so that the environment is deterministic and the learning behaviour is easier to observe and reproduce.

The reward structure is:

* `0` for a normal movement
* `0` when the agent falls into a hole
* `1` when the agent reaches the goal

---
## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


---

## Algorithm

1. Create the `FrozenLake-v1` environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate `α`.
4. Set the discount factor `γ`.
5. Initialize epsilon for the epsilon-greedy policy.
6. For every training episode:

   * Reset the environment.
   * Select the initial action using the epsilon-greedy policy.
7. For every step:

   * Execute the selected action.
   * Observe the next state and reward.
   * If the episode has terminated, update the Q-value using the reward.
   * Otherwise, select the next action using the epsilon-greedy policy.
   * Apply the SARSA update rule.
   * Move to the next state and action.
8. Decrease epsilon after each episode.
9. Repeat until all training episodes are completed.
10. Calculate the state-value function from the learned Q-table.
11. Extract the learned policy using the action with the highest Q-value for every state.
12. Calculate the average reward over the last 1000 episodes.
13. Plot the learning curve.

---


## Python Program

```python
# -------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------------------------

import gymnasium as gym
import numpy as np


# -------------------------------------------------
# CREATE CUSTOMIZED FROZENLAKE ENVIRONMENT
# -------------------------------------------------

# F = Frozen (Safe)
# H = Hole
# S = Custom Start State
# G = Custom Goal State

# Start state is at (0, 2)
# Goal state is at (3, 0)

custom_map = [
    "FFSF",
    "FHFH",
    "FFFF",
    "GFHF"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=True
)


# -------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99


# -------------------------------------------------
# VARIABLE EPSILON PARAMETERS
# -------------------------------------------------

# Epsilon is NOT fixed.
# It starts at epsilon_start and gradually decreases
# until it reaches epsilon_min.

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

epsilon = epsilon_start


# -------------------------------------------------
# INITIALIZE Q-TABLE
# -------------------------------------------------

# Rows    = Number of states
# Columns = Number of possible actions

Q = np.zeros(
    (
        env.observation_space.n,
        env.action_space.n
    )
)


# -------------------------------------------------
# EPSILON-GREEDY ACTION SELECTION
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration:
    # Select a random action with probability epsilon
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation:
    # Select the action with the highest Q-value
    else:
        return np.argmax(Q[state])


# -------------------------------------------------
# SARSA TRAINING
# -------------------------------------------------

episode_rewards = []

# Start epsilon from the initial value
epsilon = epsilon_start


for episode in range(num_episodes):

    # Reset the environment
    state, _ = env.reset()

    # Select the initial action using epsilon-greedy policy
    action = epsilon_greedy_action(state, epsilon)

    # Store total reward for the current episode
    total_reward = 0


    # ---------------------------------------------
    # RUN ONE EPISODE
    # ---------------------------------------------

    for step in range(max_steps_per_episode):

        # Take the selected action
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Check whether the episode has ended
        done = terminated or truncated


        # -----------------------------------------
        # IF EPISODE ENDS
        # -----------------------------------------

        if done:

            # Update the final Q-value
            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

            total_reward += reward

            break


        # -----------------------------------------
        # SELECT NEXT ACTION USING EPSILON-GREEDY
        # -----------------------------------------

        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )


        # -----------------------------------------
        # SARSA UPDATE RULE
        #
        # Q(s,a) = Q(s,a) + alpha *
        # [reward + gamma * Q(s',a') - Q(s,a)]
        # -----------------------------------------

        Q[state, action] += alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )


        # Move to the next state-action pair
        state = next_state
        action = next_action

        total_reward += reward


    # ---------------------------------------------
    # STORE EPISODE REWARD
    # ---------------------------------------------

    episode_rewards.append(total_reward)


    # ---------------------------------------------
    # VARIABLE EPSILON DECAY
    #
    # Epsilon decreases after every episode.
    # It will never become smaller than epsilon_min.
    # ---------------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# CALCULATE STATE-VALUE FUNCTION
# -------------------------------------------------

# Maximum Q-value for every state

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# EXTRACT LEARNED POLICY
# -------------------------------------------------

# Best action for every state

learned_policy = np.argmax(Q, axis=1)


# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

print("========================================")
print("SARSA TRAINING COMPLETED")
print("========================================")


print("\nCustom FrozenLake Map:")

for row in custom_map:
    print(row)


print("\nCustom Start State:")
print("(0, 2)")


print("\nCustom Goal State:")
print("(3, 0)")


print("\nFinal Epsilon Value:")
print(epsilon)


print("\nFinal Q-Table:")
print(Q)


print("\nEstimated State-Value Function:")
print(state_values.reshape(4, 4))


print("\nLearned Policy:")
print(learned_policy.reshape(4, 4))


# -------------------------------------------------
# CALCULATE AVERAGE REWARD
# -------------------------------------------------

average_reward = np.mean(
    episode_rewards[-1000:]
)


print("\nAverage Reward Over Last 1000 Episodes:")
print(average_reward)


# -------------------------------------------------
# CLOSE ENVIRONMENT
# -------------------------------------------------

env.close()





```
---

## Output


# Final Q-table:
<img width="218" height="286" alt="Screenshot 2026-08-27 160700" src="https://github.com/user-attachments/assets/6d8ec180-a889-41c6-b581-942694c58155" />





# Estimated State-Value Function:

<img width="252" height="104" alt="Screenshot 2026-08-27 160819" src="https://github.com/user-attachments/assets/c280e5ff-a4b5-421c-8275-343b193ee708" />




# Learned Policy:
<img width="163" height="88" alt="Screenshot 2026-08-27 160842" src="https://github.com/user-attachments/assets/fdf10bdf-9a31-45fd-b387-f297fe314f19" />




# Average reward over last 1000 episodes: 

<img width="366" height="23" alt="Screenshot 2026-08-27 160909" src="https://github.com/user-attachments/assets/b6b9dab1-a023-45c3-a271-0d84baf416c4" />

# Learning Curve for Gamma = 0.1
<img width="676" height="454" alt="Screenshot 2026-08-27 155130" src="https://github.com/user-attachments/assets/ad9d1785-8fbc-46e9-9997-1cd8c0a03dfe" /><br/>
# Learning Curve for Gamma = 0.001
<img width="664" height="456" alt="Screenshot 2026-08-27 161109" src="https://github.com/user-attachments/assets/f14e579d-4545-4137-9127-a6867f9be863" />



---

## Result
```text
The SARSA control algorithm was successfully implemented and trained on a customized
FrozenLake-v1 environment. The start state and goal state were changed from the
default environment positions. A variable epsilon value was used, starting from
epsilon_start and gradually decaying to epsilon_min during training. The agent learned
an action-value function and a policy for navigating the customized environment.


```

---

## Inference
```text

Using a decaying epsilon provides a balance between exploration and exploitation.
Initially, the agent explores different paths with a high epsilon value. As training
progresses, epsilon decreases and the agent increasingly exploits the learned Q-values.
Changing the default start and goal states also demonstrates that the SARSA algorithm
can learn policies for customized FrozenLake configurations rather than only the
default environment layout.

```
---

