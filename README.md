# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python

# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    iteration = 0

    while True:
        delta = 0

        for s in range(env.observation_space.n):
            v = V[s]
            new_v = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    new_v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        iteration += 1

        if delta < theta:
            break

    return V, iteration


# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------
def policy_improvement(env, V, policy, gamma=0.99):
    policy_stable = True
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    for s in range(n_states):
        old_action = np.argmax(policy[s])
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )

        best_action = np.argmax(action_values)

        policy[s] = np.eye(n_actions)[best_action]

        if old_action != best_action:
            policy_stable = False

    return policy, policy_stable

#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration(env, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.ones((n_states, n_actions)) / n_actions

    while True:
        V, iteration = policy_evaluation(env, policy, gamma, theta)

        policy, policy_stable = policy_improvement(
            env, V, policy, gamma
        )

        if policy_stable:
            break

    return policy, V




```

## Output

<img width="448" height="150" alt="image" src="https://github.com/user-attachments/assets/43462a1d-9d36-424a-9194-89363bedfb03" />

<img width="322" height="186" alt="image" src="https://github.com/user-attachments/assets/6f67fa23-548f-49be-9b86-4a1ee6074266" />

<img width="508" height="307" alt="image" src="https://github.com/user-attachments/assets/93ed2316-b3c4-4741-b4ea-1bab68bb9e23" />

## Result

```
The Policy Iteration algorithm was successfully implemented and executed using the Gymnasium FrozenLake-v1 environment. The algorithm performed policy evaluation and policy improvement repeatedly until the optimal policy was obtained and the policy became stable.

The optimal state-value function and optimal policy were obtained for discount factors γ = 0.99 and γ = 0.90. The state-value function represents the expected rewards from each state, while the optimal policy indicates the best action to take in each state to reach the goal while avoiding holes

```

## Inference
```
Thus, the Policy Iteration algorithm was successfully used to solve the FrozenLake environment. It was observed that repeated policy evaluation and policy improvement helped the agent learn an optimal policy. The experiment also demonstrated that changing the discount factor affects the state-value function and the agent's decision-making. Hence, Policy Iteration is an effective Dynamic Programming technique for finding the optimal policy in a finite Markov Decision Process.

```

