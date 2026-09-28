# 2. Classical AI: search, reasoning and decisions

**Goal:** find a route on a small map and see why following known rules is different from learning rules from data. Lab: [shortest-path search](../notebooks/01-foundations/search.ipynb).

## First, find a way home

Imagine three possible trips from School (S) to Home (G):

```text
S -> A -> G       fares: 1 + 9 = 10
S -> B -> C -> G  fares: 2 + 2 + 2 = 6
```

Which trip has fewer stops? **Through A.** Which is cheaper? **Through B and C.** We know the roads and prices already. A computer does not need to "learn" those prices from receipts; it needs to **search** the routes. A _state_ is simply where we are now, an _action_ is taking a road, and a _goal_ is getting home. The rules that say where each road leads are the _transition model_.

Start with a queue of places to visit. Breadth-first search (BFS) checks all one-road options before two-road options, then three-road options. It reaches Home through A first: two roads. Uniform-cost search instead checks the cheapest _total fare so far_, reaching Home via B and C at cost 6. **Pause:** If the fare from A to G falls from 9 to 1, which route should the cheapest-route method choose? Answer: S-A-G costs 2.

An agent observes a state, selects an action and may pursue a goal. In a map, a state is a city, an action follows a road, and a goal is a destination. A _transition model_ says which city and cost follow each action. This is classical search: no fitted weights are needed. Machine learning is useful when the transition, scoring rule or outcome must instead be estimated from examples.

## Later: search and planning in more detail

| Method                  | Chooses next            | Guarantee and caveat                                                                                               |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| BFS                     | Shallowest frontier     | Shortest path in number of edges for equal-cost edges; memory can grow rapidly                                     |
| DFS                     | Deepest frontier        | Low memory in many implementations, but may miss a short route or loop without visited checks                      |
| Uniform-cost / Dijkstra | Lowest path cost $g(n)$ | Cheapest nonnegative-cost route; ignores proximity to goal                                                         |
| A\*                     | Lowest $f(n)=g(n)+h(n)$ | Optimal with admissible heuristic $h$ (never overestimates remaining cost), under standard graph-search conditions |

Example: `Start -> A -> Goal` costs $1+9=10$ and `Start -> B -> C -> Goal` costs $2+2+2=6$. BFS prefers the two-edge route, while uniform-cost search prefers cost 6. An admissible heuristic such as straight-line distance can make A* inspect fewer states, but may give little advantage if it is always zero (then A* behaves like uniform-cost search).

Planning sequences actions toward goals while respecting preconditions and effects (e.g., a robot can pick up an object only when its gripper is free). Constraint-satisfaction problems instead assign variables so constraints all hold, such as a class schedule with no conflicting rooms. Backtracking plus constraint propagation can prune impossible partial assignments; a learned heuristic can rank choices without replacing the hard constraint check.

### When another agent is trying to stop you

Route finding assumes the world is not actively choosing against us. In chess, board games and security simulations, each action changes the position and an opponent chooses a response. **Minimax** explores alternating moves: choose the move with the highest score assuming the opponent chooses the reply with the lowest score. The score is a utility at a terminal position, or an estimate from a cutoff position when the full tree is too large. **Alpha-beta pruning** skips branches that cannot change the final choice; it gives the same answer as minimax with less search, but good move ordering matters.

This is different from BFS: a game tree contains choices for several decision-makers, and a locally good move can be bad after the opponent's reply. A depth limit, evaluation function and tie-breaking rule therefore become part of the system's behavior. In a stochastic game, chance nodes add expected values; in a game with hidden information, the agent must reason over possible states rather than one known board.

### State assumptions matter

Before choosing an algorithm, ask whether the state is fully observable, whether actions are deterministic, whether the world changes while the agent thinks, and whether the cost or reward is known. A route map with fixed fares is a simpler problem than a partially observed robot with traffic and delayed sensors. The usual search guarantees depend on assumptions such as nonnegative costs, a correct transition model and enough memory to record visited states. When those assumptions fail, use belief states, replanning, probability or a robust fallback rather than quietly treating guesses as facts.

## Everyday rules and uncertainty

A timetable rule such as "if the shop is closed, do not enter" is exact _within that timetable_. Real life is less certain: "if clouds appear, take an umbrella" is a useful guideline, not a promise of rain. We can use logic for hard rules and probabilities when evidence is uncertain. Neither requires a giant language model.

Propositional logic expresses true/false statements; first-order logic adds objects and relationships. A rule `wet -> slippery` can support inference, but a wet road is not _certainly_ slippery. Bayesian networks represent conditional dependencies; probabilities help reason with incomplete evidence. Carefully distinguish a symbolic rule, a probability estimate, and a learned embedding: each has different error modes and auditability.

## From one choice to many choices

A navigation app cannot choose the first road only by its immediate fare; that road changes the choices available later. The same is true in a game: moving toward a reward might put you in danger afterward. **Reward** is a score for a result, a **policy** is a rule for choosing actions, and **reinforcement learning** (RL) improves such a rule by observing what happens after actions. Only after that intuition do we need the mathematical description below.

### Later: formal RL vocabulary

A Markov decision process consists of states $s$, actions $a$, transition probabilities $P(s'\mid s,a)$, rewards $r$ and discount $\gamma$. The value of a policy $\pi$ is expected discounted reward $E_\pi[\sum_{t=0}^{\infty}\gamma^t r_t]$. The Bellman equation decomposes future value: $V^\pi(s)=E[r+\gamma V^\pi(s')]$. Dynamic programming uses a known transition model; reinforcement learning estimates behavior from interaction. Q-learning, policy gradients and actor-critic methods differ in what they learn and how they update it. Exploration changes the data you observe, so offline evaluation and safe deployment are unusually difficult.

Modern LLM agents often combine these ideas: an LLM suggests actions, deterministic code checks permissions, and a search or graph workflow manages state. Calling every automated pipeline an autonomous agent obscures where decisions actually occur.

## Practice

1. Which search algorithm would you use to minimize total train fare with nonnegative ticket prices? **Uniform-cost search**.
2. Is Manhattan distance admissible for an unobstructed four-direction unit-cost grid? **Yes**, because obstacles can only increase path length.
3. Why cannot a high reward from one action establish an optimal policy? **Future transitions and exploration matter**.
4. In a two-player game, why is choosing the move with the best immediate score unsafe? **The opponent can choose a reply that changes the outcome**.
5. Name one question to ask before trusting a search guarantee. **For example: are action costs nonnegative, is the transition model correct, or is the state fully observable?**

Next: [ML](03-machine-learning.md), then [Agents and MCP](07-agents-mcp.md).
