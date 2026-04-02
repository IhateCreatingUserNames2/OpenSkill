---
name: Combinatorial Probability with Grouped Draws Without Replacement
domain: Mathematics / Probability / Combinatorics
generated: 2026-04-02 09:01 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Combinatorial Probability with Grouped Draws Without Replacement

> How to correctly compute the probability of drawing a specific composition of items from a grouped population without replacement, using combinations, the multiplication principle, fraction simplification, and exhaustive verification.

## When to Apply

Apply whenever a problem asks for the probability of drawing a specific number of items from each of two or more distinct subgroups within a finite population, where draws are made without replacement and order does not matter.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Draws without replacement from an unordered bag are modeled by combinations, not permutations.
- Total outcomes = C(n, k) where n is total population size and k is draw size.
- Favorable outcomes involving multiple distinct groups = product of individual group combination counts.
- The final probability fraction must be reduced to lowest terms by dividing by GCD.
- All mutually exclusive exhaustive cases must sum to the total outcome count (or probabilities sum to 1) as a sanity check.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Using permutations P(n,k) instead of combinations C(n,k) for unordered draws.
- ⚠️ Adding group combination counts instead of multiplying them when combining independent selections.
- ⚠️ Leaving the final probability fraction unsimplified.
- ⚠️ Skipping the exhaustive-case verification step.
- ⚠️ Treating all balls as identical or ignoring group membership when computing favorable outcomes.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce using C(n,k) = n! / (k!(n-k)!) for all selection counts; avoid using ordered permutation formulas unless the problem explicitly requires ordered sequences.
- Enforce applying the multiplication principle when favorable outcomes require simultaneous selection from two or more distinct groups (e.g., C(red,r) × C(blue,b)); avoid summing those counts.
- Enforce reducing the final probability fraction by dividing numerator and denominator by their GCD; avoid reporting an unsimplified fraction as the final answer.
- Enforce a sanity check by enumerating all mutually exclusive exhaustive outcome compositions and verifying their favorable counts sum to the total C(n,k); avoid finalizing the answer without this check.
- Enforce partitioning favorable outcomes by group membership and computing a separate combination count per group; avoid collapsing distinct groups into a single undifferentiated pool.

---

## Example Pattern

```
Population: A items of type X, B items of type Y; draw k total. P(exactly r of type X and s of type Y, where r+s=k) = [C(A,r) × C(B,s)] / C(A+B, k), simplified by GCD, verified by summing all valid (r,s) pair numerators to equal C(A+B,k).
```

---

## Source: Contrastive Trajectory Analysis

### Task
```
A bag contains 5 red balls and 3 blue balls. You draw 3 balls one by one WITHOUT replacement. What is the probability that you draw exactly 2 red balls and 1 blue ball? Show your full step-by-step reasoning.
```

### Weak Agent Trajectory (`meta-llama/llama-3.1-8b-instruct`)
<details>
<summary>Expand weak agent reasoning</summary>

To find the probability of drawing exactly 2 red balls and 1 blue ball, we need to consider the total number of ways to draw 3 balls from the bag (without replacement) and the number of favorable outcomes where we draw exactly 2 red balls and 1 blue ball.

**Step 1: Determine the total number of ways to draw 3 balls from the bag**

The bag contains 8 balls (5 red + 3 blue), and we are drawing 3 balls without replacement. The number of ways to choose k items from a set of n items without replacement is given by the combination formula:

nCk = n! / (k!(n-k)!)

where n! denotes the factorial of n (e.g., 5! = 5*4*3*2*1).

In this case, we have n = 8 balls and k = 3 balls drawn. So, the total number of ways to draw 3 balls from the bag is:

8C3 = 8! / (3!(8-3)!)
= 8! / (3!5!)
= (8*7*6) / (3*2*1)
= 56

**Step 2: Calculate the number of favorable outcomes**

To draw exactly 2 red balls and 1 blue ball, we need to choose 2 red balls from the 5 red balls and 1 blue ball from the 3 blue balls.

The number of ways to choose 2 red balls from 5 red balls is:
5C2 = 5! / (2!(5-2)!)
= 5! / (2!3!)
= (5*4) / (2*1)
= 10

The number of ways to choose 1 blue ball from 3 blue balls is:
3C1 = 3! / (1!(3-1)!)
= 3! / (1!2!)
= 3

Since these two events are independent, we multiply the number of ways to choose 2 red balls and 1 blue ball:

Favorable outcomes = 10 * 3
= 30

**Step 3: Calculate the probability**

Now that we have the total number of ways to draw 3 balls (56) and the number of favorable outcomes (30), we can calculate the probability of drawing exactly 2 red balls and 1 blue ball:

Probability = Number of favorable outcomes / Total number of ways to draw 3 balls
= 30 / 56

To simplify the fraction, we can find the greatest common divisor (GCD) of 30 and 56, which is 2. Dividing both numerator and denominator by 2:

Probability = (30 / 2) / (56 / 2)
= 15 / 28

**Final answer:**

The probability of drawing exactly 2 red balls and 1 blue ball is 15/28.

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

# Probability of Drawing Exactly 2 Red Balls and 1 Blue Ball

## Setting Up the Problem

**Given information:**
- Bag contains: 5 red balls + 3 blue balls = **8 balls total**
- Drawing: 3 balls without replacement
- Goal: Find P(exactly 2 red, 1 blue)

---

## Method: Combinations (Counting Favorable vs Total Outcomes)

### Step 1: Count Total Possible Outcomes

The number of ways to choose 3 balls from 8 balls:

$$\binom{8}{3} = \frac{8!}{3! \cdot 5!} = \frac{8 \times 7 \times 6}{3 \times 2 \times 1} = \frac{336}{6} = 56$$

---

### Step 2: Count Favorable Outcomes

We need **exactly 2 red** AND **exactly 1 blue**:

**Ways to choose 2 red balls from 5:**
$$\binom{5}{2} = \frac{5!}{2! \cdot 3!} = \frac{5 \times 4}{2 \times 1} = 10$$

**Ways to choose 1 blue ball from 3:**
$$\binom{3}{1} = \frac{3!}{1! \cdot 2!} = 3$$

**Total favorable outcomes** (by multiplication principle):
$$\binom{5}{2} \times \binom{3}{1} = 10 \times 3 = 30$$

---

### Step 3: Calculate the Probability

$$P(\text{2 red, 1 blue}) = \frac{\text{Favorable outcomes}}{\text{Total outcomes}} = \frac{30}{56}$$

$$\boxed{P = \frac{30}{56} = \frac{15}{28} \approx 0.5357}$$

---

## Verification: Check All Cases Sum to 1

| Outcome | Calculation | Count |
|---------|------------|-------|
| 3 Red, 0 Blue | C(5,3)×C(3,0) = 10×1 | 10 |
| **2 Red, 1 Blue** | **C(5,2)×C(3,1) = 10×3** | **30** |
| 1 Red, 2 Blue | C(5,1)×C(3,2) = 5×3 | 15 |
| 0 Red, 3 Blue | C(5,0)×C(3,3) = 1×1 | 1 |
| **Total** | | **56** ✓ |

All cases sum to **56**, confirming our total is correct.

---

## Final Answer

$$P(\text{exactly 2 red and 1 blue}) = \frac{15}{28} \approx \mathbf{53.57\%}$$

</details>

### Raw Extracted Constraints
```
1. When solving combinatorial probability problems with draws without replacement, enforce using combinations (C(n,k)) to count unordered selections for both total and favorable outcomes; avoid treating sequential draws as ordered permutations unless explicitly required.

2. When calculating favorable outcomes involving multiple independent selection criteria (e.g., choosing from two distinct groups), enforce applying the multiplication principle to combine separate combination counts; avoid adding the counts instead of multiplying them.

3. When computing a final probability fraction, enforce simplifying by dividing numerator and denominator by their greatest common divisor; avoid leaving an unsimplified fraction as the final answer when a simpler form exists.

4. When solving a probability problem, enforce verifying the answer by checking that all mutually exclusive and exhaustive cases sum to the total outcome count (or probability sums to 1); avoid skipping this sanity check, as it catches arithmetic errors in both numerator and denominator.

5. When selecting balls from a bag containing distinct color groups, enforce partitioning the favorable outcomes by separately counting selections from each group then combining; avoid treating all balls as identical or ignoring group membership when computing combinations.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
