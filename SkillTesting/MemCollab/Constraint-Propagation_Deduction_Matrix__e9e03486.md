---
name: Constraint-Propagation Deduction Matrix Reasoning
domain: Logic / Constraint Satisfaction / Analytical Reasoning
generated: 2026-04-02 08:31 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Constraint-Propagation Deduction Matrix Reasoning

> A systematic method for solving multi-attribute logic puzzles (e.g., who owns what, lives where, drives which car) by propagating linked constraints immediately, tracking all categories in parallel, and exhausting all transitive inference chains before declaring ambiguity or making assignments.

## When to Apply

Apply this skill whenever solving a logic puzzle, constraint satisfaction problem, or deduction matrix where multiple entities must be matched to multiple attribute categories using a set of conditional or linking clues — especially when clues connect attributes across categories (e.g., 'the X owner does Y') or when the risk of premature assignment or false ambiguity is present.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Every clue that links two attributes must propagate eliminations across both attributes simultaneously for all entities.
- No entity receives a final assignment in any category until all other entities have been ruled out for that attribute.
- All remaining unassigned attributes across all categories must be tracked in parallel at every deduction step.
- Before declaring ambiguity between two scenarios, all transitive constraint chains must be fully exhausted.
- Linked attribute assignments (e.g., pet + car derived from a single clue) must be applied to the same entity in the same step.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Treating a linked-attribute clue (e.g., 'Cat owner drives Green') as two independent single-attribute eliminations applied in separate, disconnected steps.
- ⚠️ Assigning an attribute to one entity (e.g., Charlie = Cat) without first confirming all other entities are eliminated for that attribute.
- ⚠️ Prematurely declaring one scenario valid without cross-validating it against all remaining clues simultaneously.
- ⚠️ Concluding that ambiguity exists between two scenarios before checking whether any constraint transitively rules out one of them.
- ⚠️ Updating the pet column for an entity without simultaneously updating the car column when the assignment was derived from a pet-car linkage clue.
- ⚠️ Applying clues sequentially in isolation without re-checking earlier eliminations after each new deduction.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce immediate cross-category propagation when a clue links two attributes; avoid deferring the secondary attribute elimination to a later step.
- Enforce parallel tracking of all unassigned attributes for every entity after each deduction step; avoid focusing on a single entity or category in isolation.
- Enforce full multi-clue cross-validation before finalizing any scenario; avoid accepting a scenario as valid after it passes only a subset of clues.
- Enforce exhausting all transitive inference chains (A→B→C eliminations) before labeling a puzzle as ambiguous; avoid stopping at direct eliminations only.
- Enforce simultaneous column updates for all attributes linked by a single clue; avoid separating linked assignments into disconnected reasoning steps.
- Enforce process-of-elimination confirmation: an attribute is assigned to an entity only when it is the sole remaining possibility after all others are ruled out.

---

## Example Pattern

```
Given entities {A, B, C}, attributes {Color: Red/Blue/Green, Pet: Dog/Cat/Bird}: If clue states 'Cat owner drives Green' and B is already assigned Red, immediately eliminate Cat from B and note B cannot drive Green. Then propagate: if A is eliminated from Cat by another clue, Cat must belong to C, and simultaneously assign Green car to C in the same step — not as a separate downstream deduction.
```

---

## Source: Contrastive Trajectory Analysis

### Task
```
Three friends (Alice, Bob, Charlie) have different pets (Dog, Cat, Bird) and drive different colored cars (Red, Blue, Green).
- Alice is allergic to feathers.
- The person with the Cat drives a Green car.
- Bob drives a Red car.
- Charlie hates dogs.
Who owns which pet and drives which car? Show your deduction matrix step-by-step.
```

### Weak Agent Trajectory (`meta-llama/llama-3.1-8b-instruct`)
<details>
<summary>Expand weak agent reasoning</summary>

To solve this problem, we will use a deduction matrix. A deduction matrix is a table that helps us systematically eliminate possibilities and narrow down the options.

Let's start by creating a matrix with the three friends, their possible pets, and their possible cars.

**Deduction Matrix**

| Friend | Pet | Car |
| --- | --- | --- |
| Alice | Dog, Cat, Bird | Red, Blue, Green |
| Bob | Dog, Cat, Bird | Red, Blue, Green |
| Charlie | Dog, Cat, Bird | Red, Blue, Green |

Now, let's apply the given constraints one by one.

**Constraint 1: Alice is allergic to feathers**
Since Alice is allergic to feathers, she cannot have a Bird. We can eliminate the Bird option for Alice.

**Updated Matrix**

| Friend | Pet | Car |
| --- | --- | --- |
| Alice | Dog, Cat | Red, Blue, Green |
| Bob | Dog, Cat, Bird | Red, Blue, Green |
| Charlie | Dog, Cat, Bird | Red, Blue, Green |

**Constraint 2: The person with the Cat drives a Green car.**
Since the person with the Cat drives a Green car, and Alice and Charlie are the only ones left with Cat as a possible pet, we can determine that Charlie is the one with the Cat is not correct because we don't know that yet, and Alice is not the one with the Cat because we know the Cat drives a Green car. Therefore, Charlie is the one who drives the Green car is not correct. The only option left is that Alice is not the one with the Cat and drives a car that isn't green, then Charlie is the one with the Cat and drives the Green car is correct.

**Updated Matrix**

| Friend | Pet | Car |
| --- | --- | --- |
| Alice | Dog | Red, Blue |
| Bob | Dog, Cat, Bird | Red, Blue, Green |
| Charlie | Cat | Green |

**Constraint 3: Bob drives a Red car.**
Since Bob drives a Red car, and the only remaining option for Bob's car is Green, we can eliminate the Green car option from Bob's possible cars.

**Updated Matrix**

| Friend | Pet | Car |
| --- | --- | --- |
| Alice | Dog | Red, Blue |
| Bob | Dog, Cat, Bird | Blue |
| Charlie | Cat | Green |

**Constraint 4: Charlie hates dogs.**
Since Charlie hates dogs, Charlie cannot have a Dog. We can eliminate the Dog option for Charlie.

**Updated Matrix**

| Friend | Pet | Car |
| --- | --- | --- |
| Alice | Dog | Red, Blue |
| Bob | Dog, Cat, Bird | Blue |
| Charlie | Cat | Green |

Now, we can determine the final solution.

**Final Solution**

Based on the updated matrix, we can conclude:

* Alice has a Dog and drives a Blue car.
* Bob has a Cat or Bird (we don't know which one yet) and drives a Blue car.
* Charlie has a Cat and drives a Green car.

Since the only remaining option for Bob's pet is Bird, we can conclude that:

* Bob has a Bird and drives a Blue car.

Therefore, the final solution is:

* Alice owns a Dog and drives a Blue car.
* Bob owns a Bird and drives a Blue car.
* Charlie owns a Cat and drives a Green car.

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

# Logic Puzzle Solution: Step-by-Step Deduction Matrix

## Initial Setup

| Person | Pet | Car |
|---------|-----|-----|
| Alice | ? | ? |
| Bob | ? | ? |
| Charlie | ? | ? |

---

## Clues to Process

1. Alice is allergic to feathers → **Alice ≠ Bird**
2. The person with the Cat drives a Green car → **Cat owner → Green car**
3. Bob drives a Red car → **Bob = Red**
4. Charlie hates dogs → **Charlie ≠ Dog**

---

## Step-by-Step Deductions

### Step 1 — Apply Clue 3: Bob drives Red
| Person | Pet | Car |
|---------|-----|-----|
| Alice | ? | ? |
| Bob | ? | **Red** |
| Charlie | ? | ? |

### Step 2 — Apply Clue 2: Cat owner drives Green
Bob drives **Red**, therefore:
> Bob ≠ Cat (because Cat owner must drive Green, but Bob drives Red)

| Person | Pet | Car |
|---------|-----|-----|
| Alice | ? | ? |
| Bob | ~~Cat~~ | **Red** |
| Charlie | ? | ? |

### Step 3 — Apply Clue 1: Alice ≠ Bird
| Person | Pet | Car |
|---------|-----|-----|
| Alice | ~~Bird~~ | ? |
| Bob | ~~Cat~~ | **Red** |
| Charlie | ? | ? |

### Step 4 — Apply Clue 4: Charlie ≠ Dog
| Person | Pet | Car |
|---------|-----|-----|
| Alice | ~~Bird~~ | ? |
| Bob | ~~Cat~~ | **Red** |
| Charlie | ~~Dog~~ | ? |

### Step 5 — Solve for Charlie's Pet
Charlie cannot have **Dog** or... let's check what's left:
- Bob ≠ Cat → Bob has **Dog** or **Bird**
- Charlie ≠ Dog → Charlie has **Cat** or **Bird**
- Alice ≠ Bird → Alice has **Dog** or **Cat**

Since Charlie ≠ Dog, and Bob ≠ Cat:
> The only person who can have the **Cat** is **Alice or Charlie**
> The only person who can have the **Dog** is **Alice or Bob**

Since Charlie ≠ Dog → Charlie has **Cat** or **Bird**
Since Alice ≠ Bird → Alice has **Dog** or **Cat**

If Charlie has **Bird** → Alice and Bob share Dog and Cat → Bob can't have Cat → **Bob = Dog, Alice = Cat** ✓
If Charlie has **Cat** → Charlie drives Green (from Clue 2) → Alice gets Dog or Bird, but Alice ≠ Bird → **Alice = Dog, Bob = Bird** ✓

### Step 6 — Break the Tie Using Car Logic
From Clue 2: **Cat owner drives Green**
- Cars available: Red (Bob), Blue, Green
- If **Alice = Cat** → Alice drives **Green** ✓ (possible)
- If **Charlie = Cat** → Charlie drives **Green** ✓ (possible)

Check Bob's pet in each scenario:
- Scenario A: Charlie=Bird, Alice=Cat, Bob=Dog
- Scenario B: Charlie=Cat, Alice=Dog, Bob=Bird

In **Scenario B**: Bob = Bird → But Bob drives Red, and Cat owner drives Green (Charlie=Green). That works mechanically, BUT...

> **Alice is allergic to feathers (Clue 1) → Alice ≠ Bird** ✓ (satisfied in both)
> In Scenario B, Bob has Bird — no rule prevents this ✓

We need one more check — **both scenarios seem valid so far**. Let's re-examine:

In Scenario A: Alice=Cat → Alice drives **Green**; Charlie=Bird → Charlie drives **Blue**
In Scenario B: Charlie=Cat → Charlie drives **Green**; Alice=Dog → Alice drives **Blue**

**Both are internally consistent!** Re-read clue 1 carefully:
> "Alice is allergic to feathers" → Alice ≠ Bird ✓ (satisfied in both)

Charlie hates dogs → Charlie ≠ Dog ✓ (satisfied in both)

**Scenario A is the unique solution** because in Scenario B, Bob would have the Bird — and while not explicitly forbidden, Scenario A is confirmed by process of elimination:

- Alice ≠ Bird, Charlie ≠ Dog → **Alice = Dog or Cat; Charlie = Cat or Bird**
- Bob ≠ Cat → Bob = **Dog or Bird**
- Only Bob can have **Dog** if Alice takes Cat... let's finalize:

> Charlie ≠ Dog, Bob ≠ Cat → **Bob must have Dog or Bird; Charlie must have Cat or Bird**
> Since both Bob and Charlie could have Bird, but only ONE can → assign uniquely:
> **Charlie = Cat** (the only one not ruled out for cat besides Alice), giving Alice = Dog, Bob = Bird

✅ **Final unique solution = Scenario B**

---

## ✅ Final Answer

| Person | Pet | Car |
|---------|-----|-----|
| **Alice** | 🐕 **Dog** | 🔵 **Blue** |
| **Bob** | 🐦 **Bird** | 🔴 **Red** |
| **Charlie** | 🐱 **Cat** | 🟢 **Green** |

### Verification:
| Clue | Check |
|------|-------|
| Alice is allergic to feathers → no Bird | Alice has Dog ✅ |
| Cat owner drives Green | Charlie (Cat) drives Green ✅ |
| Bob drives Red | Bob drives Red ✅ |
| Charlie hates dogs → no Dog | Charlie has Cat ✅ |

</details>

### Raw Extracted Constraints
```
1. When a clue links two attributes together (e.g., "Cat owner drives Green"), enforce propagating that constraint immediately to eliminate that attribute combination for anyone already assigned the conflicting attribute; avoid treating linked-attribute clues as independent single-attribute eliminations.

2. When multiple scenarios remain after applying all direct elimination clues, enforce testing each scenario against all remaining clues simultaneously before declaring a tie; avoid prematurely labeling one scenario as valid without completing full cross-validation.

3. When performing process-of-elimination reasoning, enforce tracking which attributes remain unassigned for each entity in parallel across all categories; avoid making unilateral assignments to one entity (e.g., Charlie = Cat) without first confirming all other entities are ruled out for that attribute.

4. When a deduction matrix yields two seemingly valid scenarios, enforce checking whether any constraint transitively rules out one scenario through a chain of inferences; avoid concluding ambiguity exists before exhausting all transitive constraint chains.

5. When assigning cars derived from pet-ownership clues, enforce updating the car column simultaneously with the pet column for the same entity; avoid separating linked attribute assignments into disconnected steps that may introduce inconsistencies.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
