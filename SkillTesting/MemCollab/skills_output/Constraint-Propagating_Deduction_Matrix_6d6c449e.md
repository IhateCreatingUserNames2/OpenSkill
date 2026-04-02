---
name: Constraint-Propagating Deduction Matrix
domain: Logic / Reasoning / Puzzle Solving
generated: 2026-04-02 09:12 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Constraint-Propagating Deduction Matrix

> A systematic skill for solving attribute-assignment logic puzzles by maintaining a live deduction matrix, propagating every clue bidirectionally and immediately, and enforcing global consistency after each step.

## When to Apply

Apply this skill whenever solving any constraint-based assignment puzzle where a fixed set of entities must be uniquely matched to multiple attribute categories using a list of logical clues — including Einstein-style grid puzzles, logic grid puzzles, and any deduction problem with mutually exclusive one-to-one mappings.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Every confirmed assignment must simultaneously eliminate all other options for that person in that category AND eliminate that option for all other persons.
- Conditional clues (e.g., 'X owns Cat → X drives Green') must be enforced in both directions: forward (Cat owner must have Green) and backward (anyone without Green cannot have Cat).
- After each clue application, the matrix must reach a locally stable state before the next clue is processed.
- When only one option remains for a person in any category, it is immediately confirmed — not left as uncertain.
- A single canonical matrix state is maintained throughout; no earlier confirmed cell may be overridden or re-assigned later.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Treating a conditional clue as one-directional (applying only the forward implication and ignoring the contrapositive).
- ⚠️ Deferring matrix updates — moving to the next clue before propagating all consequences of the current one.
- ⚠️ Leaving duplicate or ambiguous assignments (two people marked possible for the same unique attribute).
- ⚠️ Re-deducing an already-confirmed cell and arriving at a contradictory conclusion.
- ⚠️ Treating a cell as uncertain after elimination has reduced it to the only remaining option.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce bidirectional propagation of every conditional clue; avoid applying only the forward direction and missing eliminations implied by the contrapositive.
- Enforce immediate full-matrix update after each clue before advancing; avoid batching updates or skipping intermediate propagation.
- Enforce simultaneous cross-person and cross-category elimination upon any confirmation; avoid isolated single-cell updates that leave sibling cells ambiguous.
- Enforce auto-confirmation when a category is reduced to one possibility for a person; avoid continuing to mark that cell as uncertain.
- Enforce a single consistent matrix state from start to finish; avoid re-assigning confirmed attributes or contradicting earlier deductions in later steps.

---

## Example Pattern

```
Given: Entity B is assigned Attribute-X. Conditional clue: 'Attribute-Y owner has Attribute-Z.' Step 1: Confirm B=X, eliminate X for all others, eliminate all other attributes in that category for B. Step 2: Apply conditional — find who can satisfy Attribute-Z; eliminate Attribute-Y from anyone who cannot have Attribute-Z; eliminate Attribute-Z from anyone confirmed without Attribute-Y. Step 3: If elimination leaves only one candidate per attribute, auto-confirm and propagate again. Repeat until all cells are resolved.
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

To solve this problem, we can create a matrix to represent the information given. Let's denote the person who owns the pet as P (Person) and the pet as A (Animal) and the car as C (Car). The first step is to create the matrix with all the possible combinations and mark the ones that we know. Here is the initial matrix:

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   | C  |        |
| A   | B  |        |
| B   | D  |        |
| B   | C  |        |
| B   | B  |        |
| C   | D  |        |
| C   | C  |        |
| C   | B  |        |

From the given information:
- Alice is allergic to feathers, which means she cannot have a Bird (B). 
- The person with the Cat (C) drives a Green car.
- Bob drives a Red car.
- Charlie hates dogs.

Now, let's mark the information we know in the matrix:

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   | C  |        |
| A   | B  |        |
| B   | D  |        |
| B   | C  |        |
| B   | B  |        |
| C   | D  |        |
| C   | C  |        |
| C   | B  |        |

Since Alice is allergic to feathers, we can eliminate the possibility of her having a Bird (B). 

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   | C  |        |
| A   |    |        |
| B   | D  |        |
| B   | C  |        |
| B   | B  |        |
| C   | D  |        |
| C   | C  |        |
| C   | B  |        |

Since the person with the Cat (C) drives a Green car, we can mark the Green car for the person who has the Cat.

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   | C  |        |
| A   |    |        |
| B   | D  | R      |
| B   | C  |        |
| B   | B  |        |
| C   | D  |        |
| C   | C  | G      |
| C   | B  |        |

Since Bob drives a Red car, we can mark the Red car as Bob's.

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   | C  |        |
| A   |    |        |
| B   | D  | R      |
| B   | C  |        |
| B   | B  |        |
| C   | D  |        |
| C   | C  | G      |
| C   | B  |        |

Since Charlie hates dogs, we can eliminate the possibility of Charlie having a Dog (D).

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   | C  |        |
| A   |    |        |
| B   | D  | R      |
| B   | C  |        |
| B   | B  |        |
| C   |    |        |
| C   | C  | G      |
| C   | B  |        |

Since the person with the Cat (C) drives a Green car, we can now conclude that Charlie has the Cat (C) and drives the Green car.

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   | C  |        |
| A   |    |        |
| B   | D  | R      |
| B   | C  |        |
| B   | B  |        |
| C   |    | G      |
| C   | C  |        |
| C   | B  |        |

Since Charlie has the Cat (C) and drives the Green car, this means that Alice must have the Dog (D).

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   |    |        |
| A   |    |        |
| B   | D  | R      |
| B   | C  |        |
| B   | B  |        |
| C   |    | G      |
| C   | C  |        |
| C   | B  |        |

Since Alice has the Dog (D), this means Bob cannot have the Dog (D), and therefore, Bob must have the Bird (B).

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   |    |        |
| A   |    |        |
| B   |    | R      |
| B   | C  |        |
| B   | B  |        |
| C   |    | G      |
| C   | C  |        |
| C   | B  |        |

Since the only option left for Alice is the Cat (C), we can mark the Cat for Alice.

| P  | A  | C       |
| --- | --- | --- |
| A   |    |        |
| A   | C  |        |
| A   |    |        |
| B   |    | R      |
| B   | C  |        |
| B   | B  |        |
| C   |    | G      |
| C   | C  |        |
| C   | B  |        |

Since Charlie has the Cat (C) and drives the Green car, and Bob has the Bird (B), we can conclude that Charlie cannot have the Bird (B).

| P  | A  | C       |
| --- | --- | --- |
| A   |    |        |
| A   | C  |        |
| A   |    |        |
| B   |    | R      |
| B   | C  |        |
| B   | B  |        |
| C   |    | G      |
| C   |    |        |
| C   | B  |        |

Since Alice has the Cat (C) and Bob has the Bird (B), this means that Charlie must have the Dog (D).

| P  | A  | C       |
| --- | --- | --- |
| A   |    |        |
| A   | C  |        |
| A   |    |        |
| B   |    | R      |
| B   | C  |        |
| B   | B  |        |
| C   | D  | G      |
| C   |    |        |
| C   | B  |        |

Since the only option left for Bob is the Cat (C), we can mark the Cat for Bob.

| P  | A  | C       |
| --- | --- | --- |
| A   |    |        |
| A   | C  |        |
| A   |    |        |
| B   |    | R      |
| B   |    |        |
| B   | B  |        |
| C   | D  | G      |
| C   |    |        |
| C   | B  |        |

Since Bob has the Cat (C) and drives the Red car, we can conclude that Charlie has the Dog (D) and drives the Green car.

| P  | A  | C       |
| --- | --- | --- |
| A   |    |        |
| A   | C  |        |
| A   |    |        |
| B   |    | R      |
| B   |    |        |
| B   | B  |        |
| C   | D  | G      |
| C   |    |        |
| C   | B  |        |

Since the only option left for Alice is the Dog (D), we can mark the Dog for Alice.

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   |    |        |
| A   |    |        |
| B   |    | R      |
| B   |    |        |
| B   | B  |        |
| C   |    | G      |
| C   |    |        |
| C   | B  |        |

Since the only option left for Bob is the Bird (B), we can mark the Bird for Bob.

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   |    |        |
| A   |    |        |
| B   |    | R      |
| B   |    |        |
| B   | B  |        |
| C   |    | G      |
| C   |    |        |
| C   | B  |        |

Since the only option left for Charlie is the Bird (B), we can mark the Bird for Charlie.

| P  | A  | C       |
| --- | --- | --- |
| A   | D  |        |
| A   |    |        |
| A   |    |        |
| B   |    | R      |
| B   |    |        |
| B   | B  |        |
| C   |    | G      |
| C   |    |        |
| C   | B  |        |

Now, we can fill in the remaining information:
- Alice has the Dog (D).
- Bob has the Bird (B).
- Charlie has the Bird (B) and drives the Green car.

The final answer is:

Alice: Dog, Charlie's car
Bob: Bird, Red car
Charlie: Bird, Green car

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

# Logic Puzzle: Step-by-Step Deduction Matrix

## Setting Up the Matrix

I'll track possibilities with ✓ (confirmed), ✗ (eliminated), and ? (unknown).

---

## Starting Matrix (all unknown)

| Person | Dog | Cat | Bird | Red | Blue | Green |
|--------|-----|-----|------|-----|------|-------|
| Alice | ? | ? | ? | ? | ? | ? |
| Bob | ? | ? | ? | ? | ? | ? |
| Charlie| ? | ? | ? | ? | ? | ? |

---

## Step 1: Apply Clue 3 — *"Bob drives a Red car"*

Bob = Red → Bob ≠ Blue, Bob ≠ Green
Also: Alice ≠ Red, Charlie ≠ Red

| Person | Dog | Cat | Bird | Red | Blue | Green |
|--------|-----|-----|------|-----|------|-------|
| Alice | ? | ? | ? | ✗ | ? | ? |
| Bob | ? | ? | ? | ✓ | ✗ | ✗ |
| Charlie| ? | ? | ? | ✗ | ? | ? |

---

## Step 2: Apply Clue 2 — *"The person with the Cat drives a Green car"*

Cat owner → Green car.
Bob drives Red (not Green), so **Bob ≠ Cat**.

| Person | Dog | Cat | Bird | Red | Blue | Green |
|--------|-----|-----|------|-----|------|-------|
| Alice | ? | ? | ? | ✗ | ? | ? |
| Bob | ? | ✗ | ? | ✓ | ✗ | ✗ |
| Charlie| ? | ? | ? | ✗ | ? | ? |

---

## Step 3: Apply Clue 1 — *"Alice is allergic to feathers"*

Allergic to feathers → Alice ≠ Bird.

| Person | Dog | Cat | Bird | Red | Blue | Green |
|--------|-----|-----|------|-----|------|-------|
| Alice | ? | ? | ✗ | ✗ | ? | ? |
| Bob | ? | ✗ | ? | ✓ | ✗ | ✗ |
| Charlie| ? | ? | ? | ✗ | ? | ? |

---

## Step 4: Apply Clue 4 — *"Charlie hates dogs"*

Charlie ≠ Dog.

| Person | Dog | Cat | Bird | Red | Blue | Green |
|--------|-----|-----|------|-----|------|-------|
| Alice | ? | ? | ✗ | ✗ | ? | ? |
| Bob | ? | ✗ | ? | ✓ | ✗ | ✗ |
| Charlie| ✗ | ? | ? | ✗ | ? | ? |

---

## Step 5: Deduce Pets

- Alice: can't have Bird → Alice has **Dog or Cat**
- Bob: can't have Cat → Bob has **Dog or Bird**
- Charlie: can't have Dog → Charlie has **Cat or Bird**

Since Alice can't have Bird, and Charlie can't have Dog:
- The Bird must go to **Bob or Charlie**
- The Dog must go to **Alice or Bob**

Now, if Charlie has Cat → Charlie drives Green (from Clue 2).
Let's test: **Charlie = Cat → Charlie = Green** ✓ (Charlie ≠ Red, so Green or Blue — this works!)

That leaves Alice and Bob with Dog and Bird.
But Alice ≠ Bird → **Alice = Dog**, **Bob = Bird** ✓

---

## Step 6: Deduce Cars

- Bob = Red ✓ (already confirmed)
- Charlie = Cat → Charlie = Green ✓
- Alice gets the remaining car = **Blue** ✓

---

## Final Completed Matrix

| Person | Dog | Cat | Bird | Red | Blue | Green |
|--------|-----|-----|------|-----|------|-------|
| Alice | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Bob | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Charlie| ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |

---

## ✅ Final Answer

| Person | Pet | Car |
|--------|-----|-----|
| **Alice** | 🐕 Dog | 🔵 Blue |
| **Bob** | 🐦 Bird | 🔴 Red |
| **Charlie** | 🐈 Cat | 🟢 Green |

</details>

### Raw Extracted Constraints
```
1. When applying a conditional clue (e.g., "Cat owner drives Green"), enforce propagating both directions—eliminating the pet for anyone who can't satisfy the condition and eliminating the car for anyone confirmed without it; avoid treating the clue as one-directional.

2. When deducing assignments step-by-step, enforce updating all related cells in the matrix immediately after each clue is applied before moving to the next clue; avoid revisiting already-resolved assignments and contradicting them with later "re-deductions."

3. When a person is confirmed to own a specific pet, enforce marking all other pets as eliminated for that person and all other persons as eliminated for that pet simultaneously; avoid leaving ambiguous or duplicate assignments for the same attribute.

4. When elimination reduces a person to only one remaining option in a category, enforce immediately confirming that option as their assignment; avoid continuing to treat it as uncertain or overriding it with inconsistent conclusions later.

5. When tracking logical deductions across multiple clues, enforce maintaining a single consistent state of the matrix throughout; avoid contradicting earlier confirmed assignments by re-assigning the same pet or car to a different person in subsequent steps.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
