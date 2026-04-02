---
name: Binary Search Boundary Update Correctness
domain: Coding / Algorithms
generated: 2026-04-02 08:29 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Binary Search Boundary Update Correctness

> Teaches how to correctly update search boundaries in binary search to guarantee loop termination and correctness by ensuring the search space strictly shrinks on every iteration, and how to diagnose, explain, and verify infinite loop bugs caused by boundary update errors.

## When to Apply

Apply when implementing, reviewing, debugging, or explaining any iterative binary search algorithm, or any divide-and-conquer search where two pointers converge toward a target by halving a search space each iteration.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Every branch of a binary search loop must strictly reduce the search space; if any branch can leave boundaries unchanged, the loop may never terminate.
- After evaluating arr[mid] and finding it is not the target, mid itself is excluded from further consideration and must not be included in the updated boundary.
- The root cause of a non-terminating binary search is a failure of the search space shrinkage invariant, not merely the symptom of an infinite loop.
- Loop termination requires that low strictly increases or high strictly decreases on every iteration where the target is not found.
- Correctness requires testing all boundary conditions: leftmost element, rightmost element, middle element, absent element, and out-of-range values.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Setting low = mid instead of low = mid + 1 after arr[mid] < target, which can freeze low when mid == low.
- ⚠️ Setting high = mid instead of high = mid - 1 after arr[mid] > target, which can freeze high when mid == high.
- ⚠️ Assuming pointer updates are correct because they move in the right direction without verifying they exclude the already-evaluated index.
- ⚠️ Identifying the bug only as 'infinite loop' without explaining that the underlying invariant violated is that the search space must strictly shrink.
- ⚠️ Validating an implementation only on easy cases where the target is near the middle, without testing adjacent-pointer edge cases where low and high differ by exactly 1.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce using low = mid + 1 when arr[mid] < target to exclude the evaluated midpoint from the lower search boundary; avoid low = mid which can stall progress when mid equals low.
- Enforce using high = mid - 1 when arr[mid] > target to exclude the evaluated midpoint from the upper search boundary; avoid high = mid which can stall progress when mid equals high.
- Enforce tracing through the adjacent-pointer edge case (low and high differing by 1) to verify that both the low-update and high-update branches cause strict progress; avoid assuming correctness without this trace.
- Enforce explaining boundary update bugs in terms of the search space shrinkage invariant being violated; avoid surface-level descriptions that only name the symptom.
- Enforce testing the corrected implementation against leftmost element, rightmost element, middle element, missing element, and out-of-range values before declaring it correct; avoid only testing the happy path.

---

## Example Pattern

```
Given a sorted array and a target, if mid is evaluated and does not match: use low = mid + 1 (not mid) when target is in the upper half, and high = mid - 1 (not mid) when target is in the lower half. Trace the case where low=0, high=1 to confirm mid=0, and that updating low=mid+1=1 causes low==high on the next step, terminating the loop correctly rather than repeating mid=0 forever.
```

---

## Source: Contrastive Trajectory Analysis

### Task
```
Identify the bug in this Binary Search implementation and provide the corrected code. Explain why it fails:
def binary_search(arr, target):
  low, high = 0, len(arr) - 1
  while low <= high:
    mid = (low + high) // 2
    if arr[mid] == target: return mid
    elif arr[mid] < target: low = mid
    else: high = mid
  return -1
```

### Weak Agent Trajectory (`meta-llama/llama-3.1-8b-instruct`)
<details>
<summary>Expand weak agent reasoning</summary>

### Bug Identification and Correction

The given binary search implementation has two issues:

1.  **Incorrect update of the `high` variable in the `else` clause:** When `arr[mid]` is greater than the `target`, the code updates `high` to `mid` instead of updating it to `mid - 1`. This leads to an incorrect search space and causes the algorithm to enter an infinite loop.

2.  **Incorrect update of the `low` variable in the `elif` clause:** When `arr[mid]` is less than the `target`, the code updates `low` to `mid`, not `mid + 1`. This also leads to an incorrect search space.

### Corrected Code

Here's the corrected binary search implementation:

```python
def binary_search(arr, target):
    """
    Performs a binary search on the given sorted array to find the index of the target element.

    Args:
    arr (list): A sorted list of elements.
    target: The element to search for in the list.

    Returns:
    int: The index of the target element if found, -1 otherwise.
    """
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            # Update low to mid + 1 to continue searching in the right half
            low = mid + 1
        else:
            # Update high to mid - 1 to continue searching in the left half
            high = mid - 1
    return -1
```

### Explanation

The binary search algorithm works by repeatedly dividing the search space in half and searching for the target in one of the two halves. The corrected code updates the `low` and `high` variables correctly to ensure the algorithm searches the correct halves of the array.

When `arr[mid]` is less than the `target`, the code updates `low` to `mid + 1` to continue searching in the right half. When `arr[mid]` is greater than the `target`, the code updates `high` to `mid - 1` to continue searching in the left half.

### Example Use Case

```python
arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
target = 23
index = binary_search(arr, target)
print(index)  # Output: 5
```

In this example, the binary search implementation finds the index of the `target` element (23) in the given sorted array.

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

## Bug Analysis

### The Problem: Infinite Loop

The bug is in the pointer update logic when `arr[mid] < target`:

```python
elif arr[mid] < target: low = mid  # ❌ BUG HERE
```

And similarly when `arr[mid] > target`:

```python
else: high = mid  # ❌ BUG HERE
```

---

### Why It Causes an Infinite Loop

Consider searching for `5` in `[1, 3, 5, 7, 9]`:

```
Initial:  low=0, high=4
Step 1:   mid=2, arr[2]=5 → Found! ✅ (lucky case)

Now consider searching for 7:
Step 1:   low=0, high=4, mid=2, arr[2]=3 < 7  → low = mid = 2
Step 2:   low=2, high=4, mid=3, arr[3]=7      → Found! ✅

Now consider searching for 9 in [1, 3]:
Step 1:   low=0, high=1, mid=0, arr[0]=1 < 3  → low = mid = 0
Step 2:   low=0, high=1, mid=0, arr[0]=1 < 3  → low = mid = 0
          ♾️ INFINITE LOOP — low never advances past mid!
```

**Root Cause:** When `low` and `high` are adjacent (differ by 1), `mid` always equals `low`. Setting `low = mid` means `low` never increases, causing an **infinite loop**.

The same problem occurs with `high = mid` — it never decreases past `mid`.

---

### The Fix

Exclude `mid` from the next search range since we already know `arr[mid]` is not the target:

```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1   # ✅ Move low PAST mid
        else:
            high = mid - 1  # ✅ Move high PAST mid

    return -1
```

---

### Verification

```python
arr = [1, 3, 5, 7, 9]

print(binary_search(arr, 1))   # → 0  ✅ (leftmost element)
print(binary_search(arr, 9))   # → 4  ✅ (rightmost element)
print(binary_search(arr, 5))   # → 2  ✅ (middle element)
print(binary_search(arr, 4))   # → -1 ✅ (not found)
print(binary_search(arr, 10))  # → -1 ✅ (out of range)
```

---

### Summary Table

| Condition | Buggy Code | Fixed Code | Why |
|---|---|---|---|
| `arr[mid] < target` | `low = mid` | `low = mid + 1` | `mid` already checked; skip it |
| `arr[mid] > target` | `high = mid` | `high = mid - 1` | `mid` already checked; skip it |

> **Key Insight:** After checking `arr[mid]`, we know it's not the target, so the next search range must **exclude** `mid`. Using `mid + 1` and `mid - 1` guarantees the search space strictly shrinks each iteration, ensuring termination.

</details>

### Raw Extracted Constraints
```
1. When updating search boundaries in binary search after a failed comparison, enforce excluding the already-checked midpoint by using `mid + 1` or `mid - 1`; avoid setting boundaries equal to `mid` which can prevent the search space from shrinking.

2. When analyzing potential infinite loops in iterative algorithms, enforce tracing through edge cases where pointers are adjacent (differ by 1) to verify the loop variable strictly progresses each iteration; avoid assuming pointer updates always reduce the search space without verification.

3. When fixing pointer/boundary update bugs, enforce explaining the root cause (search space fails to shrink) rather than just stating the symptom (infinite loop); avoid surface-level identification that names the bug without clarifying the underlying invariant being violated.

4. When verifying a search algorithm's correctness, enforce testing boundary conditions including the leftmost element, rightmost element, middle element, missing element, and out-of-range values; avoid only testing the "happy path" where the target is easily found.

5. When reasoning about loop termination in binary search, enforce confirming that every branch of the loop strictly reduces the search space each iteration; avoid treating pointer updates as correct simply because they move in the right direction without confirming they exclude already-evaluated indices.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
