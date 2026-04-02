---
name: Binary Search Boundary Convergence
domain: Coding / Algorithms
generated: 2026-04-02 09:04 UTC
method: MemCollab Contrastive Trajectory Distillation
weak_agent: meta-llama/llama-3.1-8b-instruct
strong_agent: anthropic/claude-sonnet-4.6
---

# Binary Search Boundary Convergence

> How to correctly update search boundaries in binary search to guarantee loop termination and avoid infinite loops caused by failing to shrink the search space past the evaluated midpoint.

## When to Apply

Apply when implementing, reviewing, or debugging any iterative binary search or divide-and-conquer search algorithm where pointer or boundary variables are updated based on a midpoint comparison, especially when investigating potential infinite loops or off-by-one errors.

---

## Reasoning Invariants
*(Principles that MUST be preserved for correct reasoning)*

- Each iteration of a binary search loop must strictly reduce the search space by at least one element.
- After evaluating arr[mid] and finding it is not the target, mid must be excluded from the next search space.
- Loop termination is guaranteed if and only if low and high converge by moving strictly past mid on every non-matching branch.
- Setting low = mid + 1 when arr[mid] < target and high = mid - 1 when arr[mid] > target are the canonical correct updates.
- A bug that only manifests on certain inputs (e.g., missing elements, adjacent boundaries) is still a bug — lucky termination does not imply correctness.

---

## Violation Patterns
*(Forbidden patterns that lead to failure)*

- ⚠️ Setting low = mid instead of low = mid + 1 leaves mid in the search space and can cause the loop to stall.
- ⚠️ Setting high = mid instead of high = mid - 1 leaves mid in the search space and can cause the loop to stall.
- ⚠️ Claiming correctness based only on cases where the target exists near the middle without testing missing-element edge cases.
- ⚠️ Describing the bug vaguely as 'does not correctly define the range' without demonstrating a concrete convergence failure.
- ⚠️ Assuming the algorithm is correct because it terminates on some inputs despite containing a boundary update bug.

---

## Normative Constraints
*(Reusable rules: enforce ... ; avoid ...)*

- Enforce updating low = mid + 1 (not mid) when arr[mid] < target; avoid any update that keeps mid within the new search range.
- Enforce updating high = mid - 1 (not mid) when arr[mid] > target; avoid any update that keeps mid within the new search range.
- Enforce demonstrating the infinite loop with a concrete trace where two adjacent elements and a missing target cause boundaries to never converge; avoid stopping analysis at a case where the loop happens to terminate.
- Enforce explaining convergence failure by showing the search space size does not decrease across iterations; avoid vague range descriptions without iteration-by-iteration state traces.
- Enforce testing edge cases: target at index 0, target at last index, target below minimum, target above maximum, and the specific two-element adjacent case that triggers the original infinite loop; avoid testing only happy-path middle-element cases.
- Enforce verifying loop termination by confirming high - low strictly decreases each iteration; avoid assuming termination without proof of monotonic space reduction.

---

## Example Pattern

```
Search for missing element 6 in sorted array [1, 3, 5, 7]: with low=mid bug, state (low=2, high=3, mid=2) repeats forever because arr[2]=5 < 6 sets low=mid=2 with no progress. Correct fix: low=mid+1=3, next iteration low=3 > high=3 exits loop and returns -1 correctly.
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

## Bug Identification

The bug in the provided Binary Search implementation is in the update of the `low` and `high` pointers. 

When the element at the `mid` index is less than the target, the `low` pointer should be updated to `mid + 1` to continue the search in the right half of the array. 

However, in the given code, `low` is updated to `mid`, which is incorrect because it does not correctly define the range of the left half of the array for the next iteration.

Similarly, when the element at the `mid` index is greater than the target, the `high` pointer should be updated to `mid - 1` to continue the search in the left half of the array. 

However, in the given code, `high` is updated to `mid`, which is also incorrect because it does not correctly define the range of the right half of the array for the next iteration.

## Corrected Code

```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: # If the target is greater than the middle element, 
            low = mid + 1 # update low to mid + 1 to search the right half
        else: # If the target is less than the middle element, 
            high = mid - 1 # update high to mid - 1 to search the left half
    return -1  # Return -1 if the target is not found in the array
```

## Explanation

Here is the step-by-step explanation of the corrected code:

1. We initialize the `low` pointer to the first index of the array and the `high` pointer to the last index of the array.

2. We then enter a while loop that continues until `low` is less than or equal to `high`.

3. Inside the loop, we calculate the `mid` index by averaging `low` and `high`.

4. We then check if the element at the `mid` index is equal to the target. If it is, we return the `mid` index as the target has been found.

5. If the element at the `mid` index is less than the target, we update `low` to `mid + 1` to continue the search in the right half of the array. This is because we have confirmed that the target is not in the left half, so we can ignore that part of the array.

6. If the element at the `mid` index is greater than the target, we update `high` to `mid - 1` to continue the search in the left half of the array. This is because we have confirmed that the target is not in the right half, so we can ignore that part of the array.

7. If the while loop ends without finding the target, we return -1 to indicate that the target is not in the array.

This corrected code should now work as expected, performing a binary search on the input array to find the index of the target element.

</details>

### Strong Agent Trajectory (`anthropic/claude-sonnet-4.6`)
<details>
<summary>Expand strong agent reasoning</summary>

## Bug Analysis

### The Problem: Infinite Loop

The bug is in the pointer updates when the target is **not found at `mid`**:

```python
elif arr[mid] < target: low = mid   # ❌ Should be mid + 1
else: high = mid                     # ❌ Should be mid - 1
```

---

### Why It Causes an Infinite Loop

Consider searching for `5` in `[1, 3, 5, 7]`:

```
Step 1: low=0, high=3 → mid=1 → arr[1]=3 < 5 → low = mid = 1  (no progress!)
Step 2: low=1, high=3 → mid=2 → arr[2]=5 == 5 → returns 2  ✅ (lucky)
```

But searching for `6` in `[1, 3, 5, 7]`:

```
Step 1: low=0, high=3 → mid=1 → arr[1]=3 < 6 → low = mid = 1
Step 2: low=1, high=3 → mid=2 → arr[2]=5 < 6 → low = mid = 2
Step 3: low=2, high=3 → mid=2 → arr[2]=5 < 6 → low = mid = 2  ♾️ INFINITE LOOP
         ↑ low never advances past mid, stuck forever!
```

**Root Cause:** When `arr[mid]` is not the target, `mid` has already been **evaluated and eliminated**. The search space must exclude `mid` by moving **past** it (`mid ± 1`), otherwise the boundaries never converge.

---

### Corrected Code

```python
def binary_search(arr, target):
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1   # ✅ Exclude mid, search RIGHT half
        else:
            high = mid - 1  # ✅ Exclude mid, search LEFT half

    return -1  # Target not found
```

---

### Verification

```python
arr = [1, 3, 5, 7, 9, 11]

print(binary_search(arr, 7))   # ✅ Returns 3
print(binary_search(arr, 1))   # ✅ Returns 0  (left edge)
print(binary_search(arr, 11))  # ✅ Returns 5  (right edge)
print(binary_search(arr, 6))   # ✅ Returns -1 (not found, no infinite loop)
print(binary_search(arr, 0))   # ✅ Returns -1 (below range)
print(binary_search(arr, 12))  # ✅ Returns -1 (above range)
```

---

### Summary Table

| Condition | Buggy Code | Fixed Code | Effect |
|---|---|---|---|
| `arr[mid] < target` | `low = mid` | `low = mid + 1` | Excludes evaluated `mid` |
| `arr[mid] > target` | `high = mid` | `high = mid - 1` | Excludes evaluated `mid` |

**The golden rule:** Once `arr[mid]` is checked and doesn't match, it must be **permanently excluded** from the search space to guarantee convergence.

</details>

### Raw Extracted Constraints
```
1. When updating search boundaries in binary search after a non-matching comparison, enforce moving pointers to `mid ± 1` to exclude the already-evaluated index; avoid setting `low = mid` or `high = mid` which leaves `mid` in the search space and risks infinite loops.

2. When analyzing a potential infinite loop bug, enforce tracing a concrete example where boundaries never converge (e.g., two adjacent elements where target is not present); avoid stopping at a "lucky" case where the loop terminates correctly despite the bug.

3. When explaining why a boundary update is incorrect, enforce demonstrating that the root cause is failure to shrink the search space past the evaluated element; avoid vague descriptions like "does not correctly define the range" without showing the convergence failure.

4. When verifying a corrected search algorithm, enforce testing edge cases including left edge, right edge, missing elements below range, above range, and elements causing the original infinite loop; avoid testing only the happy path where the target exists near the middle.

5. When reasoning about loop termination in iterative algorithms, enforce confirming that each iteration strictly reduces the search space by at least one element; avoid assuming correctness based on cases where progress happens to occur by coincidence.
```

---

*Generated by SkillCrafter using MemCollab — Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation*
*Paper: https://arxiv.org/abs/2603.23234*
