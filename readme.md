For a genetic algorithm, I’d make the **fitness** a **single scalar** that rewards good aerodynamics and **penalizes violation of the fuselage-volume requirement**.

## Recommended objective

If your goal is:

- **maximize trim L/D**
- keep the aircraft **trimmed** at cruise
- keep the aircraft **statically stable**
- make sure the **fuselage volume** is close to a target

then a good fitness function is:

\[
\text{fitness} = \text{L/D} - P_\text{trim} - P_\text{stability} - P_\text{volume}
\]

For a GA, it’s usually easiest to **maximize** this score.

---

## Volume constraint

Since you said “the fuse cell has a certain volume,” you likely want one of these:

### 1. Exact target volume
Penalize deviation from a target volume:

\[
P_\text{volume} = w_v \left(\frac{V_\text{fuse} - V_\text{target}}{V_\text{target}}\right)^2
\]

### 2. Minimum volume only
If the fuselage must be at least some volume:

\[
P_\text{volume} = w_v \max(0, V_\text{target} - V_\text{fuse})^2
\]

---

## What I would use for your case

Because your code already computes trim L/D and stability, I’d use:

\[
\text{fitness} =
\text{L/D}
- w_\alpha |\alpha_\text{trim}|
- w_c \max(0, -dC_m/d\alpha)
- w_v \left(\frac{V_\text{fuse} - V_\text{target}}{V_\text{target}}\right)^2
\]

Where:

- `L/D` = aerodynamic efficiency
- `alpha_trim` = trim angle at cruise
- `cm_alpha` = stability slope, and you generally want it **negative**
- `V_fuse` = fuselage volume
- `V_target` = desired fuselage volume

---

## Important note about stability sign

For a statically stable aircraft, you usually want:

- `cm_alpha < 0`

So you can penalize only if it becomes positive:

\[
P_\text{stability} = w_s \max(0, cm_\alpha)^2
\]

---

# Suggested extraction from your code

Right now `evaluate_configuration(p)` already computes most of what you need. I’d split it into:

1. a function that computes **geometry / volume**
2. a function that evaluates **aero**
3. a function that returns a **scalar fitness**

---

## Example implementation

```python
def calculate_fuselage_volume_mm3(p):
    l_nose = p["fuse_length"] * p["nose_ratio"]
    l_tail = p["fuse_length"] * p["tail_ratio"]
    l_mid = p["fuse_length"] - l_nose - l_tail
    r_max = p["fuse_max_diam"] / 2.0

    v_nose = (2.0 / 3.0) * np.pi * (r_max ** 2) * l_nose
    v_mid = np.pi * (r_max ** 2) * l_mid
    r_tail_tip = r_max * (1.0 - 0.85)
    v_tail = (1.0 / 3.0) * np.pi * l_tail * (r_max**2 + r_max * r_tail_tip + r_tail_tip**2)

    return v_nose + v_mid + v_tail
```

Then:

```python
def fitness_function(p, target_fuse_volume_mm3, 
                     w_volume=50.0, w_alpha=0.5, w_stability=20.0):
    result = evaluate_configuration(p)

    # Infeasible aerodynamic state
    if np.isnan(result["ld"]):
        return -1e9

    v_fuse = calculate_fuselage_volume_mm3(p)

    # Volume penalty: exact target
    volume_penalty = w_volume * ((v_fuse - target_fuse_volume_mm3) / target_fuse_volume_mm3) ** 2

    # Penalize large trim angle
    alpha_penalty = w_alpha * abs(result["alpha_trim"])

    # Penalize positive cm_alpha (unstable)
    stability_penalty = w_stability * max(0.0, result["cm_alpha"]) ** 2

    fitness = result["ld"] - volume_penalty - alpha_penalty - stability_penalty
    return fitness
```

---

# If you want a hard volume constraint instead

If the fuselage volume must be **at least** a certain value, use:

```python
def fitness_function(p, min_fuse_volume_mm3,
                     w_volume=100.0, w_alpha=0.5, w_stability=20.0):
    result = evaluate_configuration(p)

    if np.isnan(result["ld"]):
        return -1e9

    v_fuse = calculate_fuselage_volume_mm3(p)

    volume_penalty = w_volume * max(0.0, min_fuse_volume_mm3 - v_fuse) ** 2
    alpha_penalty = w_alpha * abs(result["alpha_trim"])
    stability_penalty = w_stability * max(0.0, result["cm_alpha"]) ** 2

    return result["ld"] - volume_penalty - alpha_penalty - stability_penalty
```

---

# Practical recommendation

For a GA, I’d recommend:

- **maximize fitness**
- use a **large penalty** for violating fuselage volume
- use a **large penalty** if `cm_alpha > 0`
- optionally penalize `abs(alpha_trim)` to prefer easy trim

A good starting point is:

- `w_volume = 50 to 500`
- `w_alpha = 0.2 to 1.0`
- `w_stability = 20 to 200`

You’ll need to tune these based on the scale of your `L/D` values.

---

# Best design choice

If you want the cleanest GA setup, use:

- **objective**: maximize `L/D`
- **constraints**:
  - fuselage volume = target or ≥ target
  - `cm_alpha < 0`
  - `alpha_trim` within acceptable range

That is often better than trying to hide everything in one number, but if your GA library expects a single scalar, the penalty form above is the standard way.

If you want, I can rewrite your `test2.py` into a clean GA-ready version with:

- `fitness_function()`
- `calculate_fuselage_volume()`
- optional constraint handling
- a sample loop using `DEAP`, `pygad`, or plain custom GA.
