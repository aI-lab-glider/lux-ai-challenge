# Notes

```python
def format_action_vec(a: np.ndarray):
    # (0 = move, 1 = transfer X amount of R, 2 = pickup X amount of R, 3 = dig, 4 = self destruct, 5 = recharge X)
    a = np.array(a).astype(int)
    a_type = a[0]
    if a_type == 0:
        act = MoveAction(a[1], dist=1, repeat=a[4], n=a[5])
    elif a_type == 1:
        act = TransferAction(a[1], a[2], a[3], repeat=a[4], n=a[5])
    elif a_type == 2:
        act =  PickupAction(a[2], a[3], repeat=a[4], n=a[5])
    elif a_type == 3:
        act = DigAction(repeat=a[4], n=a[5])
    elif a_type == 4:
        act = SelfDestructAction(repeat=a[4], n=a[5])
    elif a_type == 5:
        act =  RechargeAction(a[3], repeat=a[4], n=a[5])
    else:
        raise ValueError(f"Action {a} is invalid type, {a[0]} is not valid")
    return act
```

# TODO

1. Execute on kaggle RL agent vs normal agent.
2. Implement:
>
> - Use heuristic, that minimizes distance to the most important resources
> - Use MCTS for agents

3. Sumbit RL agent that will use that heuristic
