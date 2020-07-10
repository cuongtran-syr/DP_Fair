Difference between DP_Fair vs DP_Fair_v2

1. We add Gaussian noise to avg of violation list, before using noisy avg violation list to update lambda ( we might need to use large step-size so we ran fewer epochs)
2. We use mean square error instead of log-loss to have bounded sensitivity. The S is now 1. We need S in dual step updates.

3. We will use more groups (number group > 2) here.

4. To speed up we will remove some gradients check here.

