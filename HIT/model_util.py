
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy, HITPolicy


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'HIT':
        policy = HITPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'HIT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer