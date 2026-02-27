def merge_groups(I_i, groups):
    merged = False
    new_groups = []
    for G in groups:
        if not I_i.isdisjoint(G):
            G.update(I_i)
            new_groups.append(G)
            merged = True
        else:
            new_groups.append(G)
    if not merged:
        new_groups.append(I_i)
    return new_groups

def complement_partitions(I_hat, p):
    covered = set().union(*I_hat)
    return set(range(p)) - covered

def flatten_purevar_indices(I_hat):
    return sorted(set().union(*I_hat))

