#params
# ids: list of ids
# returns: dictionary of pair counts
def get_pair_counts(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# params
# counts: dictionary of pair counts
# returns: most common pair
def get_top_pair(counts):
    return max(counts, key=counts.get)

# params
# ids: list of ids
# pair: pair to merge
# idx: new id to replace the pair
# returns: list of ids with the pair merged
def merge(ids, pair, idx):
    newIds = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] ==  pair[1]:
            newIds.append(idx)
            i += 2
        else:
            newIds.append(ids[i])
            i += 1
    return newIds


"""
tokens = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
stats = get_pair_counts(tokens)
top_pair = get_top_pair(stats)
print(stats)
print(get_top_pair(stats))
print(merge(tokens, top_pair, 5))
"""


