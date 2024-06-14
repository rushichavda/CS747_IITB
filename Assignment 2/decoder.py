import os 
import argparse

ACTIONS = [0,1,2,3,4,5,6,7,8,9]

def parse_states_file(filepath):
    with open(filepath) as f:
        statelist = []
        for line in f:
            statelist.append(line.strip())
    return statelist


def parse_valuepolicy_file(filepath):
    with open(filepath) as f:
        value_policy = []
        for line in f:
            tmp = line.split()
            value_policy.append( ( float(tmp[0]), ACTIONS[int(tmp[1])] ) )
    return value_policy

def decode(statelist, value_policy):
    contents = []
    for state, vp in zip(statelist, value_policy):
        contents.append(f"{state} {vp[1]} {vp[0]}")

    return contents

def parse_arguments(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help="Path to states file")
    parser.add_argument('--value-policy', type=str, required=True, help="Path to value policy file")

    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    statelist = parse_states_file(args.states)
    vp_list = parse_valuepolicy_file(args.value_policy)
    res = decode(statelist=statelist, value_policy=vp_list)

    print("\n".join(res))