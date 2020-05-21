"""utitlities
"""
from collections import OrderedDict
import numpy as np
from pprint import pprint


def sample_candidate(candidates):
    """pick a candidate from search options
    Arguments:
        candidates {[type]} -- [description]
    Returns:
        [type] -- [description]
    """
    choice_candidates = []
    for candidate in candidates:
        # c = np.random.choice(candidate)
        # choice_candidates.append(c)
        if type(candidate) == int:
            c = candidate
        else:
            c = np.random.choice(candidate)
        choice_candidates.append(c)
    return choice_candidates
def cadidates_to_options(net_n_candidates_dict):
    # print(net_n_candidates_dict)
    model_options = [k[1][0] for k in net_n_candidates_dict.items()]
    model_options = [
        model_options[0][0],
        model_options[0][1],
        model_options[1:8],
        model_options[8],
        model_options[9],
    ]
    return model_options

def options_to_candidates(model_options):
    # block 1
    input_channel = model_options[0]
    last_channel = model_options[1]
    features_block_setting = [input_channel, last_channel]
    # blocks 2 - 8
    encoder_block_settings = model_options[2]
    # block 9
    skip_outs = model_options[3]
    # block 10
    decoder_outs = model_options[4]

    channels_init = np.arange(16, 256, 8)  # 16 - 512
    repeats_init = np.arange(1, 5)  # 1 - 6
    t_factor_init = np.arange(1, 5)  # 1 - 6

    # build network dictionary and search space
    net_n_candidates_dict = OrderedDict()
    net_n_candidates_dict[1] = (
        features_block_setting,
        [np.arange(16, 64, 8), np.arange(16, 256, 8)],
    )
    b_i = 1
    for block in encoder_block_settings:
        b_i += 1
        net_n_candidates_dict[b_i] = (
            block,
            [t_factor_init, channels_init, repeats_init, block[-1]],
        )

    b_i += 1
    net_n_candidates_dict[b_i] = (
        skip_outs,
        [channels_init, channels_init, channels_init],
    )
    b_i += 1
    net_n_candidates_dict[b_i] = (
        decoder_outs,
        [channels_init, channels_init, np.arange(16, 128, 8)],
    )

    return net_n_candidates_dict


if __name__ == "__main__":
    model_options = [
        16,
        80,
        [
            [1, 16, 1, 1],
            [2, 64, 1, 2],
            [2, 32, 1, 2],
            [2, 80, 1, 2],
            [6, 96, 3, 1],
            [6, 128, 2, 2],
            [6, 256, 1, 1],
        ],
        [224, 160, 80],
        [128, 128, 64],
    ]

    search_candidates = options_to_candidates(model_options)
    # pprint(search_candidates)
    for key in range(1, 11):
        print("\nBlock: ", key)
        init_params, candidates = search_candidates[key]
        print("init params: {}, candidates: {}".format(init_params, candidates))