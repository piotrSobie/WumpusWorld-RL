from wumpusEnvs.wumpusEnvLv1 import WumpusWorldLv1
from wumpusEnvs.wumpusEnvLv2 import WumpusWorldLv2
from wumpusEnvs.wumpusEnvLv3v1 import WumpusWorldLv3v1
from wumpusEnvs.wumpusEnvLv3v2 import WumpusWorldLv3v2
from wumpusEnvs.wumpusEnvLv3v3 import WumpusWorldLv3v3
from wumpusEnvs.wumpusEnvLv4 import WumpusWorldLv4

from manualPlay import manual_play_lv1, manual_play_lv2_plus
from qLearn import q_learn
from dqn.dqnAlg import dqn_algorithm


if __name__ == '__main__':
    # manual_play_lv1()

    # examined_env = WumpusWorldLv1()
    # examined_env = WumpusWorldLv2()
    # examined_env = WumpusWorldLv3v1()
    # examined_env = WumpusWorldLv3v2()
    # examined_env = WumpusWorldLv3v3()
    examined_env = WumpusWorldLv4()

    manual_play_lv2_plus(examined_env)
    # q_learn(examined_env, show_learned_path=False)
    # dqn_algorithm(examined_env)
