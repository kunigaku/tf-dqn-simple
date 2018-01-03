import numpy as np

from catch_ball import CatchBall
from dqn_model import get_dqn_model


if __name__ == "__main__":
    # environment, agent
    env = CatchBall()
    dqn = get_dqn_model(env)
    # After training is done, we save the final weights.
    dqn.load_weights('dqn_{}_weights.h5f'.format("catch_ball"))
    dqn.test(env, nb_episodes=50, visualize=True)
