import numpy as np
import datetime

from catch_ball import CatchBall
from dqn_model import get_dqn_model


if __name__ == "__main__":
    # environment, agent
    env = CatchBall()
    dqn = get_dqn_model(env)

    try:
        dqn.load_weights('dqn_{}_weights.h5f'.format("catch_ball"))
        print("start from saved weights")
    except:
        print("start from random weights")

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format("catch_ball"), overwrite=True)
    dqn.save_weights('dqn_{}_weights.{}.h5f.bak'.format(
        "catch_ball", datetime.datetime.today().strftime("%Y-%m%d-%H%M")
    ), overwrite=True)
