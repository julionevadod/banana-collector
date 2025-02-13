import argparse
import configparser

from banana_collector_rl.src.Agent import Agent


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("conf/conf.ini")

    parser = argparse.ArgumentParser(
        prog="Banana Collector Game: DQN solution",
        description=""""
        Current program aims to solve Banana Collector Unity environment
        using value based methods in the field of reinforcement learning
        """
    )

    parser.add_argument(
        "-i",
        "--iterations",
        default=config["DEFAULT"]["ITERATIONS"],
        help="Number of environment steps to train for."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=config["DEFAULT"]["BATCH_SIZE"],
        help="Batch size for learning steps."
    )
    parser.add_argument(
        "-g",
        "--gamma",
        default=config["DEFAULT"]["GAMMA"],
        help="Discount Rate."
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        default=config["DEFAULT"]["LEARNING_RATE"],
        help="Learning Rate for the Optimizer."
    )
    parser.add_argument(
        "-d",
        "--eps_decay",
        default=config["DEFAULT"]["EPSILON_DECAY"],
        help="Decay of epsilon after each iteration."
    )
    parser.add_argument(
        "-e",
        "--eps_end",
        default=config["DEFAULT"]["EPSILON_END"],
        help="Minimum value of epsilon allowed."
    )

    args = parser.parse_args()

    try:
        iterations = int(args.iterations)
    except ValueError:
        raise ValueError(
            "(i)terations argument should be int. Provided value ({}) cannot be casted.".format(args.iterations)
        )
    if iterations <= 0:
        raise ValueError(
            "(i)terations argument should be a positive integer. Got {}".format(iterations)
        )

    try:
        batch_size = int(args.batch_size)
    except ValueError:
        raise ValueError(
            "(b)atch_size argument should be int. Provided value ({}) cannot be casted.".format(args.batch_size)
        )
    if batch_size <= 0:
        raise ValueError(
            "(b)atch_size argument should be a positive integer. Got {}".format(batch_size)
        )

    try:
        gamma = float(args.gamma)
    except ValueError:
        raise ValueError(
                "(g)amma argument should be a float number between 0 and 1. Provided value ({}) cannot be casted.".format(args.gamma)
            )
    if (gamma < 0) or (gamma > 1):
        raise ValueError(
            "(g)amma argument should be a float number between 0 and 1. Got {}".format(gamma)
        )

    try:
        lr = float(args.learning_rate)
    except ValueError:
        raise ValueError(
                "(l)earning_rate argument should be a float number between 0 and 1. Provided value ({}) cannot be casted.".format(args.learning_rate)
            )
    if (lr < 0) or (lr > 1):
        raise ValueError(
            "(l)earning_rate argument should be a float number between 0 and 1. Got {}".format(lr)
        )

    try:
        eps_decay = float(args.eps_decay)
    except ValueError:
        raise ValueError(
                "(e)ps_decay argument should be a float number between 0 and 1. Got {}".format(args.eps_decay)
            )
    if (eps_decay < 0) or (eps_decay > 1):
        raise ValueError(
            "(e)ps_decay argument should be a float number between 0 and 1.Provided value ({}) cannot be casted.".format(eps_decay)
        )

    try:
        eps_end = float(args.eps_end)
    except ValueError:
        raise ValueError(
                "(e)ps_end argument should be a float number between 0 and 1. Provided value ({}) cannot be casted.".format(args.eps_end)
            )
    if (eps_end < 0) or (eps_end > 1):
        raise ValueError(
            "(e)ps_end argument should be a float number between 0 and 1. Got {}".format(eps_end)
        )

    agent = Agent(
        gamma=gamma,
        lr=lr,
        eps_decay=eps_decay,
        eps_end=eps_end
    )

    scores = agent.learn(
        n_iterations=iterations,
        batch_size=batch_size
    )

    agent.env.close()

    agent.save("checkpoints/")
