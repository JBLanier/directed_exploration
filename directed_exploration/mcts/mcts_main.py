from directed_exploration.mcts.Coach import Coach
from directed_exploration.mcts.mcts_cnn import MCTS_CNN
from directed_exploration.utils.AsyncAtariSubprocVecEnv import AsyncAtariSubprocEnv
from directed_exploration.logging_ops import init_logging, get_logger
from directed_exploration.utils.data_util import DotDict

import os
import datetime
import tensorflow as tf

from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind



args = DotDict({
    'numIters': 1000,
    # 'tempThreshold': 15,
    # 'updateThreshold': 0.6,
    # 'maxlenOfQueue': 200000,
    'numMCTSSims': 36,
    'temp': 0.0,
    # 'arenaCompare': 40,
    'cpuct': 1,
    'num_batches': 1000,
    'batch_nsteps': 5,

    # 'checkpoint': './temp/',
    # 'load_model': False,
    # 'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def make_async_atari_env(env_id, num_env, seed, monitor_dir, start_index=0):
    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            # import gym
            # return gym.make(env_id)

            env = make_atari(env_id)
            env.seed(seed + rank)
            # env = Monitor(env, os.path.join(monitor_dir, str(rank)))
            return wrap_deepmind(env, scale=True, frame_stack=False)

        return _thunk

    set_global_seeds(seed)
    return AsyncAtariSubprocEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":

    working_dir = None

    if working_dir is None:
        date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        working_dir = './mcts_{}'.format(date_identifier)

    monitor_dir = os.path.join(working_dir, 'monitor')
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir, exist_ok=True)

    init_logging(
        logfile=os.path.join(working_dir, 'events.log'),
        redirect_stdout=True,
        redirect_stderr=True,
        handle_tensorflow=True
    )

    logger = get_logger()

    logger.info("Working dir: {}".format(working_dir))

    subproc_env_group = make_async_atari_env(
        env_id="PongNoFrameskip-v4",
        num_env=12,
        seed=42,
        monitor_dir=monitor_dir
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(working_dir)

    nnet = MCTS_CNN(
        obs_space=subproc_env_group.observation_space,
        action_space=subproc_env_group.action_space,
        working_dir=working_dir,
        sess=sess,
        summary_writer=summary_writer
    )

    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(subproc_env_group, nnet, args, summary_writer)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples()
    c.learn()
    subproc_env_group.close()
