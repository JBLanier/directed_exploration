from directed_exploration.iterative_exploration import do_iterative_exploration
from directed_exploration.de_logging import init_logging

import datetime
import os

if __name__ == '__main__':

    # working_dir = 'itexplore_20180524183824'
    working_dir = None

    if working_dir:
        root_save_dir = working_dir
    else:
        date_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        root_save_dir = './itexplore_{}'.format(date_identifier)

    init_logging(logfile=os.path.join(root_save_dir, 'events.log'),
                 redirect_stdout=True,
                 redirect_stderr=True,
                 handle_tensorflow=True)

    do_iterative_exploration(env_id='boxpushsimple-v0',
                             num_env=48,
                             num_iterations=1000,
                             latent_dim=1,
                             working_dir=root_save_dir,
                             num_episodes_per_environment=2,
                             max_episode_length=2000,
                             max_sequence_length=200)