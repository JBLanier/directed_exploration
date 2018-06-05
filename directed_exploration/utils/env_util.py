import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym

def record_write_subproc_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd.startswith('set_record_write:'):
            _, write_dir, prefix = cmd.split(':')
            remote.send(env.set_record_write(write_dir, prefix))
        elif cmd == 'render':
            remote.send(env.render('state_pixels'))
        else:
            raise NotImplementedError


class RecordWriteSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=record_write_subproc_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def set_record_write(self, write_dir, prefix):
        for env_num, remote in enumerate(self.remotes):
            cmd = 'set_record_write:{}:{}_env{}'.format(write_dir, prefix, env_num)
            remote.send((cmd, None))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        return np.stack([remote.recv() for remote in self.remotes])


def make_record_write_subproc_env(env_id, num_env, start_index=0):
    """
    Create a BoxPushSubprocVecEnv.
    """
    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            # env.seed(seed + rank)
            return env

        return _thunk
    # set_global_seeds(seed)
    return RecordWriteSubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_subproc_env(env_id, num_env, start_index=0):
    """
    Create a SubprocVecEnv.
    """
    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            # env.seed(seed + rank)
            return env

        return _thunk

    # set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])