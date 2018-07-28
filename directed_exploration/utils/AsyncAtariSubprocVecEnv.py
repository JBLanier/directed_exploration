import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import CloudpickleWrapper
from baselines.common.tile_images import tile_images
import logging

logger = logging.getLogger(__name__)


def atari_subproc_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        msg = remote.recv()
        cmd, data = msg
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'clone_full_state':
            state = env.unwrapped.clone_full_state()
            remote.send(state)
        elif cmd == 'restore_full_state':
            env.unwrapped.restore_full_state(data)
            remote.send(True)
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
        elif cmd == 'render':
            remote.send(env.render(mode=data))
        else:
            raise NotImplementedError


class AtariSubprocEnvHandle:
    def __init__(self, remote):
        self.remote = remote
        self.remote.send(('get_spaces', None))
        self.observation_space, self.action_space = self.remote.recv()

    def step(self, action):
        self.remote.send(('step', action))
        results = self.remote.recv()
        return results

    def reset(self):
        self.remote.send(('reset', None))
        return self.remote.recv()

    def reset_task(self):
        self.remote.send(('reset_task', None))
        return self.remote.recv()

    def render(self, mode='human'):
        self.remote.send(('render', mode))
        return self.remote.recv()

    def clone_full_state(self):
        self.remote.send(('clone_full_state', None))
        return self.remote.recv()

    def restore_full_state(self, state):
        self.remote.send(('restore_full_state', state))
        return self.remote.recv()

    def close(self):
        # Call asyncAtariSubprocEnv.close_all()
        raise NotImplementedError


class AsyncAtariSubprocEnv:
    def __init__(self, env_fns, spaces=None):

        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=atari_subproc_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

        self.env_handles = [AtariSubprocEnvHandle(remote) for remote in self.remotes]
        self.nenvs = nenvs

    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()

    def _step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def _step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        for pipe in self.remotes:
            pipe.send(('render', 'rgb_array'))
        imgs = [pipe.recv() for pipe in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:,:,::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def restore_full_states(self, states):
        for remote, state in zip(self.remotes, states):
            remote.send(('restore_full_state', state))
        return np.stack([remote.recv() for remote in self.remotes])

    def clone_full_states(self):
        for remote in self.remotes:
            remote.send(('clone_full_state', None))
        return np.stack([remote.recv() for remote in self.remotes])