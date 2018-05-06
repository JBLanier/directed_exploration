

from vae import VAE
from staternn import StateRNN
import argparse
import random
import math
import numpy as np
import os
import cv2

ACTION_DIMS = 5

def action_for_index(index):
    actions = [(0, 0), (0.5, 0), (0.5, 90), (0.5, 180), (0.5, -90)]
    action = actions[index]
    return (0, (action[0], math.radians(action[1])))

def main(args):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    vae = VAE()
    vae.load_model(args.load_vae_weights)

    rnn = StateRNN()

    if args.load_rnn_weights:
        rnn.load_model(args.load_rnn_weights)

    game = BoxPush(display_width=64, display_height=64)
    p = ContinousPLE(game, fps=30, display_screen=False, add_noop_action=False)
    p.init()
    total_frames = 0
    total_episodes = 0
    episodes_to_train_on = 300000
    batches = 0

    max_sequence_length = 257

    for j in range(episodes_to_train_on):

        # print("new episode")

        p.reset_game()
        action_index = 0
        p.act(action_for_index(action_index))

        last_observation = None
        last_action = None

        continue_game = True

        while continue_game:


            obs_sequence = np.zeros(shape=(max_sequence_length, 64, 64, 3), dtype=np.float32)
            if last_observation is not None:
                obs_sequence[0] = last_observation

            action_sequence = np.zeros(shape=(max_sequence_length, ACTION_DIMS), dtype=np.float32)
            if last_action is not None:
                action_sequence[0] = last_action

            sequence_length = 0

            while True:

                observation = np.swapaxes(p.getScreenRGB(), 0, 1) / 255.0
                obs_sequence[sequence_length] = observation

                if random.random() < 0.05:
                    action_index = random.randint(0, ACTION_DIMS-1)

                action_sequence[sequence_length, action_index] = 1

                # visualization ###
                if args.visualize and sequence_length >= 1:
                    prev_action_onehot = np.expand_dims(action_sequence[sequence_length-1], axis=0)
                    encoded_prev_obs = vae.vae_encoder.predict(np.expand_dims(obs_sequence[sequence_length-1], axis=0))

                    rnn_input = np.expand_dims(np.concatenate((encoded_prev_obs, prev_action_onehot), axis=1), axis=0)
                    cv2.imshow("predicted", np.squeeze(vae.decode(rnn.rnn.predict(rnn_input)[0]))[:,:,::-1])
                    cv2.waitKey(1)

                    cv2.imshow("actual", np.squeeze(vae.vae.predict(np.expand_dims(observation, axis=0)))[:, :, ::-1])
                    cv2.waitKey(1)

                #####

                p.act(action_for_index(action_index))
                sequence_length += 1

                r = random.random()
                if r < 0.005 or p.game_over():
                    continue_game = False
                    # break

                if sequence_length >= max_sequence_length:
                    break

            # print("batch had {} steps".format(sequence_length))

            last_observation = obs_sequence[-1]
            last_action = action_sequence[-1]

            # print(vae.vae_encoder.predict(obs_sequence).shape)
            encoded_observations = vae.vae_encoder.predict(obs_sequence)
            rnn_sequence = np.expand_dims(np.concatenate((encoded_observations, action_sequence), axis=1), axis=0)
            # print("rnn sequence shape: {}".format(rnn_sequence.shape))

            rnn_inputs = rnn_sequence[:, :-1, :]
            rnn_targets = rnn_sequence[:, 1:, :-ACTION_DIMS]

            # if not np.array_equal(rnn_targets,np.expand_dims(encoded_observations[1:], axis=0)):
            #     print("try again")
            #     print("rrn targers: {}".format(rnn_targets))
            #     print("check: {}".format(np.expand_dims(encoded_observations[1:], axis=0)))
            #
            #     exit(1)
            # print("rnn inputs shape: {}".format(rnn_inputs.shape))
            # print("rnn targets shape: {}".format(rnn_targets.shape))

            print(rnn.rnn.train_on_batch(x=rnn_inputs, y=rnn_targets))
            batches += 1
            if batches % 20 == 0:
                print("Finished batch {}".format(batches))
            if batches % 1000 == 0:
                path = "rnn_weights.hdf5"
                rnn.save_model(path)
                print("saved rnn weights to {}".format(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-vae-weights", help="path to start VAE weights with",
                        type=str, required=True)
    parser.add_argument("--load-rnn-weights", help="path to start RNN weights with",
                        type=str, required=False)
    parser.add_argument("--visualize", help="visualize predictions",
                        action="store_true")
    args = parser.parse_args()
    main(args)