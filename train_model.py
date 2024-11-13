import torch
import torch.optim as optim
import numpy as np

from rollouts import collect_traj_MBRL
from utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.MSELoss()




def train_single(num_epochs, num_batches,batch_size, model, optimizer, replay_buffer):
    for epoch in range(num_epochs):

        for i in range(num_batches):
            optimizer.zero_grad()
            t1_observations, t1_actions, _, t1_next_observations, _ = replay_buffer.sample(batch_size)
            oa_in = torch.cat([t1_observations, t1_actions], dim=-1)

            next_o_pred = model(oa_in)
            loss = loss_fn(next_o_pred, t1_next_observations)

            loss.backward()
            optimizer.step()


def train_model(model, replay_buffer, optimizer, num_epochs=500, batch_size=32):
    """
    Train a single model with supervised learning
    """
    idxs = np.array(range(len(replay_buffer)))
    num_batches = len(idxs) // batch_size
    if not isinstance(model, list):
        train_single(num_epochs, num_batches, batch_size, model, optimizer, replay_buffer)
    else:   
        #========== TODO: start ==========
        # Write code to train the ensemble of models.
        # Hint1: Each model should have a different batch size for each model
        # Hint2: check out how we define optimizer and model for ensemble models. 
        # During training, each model should have their individual optimizer to increase diversity.
        # Hint3: You can use the train_single function to train each model.
        pass 
        #========== TODO: end ==========
        
        
        

# Training loop for policy gradient
def simulate_mbrl(env, model, plan_mode, num_epochs=200, max_path_length=200, mpc_horizon=10, n_samples_mpc=200, 
                  batch_size=100, num_agent_train_epochs_per_iter=1000, capacity=100000, num_traj_per_iter=100, gamma=0.99, print_freq=10, device = "cuda", reward_fn=None):

    # Set up optimizer and replay buffer
    if not isinstance(model, list):
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

    else:
        print('Initialize separate optimizers for ensemble mbrl')

        optimizer = []
        for model_id, curr_model in enumerate(model):
            optimizer.append(optim.Adam(curr_model.parameters(), lr=(model_id+1)*1e-4)) # use separate optimizer and apply different learning rate to each model

    replay_buffer = ReplayBuffer(obs_size = env.observation_space.shape[0],
                                 action_size = env.action_space.shape[0], 
                                 capacity=capacity, 
                                 device=device)

    # Iterate through data collection and planning loop
    for iter_num in range(num_epochs):
        # Sampling trajectories
        sample_trajs = []
        if iter_num == 0:
            # Seed with some initial data, collecting with mode random
            for it in range(num_traj_per_iter):
                sample_traj = collect_traj_MBRL(env=env,
                                                model=model,
                                                plan_mode='random',
                                                replay_buffer=replay_buffer,
                                                device=device,
                                                episode_length=max_path_length,
                                                reward_fn=reward_fn, #Reward function to evaluate
                                                render=False,
                                                mpc_horizon=None,
                                                n_samples_mpc=None)
                sample_trajs.append(sample_traj)
        else:
            for it in range(num_traj_per_iter):
                sample_traj = collect_traj_MBRL(env=env,
                                                model=model,
                                                plan_mode=plan_mode,
                                                replay_buffer=replay_buffer,
                                                device=device,
                                                episode_length=max_path_length,
                                                reward_fn=reward_fn, #Reward function to evaluate
                                                render=False,
                                                mpc_horizon=mpc_horizon,
                                                n_samples_mpc=n_samples_mpc)
                sample_trajs.append(sample_traj)

        # Train the model
        train_model(model, replay_buffer, optimizer, num_epochs=num_agent_train_epochs_per_iter, batch_size=batch_size)

        # Logging returns occasionally
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))