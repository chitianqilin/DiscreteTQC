import sys
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch import Tensor
from torch.nn import functional as F

from typing import List, Optional

# try:
#     import pybullet_envs_gymnasium
# except ImportError:
#     pass


class Config:

    def __init__(self):
        self.exp_name                 = os.path.basename(__file__)
        self.seed                     = 1
        self.torch_deterministic      = True
        self.cuda                     = True
        self.env_id                   = os.path.basename(__file__)
        self.total_timesteps          = int(2e6)
        self.learning_rate            = 1e-4
        self.n_atoms                  = 531
        self.v_min                    = -30.0
        self.v_max                    = 500.0
        self.buffer_size              = int(3e5)
        self.gamma                    = 1 - 1 / 500
        self.target_network_frequency = 2
        self.batch_size               = 512
        self.learning_starts          = 10000
        self.train_frequency          = 1
        self.policy_frequency         = 2
        self.tau                      = 0.005
        self.tensorboard_frequency    = 100


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_atoms, n_atoms=101, v_min=-100, v_max=100, hidden=256):
        super().__init__()
        self.n_atoms = n_atoms
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_atoms = action_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        # self.encode_state = StateEncoder(state_dim, hidden)
        self.encode_state = nn.Linear(state_dim, hidden)
        self.encode_action = nn.Linear(action_dim * action_atoms, hidden)
        self.network = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_atoms),
        )

    def forward(self, x: Tensor, action: Tensor):
        hidden = self.encode_state(x) + self.encode_action(action)
        logits = self.network.forward(hidden)
        logits = logits.view(-1, self.n_atoms)
        pmfs = torch.softmax(logits, dim=-1)
        return pmfs


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_atom, hidden=256):
        super().__init__()
        self.action_dim = action_dim
        self.action_atom = action_atom
        # self.encode_state = StateEncoder(state_dim, hidden)
        self.encode_state = nn.Linear(state_dim, hidden)
        self.network = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim * action_atom),
        )
        # self.network = nn.Linear(hidden, action_dim * action_atom)

    def forward(self, x: Tensor):
        x = self.encode_state(x)
        x = self.network.forward(x)
        logits = x.view(-1, self.action_dim, self.action_atom)
        probs = logits.softmax(dim=-1)
        probs = probs.view(-1, self.action_dim * self.action_atom)
        return probs
    
    def sample(self, probs: Tensor):
        probs = probs.view(-1, self.action_atom)
        probs = Categorical(probs=probs)
        act = probs.sample().view(-1, self.action_dim)
        return act
    
    def eval(self, probs: Tensor):
        probs = probs.view(-1, self.action_atom)
        act = probs.argmax(dim=-1).view(-1, self.action_dim)
        return act


class ReplayBufferManager:
    def __init__(self, device, buffer_size=int(1e6)):
        self.buffer_size = buffer_size
        self.cursor = 0
        self.n_sample = 0
        self.device = device
        
    def add(self):
        cursor = self.cursor
        n_sample = self.n_sample
        buffer_size = self.buffer_size

        self.cursor = (cursor + 1) % buffer_size
        self.n_sample = min(n_sample + 1, buffer_size)

        return cursor
    
    def sample(self, batch_size):
        n_sample = self.n_sample
        sample = torch.randint(0, n_sample, (batch_size, ), device=self.device)
        return sample


if __name__ == "__main__":
    
    args = Config()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    env = gym.make(sys.argv[1])
    eval_env = gym.make(sys.argv[1], render_mode="rgb_array")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    def episode_trigger(episode_id):
        return episode_id % 100 == 0
    eval_env = gym.wrappers.RecordVideo(eval_env, f"videos/{run_name}", episode_trigger)

    writer = SummaryWriter(f"tf-logs/{run_name}")
    txt = open(f"./{run_name}.txt", "a")

    seq_len = 2048
    obs_dim = env.observation_space.shape[0]
    state_dim = obs_dim
    action_dim = env.action_space.shape[0]
    action_atom = 51
    prob_dim = action_atom * action_dim
    discrete_action_map = np.linspace(-1.0, 1.0, action_atom)
    hidden = 2048

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    qf1 = QNetwork(state_dim, action_dim, action_atom, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max, hidden=hidden).to(device)
    qf2 = QNetwork(state_dim, action_dim, action_atom, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max, hidden=hidden).to(device)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    target_qf1 = QNetwork(state_dim, action_dim, action_atom, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max, hidden=hidden).to(device)
    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2 = QNetwork(state_dim, action_dim, action_atom, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max, hidden=hidden).to(device)
    target_qf2.load_state_dict(qf2.state_dict())

    actor = Actor(state_dim, action_dim, action_atom, hidden=hidden).to(device)
    a_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    target_actor = Actor(state_dim, action_dim, action_atom, hidden=hidden).to(device)
    target_actor.load_state_dict(actor.state_dict())
    best_actor = Actor(state_dim, action_dim, action_atom, hidden=hidden).to(device)
    best_actor.load_state_dict(actor.state_dict())

    rb_mgr = ReplayBufferManager(device=device, buffer_size=args.buffer_size)
    rb_obs = torch.empty((args.buffer_size, obs_dim), requires_grad=False, device=device)
    rb_next_obs = torch.empty((args.buffer_size, obs_dim), requires_grad=False, device=device)
    rb_prob = torch.empty((args.buffer_size, prob_dim), requires_grad=False, device=device)
    rb_act = torch.empty((args.buffer_size, action_dim), requires_grad=False, device=device, dtype=torch.long)
    rb_reward = torch.empty((args.buffer_size, 1), requires_grad=False, device=device)
    rb_done = torch.empty((args.buffer_size, 1), requires_grad=False, device=device)
    rb_result = torch.empty((args.buffer_size, 1), requires_grad=False, device=device)
    # mix = torch.linspace(0.0, 1.0, args.batch_size, requires_grad=False, device=device).view(-1, 1)
    start_time = time.time()

    train_step = 0
    env_step = 0
    eval_step = 0
    eval_best = -float("inf")
    need_init = True
    reward_norm = float(sys.argv[2])

    while env_step < args.total_timesteps:

        for collect_actor, n_to_add in [(actor, 10), (best_actor, 0)]:

            for t in range(n_to_add):
                
                if need_init:
                    obs, _ = env.reset()
                    result = 0.0
                    need_init = False

                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    probs = collect_actor.forward(torch.tensor(obs).float().view(1, -1).to(device))
                    action = collect_actor.sample(probs)

                to_env = []
                to_action = action.flatten().cpu().numpy()
                to_prob = probs.flatten().cpu().numpy()
                for act in to_action:
                    to_env.append(discrete_action_map[act])
                to_env = np.array(to_env)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, truncated, info = env.step(to_env)
                env_step += 1
                result += reward
                done = terminated
                reward_norm = max(reward, reward_norm)
                
                cursor = rb_mgr.add()
                rb_obs[cursor]      = torch.tensor(obs).float().to(device)
                rb_next_obs[cursor] = torch.tensor(next_obs).float().to(device)
                rb_prob[cursor]     = torch.tensor(to_prob).float().to(device)
                rb_act[cursor]      = torch.tensor(to_action).to(device)
                rb_reward[cursor]   = torch.tensor(reward).float().to(device)
                rb_done[cursor]     = torch.tensor(float(done)).to(device)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                if terminated or truncated:
                    need_init = True
                    writer.add_scalar("losses/reward_norm", reward_norm, env_step)
                    writer.add_scalar("losses/result", result, env_step)
                    print(f"train {result} {train_step} {env_step}\n")
            
        # ALGO LOGIC: training.
        for _ in range(10):
            if rb_mgr.n_sample > args.learning_starts:
                B = args.batch_size
                b_i: Tensor = rb_mgr.sample(B).view(B, 1)
                b_obs      = rb_obs.gather(0, b_i.expand(B, obs_dim))
                b_next_obs = rb_next_obs.gather(0, b_i.expand(B, obs_dim))
                b_act      = rb_act.gather(0, b_i.expand(B, action_dim))
                b_prob     = rb_prob.gather(0, b_i.expand(B, prob_dim))
                b_reward   = rb_reward.gather(0, b_i) / reward_norm
                b_done     = rb_done.gather(0, b_i)
                b_result   = rb_result.gather(0, b_i)
                
                with torch.no_grad():
                    next_act_probs = target_actor.forward(b_next_obs)
                    # next_atoms = b_reward + args.gamma * target_qf1.atoms * (1 - b_done) - b_done * (args.v_max - args.v_min)
                    next_atoms = b_reward + args.gamma * target_qf1.atoms * (1 - b_done)

                    cumsum_next_pmfs1: Tensor = target_qf1.forward(b_next_obs, next_act_probs).cumsum(-1)
                    cumsum_next_pmfs2: Tensor = target_qf2.forward(b_next_obs, next_act_probs).cumsum(-1)

                    cumsum_next_pmfs = torch.maximum(cumsum_next_pmfs1, cumsum_next_pmfs2)
                    next_pmfs = cumsum_next_pmfs.clone()
                    next_pmfs[:, 1:] = cumsum_next_pmfs[:, 1:] - cumsum_next_pmfs[:, :-1]

                    # projection
                    delta_z = target_qf1.atoms[1] - target_qf1.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)
                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                one_hot: Tensor = F.one_hot(b_act, num_classes=action_atom)
                a3 = one_hot.view(B, -1)
                # a1 = torch.arange(action_atom, device=device).view(1, 1, -1)
                # a2 = b_act.view(B, -1, 1)
                # a3 = - (a1 - a2).float().pow(2) / 9
                # a3 = torch.softmax(a3, dim=-1).view(B, -1)
                b_prob = b_prob.view(B, -1)
                mix = torch.rand((B, action_dim, 1), device=device).pow(np.log(0.25) / np.log(0.5))
                mix = mix.expand(-1, -1, action_atom).reshape(B, -1)
                act = mix * a3 + (1 - mix) * b_prob

                prob = act.view(-1, action_dim, action_atom)
                prob = prob.gather(2, b_act.view(-1, action_dim, 1))
                prob = prob.view(-1, action_dim).prod(dim=-1).pow(1 / action_dim)

                old_pmfs1 = qf1.forward(b_obs, act)
                old_pmfs2 = qf2.forward(b_obs, act)

                q1_loss = (prob * ( - target_pmfs * old_pmfs1.clamp_min(1e-5).log()).sum(-1)).mean()
                q2_loss = (prob * ( - target_pmfs * old_pmfs2.clamp_min(1e-5).log()).sum(-1)).mean()
                q_loss = (q1_loss + q2_loss)

                # optimize the model
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                if train_step % args.policy_frequency == 0:
                    act_probs = actor.forward(b_obs)
                    pmfs = qf1.forward(b_obs, act_probs).cumsum(-1)
                    actor_loss = - (1 - pmfs + 1e-5).log().mean()
                    entropy_loss = torch.ones_like(pmfs) * (- act_probs * act_probs.clamp_min(1e-5).log()).sum(-1).view(-1, 1)
                    entropy_loss = (entropy_loss / (action_dim * np.log(action_atom))).clamp_max(torch.linspace(0.5, 0.0, pmfs.shape[1], device=device).view(1, -1))
                    entropy_loss = - (pmfs.detach() * entropy_loss).mean()
                    a_optimizer.zero_grad()
                    (actor_loss + 0.1 * entropy_loss).backward()
                    a_optimizer.step()

                # update the target network
                if train_step % args.target_network_frequency == 0:
                    for target_param, param in zip(target_qf1.parameters(), qf1.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1.0 - args.tau) * target_param.data
                        )
                    for target_param, param in zip(target_qf2.parameters(), qf2.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1.0 - args.tau) * target_param.data
                        )
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1.0 - args.tau) * target_param.data
                        )

                if train_step % 100 == 0:
                    entropy = (- act_probs * (act_probs + 1e-5).log()).sum(-1).mean().item() / (action_dim * np.log(action_atom))
                    writer.add_scalar("losses/entropy", entropy, train_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), train_step)
                    writer.add_scalar("losses/q1_loss", q1_loss.item(), train_step)
                    writer.add_scalar("losses/q2_loss", q2_loss.item(), train_step)
                    writer.add_scalar("losses/q_loss", q_loss.item(), train_step)

                    old_val1 = (old_pmfs1 * qf1.atoms).sum(1)
                    writer.add_scalar("losses/q1_values", old_val1.mean().item(), train_step)
                    old_val2 = (old_pmfs2 * qf2.atoms).sum(1)
                    writer.add_scalar("losses/q2_values", old_val2.mean().item(), train_step)
                    print("SPS:", int(train_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(train_step / (time.time() - start_time)), train_step)

                if train_step % 25000 == 0:
                    mean = []
                    for _ in range(100):
                        eval_obs, _ = eval_env.reset()
                        eval_result = 0.0
                        for t in range(seq_len - 1):
                            # ALGO LOGIC: put action logic here
                            with torch.no_grad():
                                probs = target_actor.forward(torch.tensor(eval_obs).float().view(1, -1).to(device))
                                action = target_actor.eval(probs)
                            to_env = []
                            to_action = action.flatten().cpu().numpy()
                            to_prob = probs.flatten().cpu().numpy()
                            for act in to_action:
                                to_env.append(discrete_action_map[act])
                            to_env = np.array(to_env)

                            # TRY NOT TO MODIFY: execute the game and log data.
                            eval_next_obs, eval_reward, eval_terminated, eval_truncated, eval_info = eval_env.step(to_env)
                            eval_result += eval_reward

                            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                            eval_obs = eval_next_obs

                            if eval_terminated or eval_truncated:
                                break

                        print("losses/test", eval_result, eval_step)
                        writer.add_scalar("losses/test", eval_result, eval_step)
                        txt.write(f"test {eval_result} {train_step}\n")

                        mean.append(eval_result)
                        eval_step += 1

                    mean = sum(mean) / len(mean)
                    writer.add_scalar("losses/mean", mean, eval_step)

                    if not os.path.isdir("./save"):
                        os.makedirs("./save")
                    if eval_best < mean:
                        eval_best = mean
                        writer.add_scalar("losses/eval_best", eval_best, eval_step)
                        best_actor.load_state_dict(target_actor.state_dict())
                    torch.save({
                        "qf1": qf1.state_dict(),
                        "qf2": qf2.state_dict(),
                        "actor": actor.state_dict(),
                        "target_qf1": target_qf1.state_dict(),
                        "target_qf2": target_qf2.state_dict(),
                        "target_actor": target_actor.state_dict(),
                        "best_actor": best_actor.state_dict(),
                    }, f"./save/{args.exp_name}.pth")

                    txt.flush()
        
            train_step += 1

    env.close()
    writer.close()
    txt.close()
