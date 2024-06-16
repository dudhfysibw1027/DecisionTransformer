import numpy as np
import torch

from decision_transformer.training.trainer import Trainer

class SeperateActorCriticTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rtg)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        rwd_dim = reward_preds.shape[2]

        # print(reward_target.shape, reward_preds.shape)
        batch_size, t, _ = reward_target.shape
        # print("first", reward_target[0, 0, 0], ", second", reward_target[0, 1, 0])
        # print("last 2", reward_target[0,  t - 2, 0], " last", reward_target[0, t - 1, 0])

        reward_target = reward_target[:, 0:t-1, :]

        loss = self.loss_fn(
            None, None, reward_preds,
            None, None, reward_target,
        )
        # print(reward_target[:, t, :])
        # print(self.batch_size, " ", print(reward_preds.shape))

        # print("reward pred:", torch.mean(reward_preds, dim=0), ", target", torch.mean(reward_target, dim=0))
        # print("reward pred:", torch.mean(reward_preds, dim=0), ", target", torch.mean(reward_target, dim=0))
        
        # update critic
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # update actor
        mean, log_std = self.actor(states)
        std = log_std.exp()
        mean = mean.reshape(-1, act_dim)
        std = std.reshape(-1, act_dim)
        normal = torch.distributions.Normal(mean, std)
        new_action = normal.sample()
        log_prob = normal.log_prob(new_action).sum(axis=-1, keepdim=True)[attention_mask.reshape(-1) > 0]

        #with torch.no_grad():
        _, _, reward_behave = self.model.forward(
            states, new_action.reshape(self.batch_size, -1, act_dim), rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )
        reward_behave = reward_behave.reshape(-1, rwd_dim)[attention_mask.reshape(-1) > 0]

        actor_loss = (-log_prob*reward_behave).mean()#*reward_behave
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean(
                (action_preds - action_target) ** 2).detach().cpu().item()

        return loss.detach().cpu().item()
