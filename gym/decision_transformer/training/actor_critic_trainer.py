import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class ActorCriticTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rtg)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        # get masked action from gaussian actor
        mean_preds, std_preds = action_preds
        act_dim = mean_preds.shape[2]
        # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        mean_preds = mean_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        std_preds = std_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # get return-to-go
        batch_size, t, _ = reward_target.shape
        reward_target = reward_target[:, 0:t - 1, :]
        rwd_dim = reward_preds.shape[2]
        reward_preds = reward_preds.reshape(-1, rwd_dim)[attention_mask.reshape(-1) > 0]
        reward_target = reward_target.reshape(-1, rwd_dim)[attention_mask.reshape(-1) > 0]

        # get action
        action_dist = torch.distributions.Normal(mean_preds, std_preds)
        action_log_probs = action_dist.log_prob(action_target).sum(-1, keepdim=True)
        action_entropy = action_dist.entropy().sum(-1, keepdim=True)

        # actor update
        # self.optimizer.zero_grad()
        # actor_loss = - (action_log_probs * (reward_target - reward_preds.detach())).mean() - 0.01 * action_entropy.mean()
        # actor_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # self.optimizer.step()

        # critic update
        # self.critic_optimizer.zero_grad()
        # critic_loss = torch.nn.functional.mse_loss(reward_preds, reward_target)
        # critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.predict_return.parameters(), .25)
        # self.critic_optimizer.step()

        # actor and critic update
        self.optimizer.zero_grad()
        actor_loss = - (
                action_log_probs * (reward_target - reward_preds.detach())).mean() - 0.01 * action_entropy.mean()
        actor_loss.backward(retain_graph=True)
        critic_loss = torch.nn.functional.mse_loss(reward_preds, reward_target)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            # self.diagnostics['training/action_error'] = torch.mean(
            #     (mean_preds - action_target) ** 2).detach().cpu().item()
            self.diagnostics['training/critic_loss'] = critic_loss.detach().cpu().item()
            self.diagnostics['training/actor_loss'] = actor_loss.detach().cpu().item()

        return actor_loss.detach().cpu().item() + critic_loss.detach().cpu().item()
