from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from typing import Union, Tuple
import pdb
from scipy.integrate import quad
# from diffusers.schedulers.scheduling_ddpm.DDPMScheduler import DDPMScheduler

class AccessDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    

    def get_log_likelihood(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            generator=None,
            return_dict: bool = True,
        ) -> Union[torch.Tensor, Tuple]:
        """
        Compute a closed-form estimate of the likelihood function for the reverse SDE.
        
        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a likelihood estimate tensor.

        Returns:
            `torch.Tensor`:
                The log-likelihood estimate of the sample under the reverse SDE.
        """
        t = timestep
        prev_t = self.previous_timestep(t)
        
        # Split model output if variance is learned
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # Compute relevant coefficients
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # Compute predicted original sample (x_0)
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError("Invalid prediction_type")

        # Compute mean of q(x_{t-1} | x_t, x_0)
        mean_xt = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t * pred_original_sample \
                + (current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t) * sample
        
        # Compute variance of q(x_{t-1} | x_t, x_0)
        variance = self._get_variance(t, predicted_variance=predicted_variance)
        
        # Compute log-likelihood
        log_likelihood = -0.5 * torch.sum((sample - mean_xt) ** 2 / variance + torch.log(2 * torch.pi * variance), dim=1)
        
        return log_likelihood
        


    



class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler

        # self.access_noise_scheduler = AccessDDPMScheduler()
        # pdb.set_trace()
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None, 
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        # pdb.set_trace()

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        # pdb.set_trace()
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        
        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition through global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # pdb.set_trace()

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, human_labels=None):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')

        # check if obs is similar to any keys in human labels
        # batch_loss_scalar = torch.ones(loss.shape, device=loss.device)
        # if human_labels is not None and 'avoid' in human_labels:
        #     type_key = 'avoid'
            
        #     for batch_idx in range(obs.shape[0]):
        #         for time_idx in range(obs.shape[1]):
        #             for idx in range(len(human_labels[type_key])):
        #                 check_obs = human_labels[type_key][idx][0][0][0]
        #                 # convert to tensor
        #                 check_obs = torch.tensor(check_obs).to(obs.device)
        #                 # normalize
        #                 check_obs = self.normalizer['obs'].normalize(check_obs)
        #                 dist_from_obs_check = torch.dist(obs[batch_idx, time_idx], check_obs)
        #                 # check_action
        #                 # pdb.set_trace()
        #                 check_action = human_labels[type_key][idx][2][0]
        #                 # convert to tensor
        #                 check_action = torch.tensor(check_action).to(action.device)
        #                 # normalize
        #                 check_action = self.normalizer['action'].normalize(check_action)
        #                 dist_from_action_check = torch.dist(action[batch_idx, time_idx], check_action)

        #                 dist_from_check = dist_from_obs_check + dist_from_action_check
        #                 # print("dist_from_check", dist_from_check)
        #                 if dist_from_check < 0.5:
        #                     print("incentivizing against", dist_from_check)
        #                     batch_loss_scalar[batch_idx][time_idx] *= -1
        #                     break

        # loss = loss * batch_loss_scalar
        batch_loss_scalar = torch.ones(loss.shape, device=loss.device)

        if human_labels is not None and 'avoid' in human_labels:
            type_key = 'avoid'
            avoid_labels = human_labels[type_key]
            
            # Extract and stack check observations and actions from the human labels.
            # Adjust the list comprehensions if the nesting differs.
            check_obs_list = [label[0][0][0] for label in avoid_labels]
            check_action_list = [label[2][0] for label in avoid_labels]
            
            # Convert lists to tensors on the appropriate device.
            check_obs_tensor = torch.tensor(check_obs_list, device=obs.device, dtype=obs.dtype)  # shape: (L, obs_dim)
            check_action_tensor = torch.tensor(check_action_list, device=action.device, dtype=action.dtype)  # shape: (L, action_dim)
            
            # Normalize the human label tensors using your normalizer.
            check_obs_tensor = self.normalizer['obs'].normalize(check_obs_tensor)
            check_action_tensor = self.normalizer['action'].normalize(check_action_tensor)
            
            # Compute Euclidean distance between each observation in obs and each check_obs.
            # Expand dimensions to allow broadcasting:
            #   obs: (B, T, obs_dim) -> (B, T, 1, obs_dim)
            #   check_obs_tensor: (L, obs_dim) -> (1, 1, L, obs_dim)
            obs_exp = obs[:,-1].unsqueeze(1)  # (B, 1, obs_dim )
            check_obs_exp = check_obs_tensor.unsqueeze(0)  # (1, 1, L, obs_dim)
            diff_obs = obs_exp - check_obs_exp  # (B, L, obs_dim)
            dist_obs = torch.norm(diff_obs, dim=-1)  # (B, T, L)

            # take min over the last dimension
            dist_obs = dist_obs.min(dim=-1).values
            # unsqueeze to make it (B, T, 1)
            dist_obs = dist_obs.unsqueeze(1) # (B, 1)
            
            # Do the same for the actions:
            #   action: (B, T, action_dim) -> (B, T, 1, action_dim)
            #   check_action_tensor: (L, action_dim) -> (1, 1, L, action_dim)
            # flatten
            action_exp = action[:,:self.n_action_steps].unsqueeze(1)  # (B, 1, T, action_dim)
            check_action_exp = check_action_tensor.unsqueeze(0)  # (1, L, T, action_dim)

            diff_action = action_exp - check_action_exp  # (B, L, T, action_dim)
            dist_action = torch.norm(diff_action, dim=-1)  # (B, L, T)
            
            # get min over the last dimension
            dist_action = dist_action.min(dim=1).values
            # repea

            # Sum the distances from obs and action for each human label candidate.
            # dist_obs is (B, T, L) and dist_action is (B, T, L).
            dist_total = dist_obs + dist_action  # (B, T/2)
            
            # Create a mask where the minimum distance is below the threshold.
            mask = dist_total < 0.3  # (B, T), boolean

            # compute number of times the condition is met
            num_times_condition_met = mask.sum()
            if num_times_condition_met > 0:
                print("num_times_condition_met", num_times_condition_met)

            # repeat mask to (B, T, 2)
            mask = mask.unsqueeze(-1).repeat(1, 1, 2)
            # currently mask is (B, T/2, 2). We need to repeat it to (B, T, 2)
            mask = mask.repeat(1, 2, 1)
            
            # Update the batch loss scalar: multiply by -1 where the condition is met.
            # batch_loss_scalar[mask] = -1.0
            # set batch_loss to -1 where the condition is met and 1 otherwise
            batch_loss_scalar = torch.where(mask, -1.0 * torch.ones_like(batch_loss_scalar), torch.ones_like(batch_loss_scalar))

        # Finally, scale the loss.
        loss = loss * batch_loss_scalar
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    