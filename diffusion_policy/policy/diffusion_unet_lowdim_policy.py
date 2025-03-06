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
            local_cond=None, global_cond=None, context_cond=None,
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
                local_cond=local_cond, global_cond=global_cond, context_cond=context_cond)

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

        # context_conditioning = None
        # if 'idx' in obs_dict:
        #     
        context_conditioning = obs_dict['idx']
        # to float
        ncond = context_conditioning.float()
        # ncond = self.normalizer['idx'].normalize(context_conditioning)
        # ncond = ncond[:,:To].reshape(ncond.shape[0], -1)

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
            context_cond=ncond,
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

    def compute_loss(self, batch):
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
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def compute_loss_with_cfg(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        
        batch_traj_indices = batch['idx']
        # drop 'idx' from batch
        batch = {k: v for k, v in batch.items() if k != 'idx'}
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        # batch_traj_indices = batch['idx']
        # pdb.set_trace()
        context_cond = batch_traj_indices[:,0,:]
        # pdb.set_trace()

        # compute score with classifier
        ##############################################

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        # context_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            # context_cond = batch_traj_indices[:,:self.n_obs_steps,:].reshape(
            #     batch_traj_indices.shape[0], -1)
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
        pred_with_class = self.model(noisy_trajectory, timesteps,local_cond=local_cond, global_cond=global_cond, 
                                     context_cond=context_cond)
        
        # pred_with_class = self.model(noisy_trajectory, timesteps,local_cond=local_cond, global_cond=global_cond, context_cond=None)
        
        # pdb.set_trace()
        ##############################################
        
        # predict score without classifier
        ##############################################
        # handle different ways of passing observation
        # obs = nbatch['obs']
        # action = nbatch['action']
        # # set last dimension of obs to zeros
        # obs = obs.clone()
        # obs[:,-1] = 0



        # local_cond = None
        # global_cond = None
        # trajectory = action
        # if self.obs_as_local_cond:
        #     # zero out observations after n_obs_steps
        #     local_cond = obs
        #     local_cond[:,self.n_obs_steps:,:] = 0
        # elif self.obs_as_global_cond:
        #     global_cond = obs[:,:self.n_obs_steps,:].reshape(
        #         obs.shape[0], -1)
        #     if self.pred_action_steps_only:
        #         To = self.n_obs_steps
        #         start = To
        #         if self.oa_step_convention:
        #             start = To - 1
        #         end = start + self.n_action_steps
        #         trajectory = action[:,start:end]
        # else:
        #     trajectory = torch.cat([action, obs], dim=-1)

        # # generate impainting mask
        # if self.pred_action_steps_only:
        #     condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        # else:
        #     condition_mask = self.mask_generator(trajectory.shape)

        # # Sample noise that we'll add to the images
        # noise = torch.randn(trajectory.shape, device=trajectory.device)
        # bsz = trajectory.shape[0]
        # # Sample a random timestep for each image
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (bsz,), device=trajectory.device
        # ).long()
        # # Add noise to the clean images according to the noise magnitude at each timestep
        # # (this is the forward diffusion process)
        # noisy_trajectory = self.noise_scheduler.add_noise(
        #     trajectory, noise, timesteps)
        
        # # compute loss mask
        # loss_mask = ~condition_mask

        # # apply conditioning
        # noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # # Predict the noise residual
        # pred_without_class = self.model(noisy_trajectory, timesteps, 
        #     local_cond=local_cond, global_cond=global_cond)
        ##############################################

        # compute aggregate score
        w = 0.0
        # pred = pred_without_class + (w) * (pred_with_class - pred_without_class)
        pred = pred_with_class


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


    def compute_score_divergence(self, trajectory: torch.Tensor, t: torch.Tensor, 
                             local_cond=None, global_cond=None, context_cond=None) -> torch.Tensor:
        """
        Compute the divergence of the score function using the Hutchinson-Skilling trace estimator.

        Args:
            trajectory (torch.Tensor): Input trajectory tensor (B, T, D).
            t (torch.Tensor): Timesteps corresponding to each trajectory sample (B,).
            local_cond (torch.Tensor, optional): Local conditioning input.
            global_cond (torch.Tensor, optional): Global conditioning input.

        Returns:
            torch.Tensor: Estimated divergence of the score function.
        """
        B, T, D = trajectory.shape  # Batch size, trajectory length, feature dim
        device = trajectory.device

        # Sample random Gaussian noise vector v ~ N(0, I)
        v = torch.randn_like(trajectory, device=device)

        # Ensure trajectory requires gradients
        trajectory.requires_grad_(True)

        # Compute score function s_theta(x)
        score = self.model(trajectory, t, local_cond=local_cond, global_cond=global_cond, context_cond=context_cond)  # Output shape (B, T, D)

        # Compute JVP (Jacobian-vector product)
        jvp = torch.autograd.grad(outputs=score, inputs=trajectory, grad_outputs=v, create_graph=True)[0]

        # Estimate divergence as E[v^T (J s_theta) v]
        divergence_estimate = torch.sum(jvp * v, dim=(1, 2))  # Sum over trajectory and feature dimensions

        return divergence_estimate.mean()  # Return the batch mean estimate

    def estimate_gradient(self, trajectory: torch.Tensor, t: torch.Tensor, 
                      local_cond=None, global_cond=None, context_cond=None, num_samples=1) -> torch.Tensor:
        """
        Estimate the gradient of the score function using the Hutchinson-Skilling estimator.

        Args:
            trajectory (torch.Tensor): Input trajectory tensor (B, T, D).
            t (torch.Tensor): Timesteps corresponding to each trajectory sample (B,).
            local_cond (torch.Tensor, optional): Local conditioning input.
            global_cond (torch.Tensor, optional): Global conditioning input.
            num_samples (int, optional): Number of samples for the estimator (default: 1).

        Returns:
            torch.Tensor: Estimated gradient of the score function.
        """
        B, T, D = trajectory.shape  # Batch size, trajectory length, feature dim
        device = trajectory.device

        # Initialize the accumulated gradient
        gradient_estimate = torch.zeros_like(trajectory, device=device)

        for _ in range(num_samples):
            # Sample random Gaussian noise vector v ~ N(0, I)
            v = torch.randn_like(trajectory, device=device)

            # Ensure trajectory requires gradients
            trajectory.requires_grad_(True)

            # Compute score function s_theta(x)
            score = self.model(trajectory, t, local_cond=local_cond, global_cond=global_cond, context_cond=context_cond)  # Output shape (B, T, D)

            # Compute JVP (Jacobian-vector product)
            jvp = torch.autograd.grad(outputs=score, inputs=trajectory, grad_outputs=v, create_graph=True)[0]

            # Estimate the gradient using the unbiased trace approximation
            gradient_estimate += jvp * v

        # Normalize by number of samples
        gradient_estimate /= num_samples

        return gradient_estimate

    def get_log_likelihood(self, batch):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # build input
        device = self.device
        dtype = self.dtype
        batch_traj_indices = batch['idx']
        # drop 'idx' from batch
        batch = {k: v for k, v in batch.items() if k != 'idx'}

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
        
        
        
        
        B, _, D = trajectory.shape
        # Compute log p_T(x_T), which is the prior log-density of standard Gaussian
        T = self.noise_scheduler.config.num_train_timesteps
        log_pT = -0.5 * torch.sum(trajectory ** 2, dim=(1, 2))
        # Initialize log likelihood integral
        log_likelihood_integral = torch.ones_like(trajectory, device=device)*0
        dt = 1.0 / (self.noise_scheduler.config.num_train_timesteps - 1)
        # duplicate to the size of trajectory
        dt_vector = torch.ones_like(trajectory) * dt

        for t in self.noise_scheduler.timesteps:
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            # 1. apply conditioning
            noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, t)
        
            # compute loss mask
            loss_mask = ~condition_mask

            # apply conditioning
            noisy_trajectory[condition_mask] = trajectory[condition_mask]
            

            # 2. predict model output
            score_divergence = self.compute_score_divergence(noisy_trajectory, t, 
                local_cond=local_cond, global_cond=global_cond, context_cond=batch_traj_indices)

            # print("score_divergence: ", score_divergence)
            gradient_estimate = self.estimate_gradient(noisy_trajectory, t, 
                local_cond=local_cond, global_cond=global_cond, context_cond=batch_traj_indices)
            # print("gradient_estimate: ", gradient_estimate)
            
            # Integrate over time (Euler method)
            # pdb.set_trace()
            log_likelihood_integral += dt_vector * gradient_estimate

            # set noisy_trajecotry requires grad to False
            # noisy_trajectory.requires_grad_(False)
            noisy_trajectory = noisy_trajectory.detach()


        # Compute log p_0(x) using reverse SDE formula
        log_p0 = log_pT - log_likelihood_integral

        # convert to bits/dim 
        bits_per_dim = -log_p0 / (T * D * torch.log(torch.tensor(2.0, device=device)))  # Convert to log_2

        # convert to likelihood
        likelihood = torch.exp(log_p0)
        # pdb.set_trace()

        return log_p0



    def get_score(self, batch, timesteps):
        # pdb.set_trace()
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
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (bsz,), device=trajectory.device
        # ).long()
        timesteps = torch.tensor(timesteps).long()
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

        
        return pred