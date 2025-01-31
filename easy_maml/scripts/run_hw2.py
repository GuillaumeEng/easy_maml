import os
import time

from easy_maml.agents.pg_agent import PGAgent, MAMLAgent

import os
import time

import gym
import numpy as np
import torch
from easy_maml.infrastructure import pytorch_util as ptu

from easy_maml.infrastructure import utils
from easy_maml.infrastructure.logger import Logger
from easy_maml.infrastructure.action_noise_wrapper import ActionNoiseWrapper
import easy_maml.environment


MAX_NVIDEO = 2


def run_training_loop(args):
    logger = Logger(args.logdir)

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment 
    env = gym.make(args.env_name, render_mode=None)
    
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # add action noise, if needed
    if args.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"
        env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    init_agent = {
        "ob_dim":ob_dim,
        "ac_dim":ac_dim,
        "discrete":discrete,
        "n_layers":args.n_layers,
        "layer_size":args.layer_size,
        "gamma":args.discount,
        "learning_rate":args.learning_rate,
        "use_baseline":args.use_baseline,
        "use_reward_to_go":args.use_reward_to_go,
        "normalize_advantages":args.normalize_advantages,
        "baseline_learning_rate":args.baseline_learning_rate,
        "baseline_gradient_steps":args.baseline_gradient_steps,
        "gae_lambda":args.gae_lambda,
        "policy_Ckp":args.policy_Ckp,
        "critics_Ckp":args.critics_Ckp    }
    if args.maml is False:
        agent = PGAgent(**init_agent)
        number_tasks = 1
        number_inner_steps = 1
    else:
        agent = MAMLAgent(**init_agent)
        agent.init_maml(learning_rate=args.learning_rate,
                        outer_lr=args.outer_learning_rate,
                        learn_inner_lr=args.learn_inner_lr,
                        baseline_learning_rate=args.baseline_learning_rate,
                        baseline_outer_lr=args.baseline_outer_learning_rate,)
        number_tasks = env.n_tasks
        number_inner_steps = args.num_inner_steps
        
        #TODO to remove at the end
        key_sampled_layer = list(agent.actor.mean_net_state_dict_ori.keys())[0]

    total_envsteps = 0
    start_time = time.time()



    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")

        maml_info = {}
        for iter_task in range(number_tasks): # Outer loop
            if args.maml is True:
                print(f"\n****** task {iter_task} *******")
                # Change environment task
                env.toggle_task(random=False)

                # Clone the networks parameter
                agent.clone()
                actor_outer_loss_batch=[]
                critic_outer_loss_batch=[]
            for iter_inner_step in range(number_inner_steps): # Inner loop
                print(f"\n*** Inner step {iter_inner_step} ****")

                # Sampling
                trajs, envsteps_this_batch = utils.sample_trajectories(
                    env, agent.actor, args.batch_size, max_ep_len
                )
                total_envsteps += envsteps_this_batch

                # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
                trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

                # update the parameters using the gathered trajectories
                train_inner_step_info: dict = agent.update(
                    obs= trajs_dict["observation"],
                    actions= trajs_dict["action"],
                    rewards= trajs_dict["reward"],
                    terminals= trajs_dict["terminal"],
                    )
                
                #TODO to remove at the end
                if 0:
                    print(f"Actor inner step {iter_inner_step} Task {iter_task} Loss {train_inner_step_info['Actor Loss']}") # you are supposed to have 3 differents params
                    print(f"Actor inner step {iter_inner_step} Task {iter_task} Layer{key_sampled_layer} params ori {ptu.to_numpy(agent.actor.mean_net_state_dict_ori[key_sampled_layer][0])}")
                    print(f"Actor inner step {iter_inner_step} Task {iter_task} Layer{key_sampled_layer} params update {ptu.to_numpy(agent.actor.parameters_save[key_sampled_layer][0])}")
                    # print(key_sampled_layer)
                    # print(agent.actor.mean_net.keys())
                    # print(f"Actor inner step {iter_inner_step} Task {iter_task} Layer{key_sampled_layer} params used for sampling {ptu.to_numpy(agent.actor.mean_net.state_dict()[key_sampled_layer][0])}")
                    print(f"\nActor inner step {iter_inner_step} Task {iter_task} logstd params ori {agent.actor.logstd_ori}")
                    print(f"Actor inner step {iter_inner_step} Task {iter_task} logstd params used {agent.actor.parameters_save['logstd']}")
                    # print(f"Actor inner step {iter_inner_step} Task {iter_task} logstd params used for sampling {ptu.to_numpy(agent.actor.logstd)}")
                    if args.use_baseline is True:
                        print(f"Baseline inner step {iter_inner_step} Task {iter_task} Loss {train_inner_step_info['Baseline Loss']}") # you are supposed to have 3 differents params
                        print(f"Critic inner step {iter_inner_step} Task {iter_task} Layer{key_sampled_layer} params ori {agent.critic.network_state_dict_ori[key_sampled_layer][0]}")
                        print(f"Critic inner step {iter_inner_step} Task {iter_task} Layer{key_sampled_layer} params update {agent.critic.parameters_save[key_sampled_layer][0]}")

            if args.maml is True:
                # sampling
                trajs, envsteps_this_batch = utils.sample_trajectories(
                    env, agent.actor, args.batch_size, max_ep_len
                )
                total_envsteps += envsteps_this_batch

                # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
                trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

                # compute and store outer loss on the task
                agent.maml_outer_loss()
                loss = agent.update(
                    obs= trajs_dict["observation"],
                    actions= trajs_dict["action"],
                    rewards= trajs_dict["reward"],
                    terminals= trajs_dict["terminal"],
                    )
                actor_outer_loss_batch.append(loss["Actor Loss"])  
                if args.use_baseline is True:
                    critic_outer_loss_batch.append(loss["Baseline Loss"])
            
        if args.maml is False:
            # keep loss for future plot
            maml_info.update(train_inner_step_info)
        else:          
            # aggregate loss on each task before doing a backward pass for the actor
            actor_outer_loss = torch.mean(torch.stack(actor_outer_loss_batch))
            for iter, task_loss in enumerate(actor_outer_loss_batch):
                maml_info[f"Actor {iter} Task Loss"]= ptu.to_numpy(task_loss)

            # aggregate loss on each task before doing a backward pass for the critic
            critic_outer_loss=None
            if args.use_baseline is True:
                critic_outer_loss = torch.mean(torch.stack(critic_outer_loss_batch))
                for iter, task_loss in enumerate(critic_outer_loss_batch):
                    maml_info[f"Critic {iter} Task Loss"]= ptu.to_numpy(task_loss)

            # step the optimizer and zero the gradients for both actor/critic
            agent.step(actor_outer_loss, critic_outer_loss)
            maml_info.update(agent.update_maml_info())

        # save model checkpoint
        if (args.test==False) \
            and (((itr > 1) and (itr%args.Ckp_frequency == 0)) or (itr==args.n_iter-1)):
            agent.checkpoint_save()
        
        
        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )
            logs = utils.compute_metrics(trajs, eval_trajs)

            # compute additional metrics
            logs.update(maml_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            logger.flush()

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0 and itr > 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-2)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument("--action_noise_std", type=float, default=0)

    parser.add_argument("--critics_Ckp", type=str, default="ckp_critic.ckp")
    parser.add_argument("--policy_Ckp", type=str, default="ckp_policy.ckp")
    parser.add_argument("--Ckp_frequency", type=int, default=50)

    parser.add_argument("--maml", "-maml", action="store_true")
    parser.add_argument("--test", "-test", action="store_true")
    parser.add_argument("--num_inner_steps", "-nis", type=int, default=3)
    parser.add_argument("--learn_inner_lr", "-lilr", action="store_true")
    parser.add_argument("--outer_learning_rate", "-olr", type=float, default=5e-3)
    parser.add_argument("--baseline_outer_learning_rate", "-bolr", type=float, default=5e-3)

    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    run_training_loop(args)


if __name__ == "__main__":
    main()
