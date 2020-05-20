import argparse
import gym
import pybullet_envs
import numpy as np
import torch
import time
import os


from models import SoftActorCritic
from helper import TimeFeatureWrapper

t0 = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#eval over mean action
def evaluate_agent(agent, eval_num, env, total_timesteps, eval_episodes=1):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			_, _, mean, _ = agent.policy_network.sample_action(torch.Tensor(obs).view(1,-1).to(device))
			obs, reward, done, _ = env.step(mean.detach().cpu().numpy()[0])
			avg_reward += reward

	avg_reward /= eval_episodes

	print ("Eval: %d  Total_timesteps: %d Evaluation: %f,  time-%d " % (eval_num, total_timesteps, avg_reward, int(time.time()-t0)))
	return avg_reward

def main(args):

    # Initialize environment and agent
    env = gym.make(args.env_name)
    
    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = SoftActorCritic(env.observation_space, env.action_space, args)
    i = 0
    ep = 1
    evaluations = [evaluate_agent(agent,0, env, i)] 
    if not args.no_hyper:
        hyperstr = 'Hyper_'
    else:
        hyperstr = ''
    file_name = "%s_%s_seed%s_scale%d" % (hyperstr, args.env_name,args.seed,args.reward_scale)

    print("Start training...")

    while ep >= 1:
        episode_reward = 0
        state = env.reset()
        done = False
        j = 0
        q1_loss=[]; q2_loss=[]; policy_loss=[]; policy_grad=[]; logs=[]; logs1=[]
        
        while not done:
            # sample action
            if i > args.start_timesteps:
                action = agent.get_action(state)
            else:
                action = env.action_space.sample()
            
            if agent.replay_memory.get_len() > args.batch_size: 
                
                # update of X amount for each env step
                if i % args.steps_ratio == 0:
                    for k in range(args.update_ratio):
                        q1, q2, policy_l, grad, q1_pi, q2_pi = agent.update_params(args.batch_size, args.gamma, args.tau, args.grad_clip)
                        q1_loss.append(q1); q2_loss.append(q2); policy_loss.append(policy_l); policy_grad.append(grad); logs.append(q1_pi),logs1.append(q2_pi)


            # prepare transition for replay memory push
            next_state, reward, done, _ = env.step(action)
            reward *= args.reward_scale
            i += 1
            j += 1
            episode_reward += reward

            ndone = 1 if j == env._max_episode_steps else float(not done)
            agent.replay_memory.push((state,action,reward,next_state,ndone))
            state = next_state
        
            # eval episode
            if i % args.eval_freq == 0:
                evaluations.append(evaluate_agent(agent, i / args.eval_freq, env, i))
                print ("alpha: %f" %agent.alpha)
                np.save("./results/%s" % (file_name), evaluations)
                for key in agent.hist:
                    np.save("./histograms/histogram_%s_%s" %(key,file_name), agent.hist[key])
        
        if agent.replay_memory.get_len() > args.batch_size: 
            print("q1: {:.4f} q2: {:.4f} policy: {:.4f} mean_norm: {:.4f} max_norm: {:.4f}".format(
            np.mean(q1_loss), np.mean(q2_loss), np.mean(policy_loss), np.mean(policy_grad), np.max(policy_grad)))

        if np.mean(policy_grad) > 10 and not args.no_hyper:
            agent.reduce_lr(args.lr_drop)
        
        if i >= args.max_time_steps:
            break


        ep += 1
    # save model
    torch.save(agent, "./pytorch_models/%s" % (file_name))
    np.save("./results/%s" % (file_name), evaluations)  

    env.close()


if __name__ == '__main__':

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="HopperBulletEnv-v0",#"HalfCheetahBulletEnv-v0",#'AntBulletEnv-v0', # Walker2DBulletEnv-v0 # HalfCheetahBulletEnv-v0
    #HopperBulletEnv-v0, #HumanoidBulletEnv-v0
        help='name of the environment')
    parser.add_argument("--seed", default=0, type=int,
        help='seed')			
    parser.add_argument("--no_hyper", action="store_true")	# use regular critic
    parser.add_argument('--epsilon', type=float, default=1e-6,
        help='.....')
    parser.add_argument('--hidden_dim', type=int, default=256,
        help='regular hidden layers dim')
    parser.add_argument('--tau', type=float, default=0.005,
        help='soft update param')
    parser.add_argument('--lr', type=float, default=3e-4,
        help='regular lr')
    parser.add_argument('--hyper_lr', type=float, default=5e-5,
        help='hyper lr')
    parser.add_argument('--batch_size', type=int, default=256,
        help='batch_size')
    parser.add_argument('--replay_memory_size', type=int, default=1e6,
        help='replay memoery size')
    parser.add_argument('--max_time_steps', type=int, default=1e6,
        help='num of training time steps')
    parser.add_argument('--alpha', type=float, default=1,
        help='max entropy vs. expected reward')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='discount factor')
    parser.add_argument("--start_timesteps", default=1e3, type=int,
        help='number of timesteps at the start for exploration')
    parser.add_argument('--min_log', type=int, default=-20,
        help='min log')
    parser.add_argument('--max_log', type=int, default=2,
        help='max log')
    parser.add_argument('--reward_scale', type=float, default=1,
        help='reward scale')
    parser.add_argument('--update_ratio', type=int, default=1,
        help='amount of params updates to env steps')
    parser.add_argument('--steps_ratio', type=int, default=1,
        help='amount of env steps for each param update')
    parser.add_argument("--eval_freq", default=1e3, type=float)
    parser.add_argument("--mean_reg", default=1e-4, type=float)
    parser.add_argument("--std_reg", default=1e-4, type=float)
    parser.add_argument('--grad_clip', type=float, default=0,
        help='policy gradient clip with Hyper critic')
    parser.add_argument('--lr_drop', type=float, default=1e-5,
        help='policy lr_drop with Hyper critic')
    parser.add_argument('--entropy_tunning', type=bool, default=False)

    args = parser.parse_args()   

    args.device = device
    print(args)

    main(args)