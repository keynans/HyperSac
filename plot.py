import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

def read_log():
  with open("log.txt",'r') as fd:
    temp = fd.readlines()
    lines = [line.split(":") for line in temp if line[0] == 'E']
    evals = np.array([float(evl[3].split(",")[0]) for evl in lines])
    return evals

def plot_hist(args):
    #regular
    hyper = "_Hyper"
    seed = 2
    #env = "__HopperBulletEnv-v0"
    env = "__Walker2DBulletEnv-v0"
    #env = "__AntBulletEnv-v0"
    a0 = np.load('histograms/histogram_linear1.weight.mean'+ hyper + env + '_seed'+ str(seed) + '_scale1.npy')
    a1 = np.load('histograms/histogram_linear2.weight.mean'+ hyper + env + '_seed'+ str(seed) + '_scale1.npy')
    a2 = np.load('histograms/histogram_linear3a.weight.mean'+ hyper + env + '_seed'+ str(seed) + '_scale1.npy')
    a3 = np.load('histograms/histogram_linear3b.weight.mean'+ hyper + env +'_seed'+ str(seed) + '_scale1.npy')
    a4 = np.load('histograms/histogram_linear1.weight.max'+ hyper + env +'_seed'+ str(seed)+ '_scale1.npy')
    a5 = np.load('histograms/histogram_linear2.weight.max'+ hyper + env +'_seed'+ str(seed) + '_scale1.npy')
    a6 = np.load('histograms/histogram_linear3a.weight.max'+ hyper + env +'_seed'+ str(seed) + '_scale1.npy')
    a7 = np.load('histograms/histogram_linear3b.weight.max'+ hyper + env +'_seed'+ str(seed) + '_scale1.npy')
    a8 = np.load('histograms/histogram_0.weight.mean'+ hyper + env + '_seed'+ str(seed) + '_scale1.npy')
    a9 = np.load('histograms/histogram_0.weight.max'+ hyper + env + '_seed'+ str(seed) + '_scale1.npy')


    plt.figure(1)
    plt.title(env+hyper+"_gradient")
    plt.subplot(221)
    plt.xlabel("steps")
    plt.ylabel("mean gradiet")
    plt.plot(range(len(a0)),a0, label="linear1")
    plt.plot(range(len(a1)),a1, label="linear2")
    plt.plot(range(len(a2)),a2, label="linear3a-mean")
    plt.plot(range(len(a3)),a3, label="linear3b-std")
    plt.legend()
    plt.subplot(222)
    plt.xlabel("steps")
    plt.ylabel("max gradient")
    plt.plot(range(len(a4)),a4, label="linear1")
    plt.plot(range(len(a5)),a5, label="linear2")
    plt.plot(range(len(a6)),a6, label="linear3a-mean")
    plt.plot(range(len(a7)),a7, label="linear3b-std")
    #plt.legend()
    plt.subplot(223)
    plt.xlabel("steps")
    plt.ylabel("q1 mean gradiet")
    plt.plot(range(len(a8)),a8, label="last layer w")
    plt.legend()
    plt.subplot(224)
    plt.xlabel("steps")
    plt.ylabel("q1 max gradient")
    plt.plot(range(len(a9)),a9, label="last layer w")
    plt.legend()
  #  plt.show()
    plt.savefig("gradient"+ hyper + env +".png")
    plt.close()  

def plot(args):
    hyper = "Hyper_"
    seed = 0
    #env = "_HopperBulletEnv-v0"
    #env = "_Walker2DBulletEnv-v0"
    #env = "_AntBulletEnv-v0"
    #env = "_HalfCheetahBulletEnv-v0"
    env = "_HumanoidBulletEnv-v0"
    last=1001
    a0 = np.load('results/'+ hyper + env + '_seed'+ str(seed+0) + '_scale1.npy')[:last]
    a1 = np.load('results/'+ hyper + env + '_seed'+ str(seed+1) + '_scale1.npy')[:last]
    a2 = np.load('results/'+ hyper + env + '_seed'+ str(seed+2) + '_scale1.npy')[:last]
    #reg
    b0 = np.load('results/'+ env + '_seed'+ str(seed+0) + '_scale1.npy')
    b1 = np.load('results/'+  env + '_seed'+ str(seed+1) + '_scale1.npy')
    b2 = np.load('results/'+  env + '_seed'+ str(seed+2) + '_scale1.npy')
  #  b3 = np.load('results/Hyper__HopperBulletEnv-v0_seed3_scale1.npy')
  #  b4 = np.load('results/Hyper__HopperBulletEnv-v0_seed4_scale1.npy')

    total = np.vstack([a0, a1 ,a2])
    min_total_reg = np.min(total,axis=0)
    max_total_reg = np.max(total, axis=0)
    avg_total_reg = np.mean(total, axis=0)
    total = np.vstack([b0, b1 ,b2])
    min_total_hyp = np.min(total,axis=0)
    max_total_hyp = np.max(total, axis=0)
    avg_total_hyp = np.mean(total, axis=0)

    plt.title(env)
    plt.xlabel("steps 1e3")
    plt.ylabel("rewards")
    asmoothed = gaussian_filter1d(min_total_reg, sigma=2)
    bsmoothed = gaussian_filter1d(max_total_reg, sigma=2)
    csmoothed = gaussian_filter1d(avg_total_reg, sigma=2)
    dsmoothed = gaussian_filter1d(min_total_hyp, sigma=2)
    esmoothed = gaussian_filter1d(max_total_hyp, sigma=2)
    fsmoothed = gaussian_filter1d(avg_total_hyp, sigma=2)
    plt.plot(range(len(avg_total_reg)), csmoothed, label="Hyper critic 0.01- SAC", color='#CC4F1B')
    plt.fill_between(range(len(min_total_reg)), bsmoothed, asmoothed, facecolor='#FF9848', alpha=0.3)
    plt.plot(range(len(avg_total_hyp)), fsmoothed, label="Regular critic 0.01 - SAC", color='#1B2ACC')
    plt.fill_between(range(len(min_total_hyp)), dsmoothed, esmoothed, facecolor='#089FFF', alpha=0.3)
    plt.legend()
    plt.savefig(hyper + env +".png")
    plt.close()  

def plot_reward_scale(args):
    hyper = "Hyper_"
    seed = 0
    #env = "_HopperBulletEnv-v0"
    #env = "_HalfCheetahBulletEnv-v0"
    #env = "_Walker2DBulletEnv-v0"
    #env = "_AntBulletEnv-v0"
    env = "_HumanoidBulletEnv-v0"
    a0 = np.load(hyper + env + '_seed'+ str(seed+0) + '_scale1.npy')
    a1 = np.load(hyper + env + '_seed'+ str(seed+1) + '_scale1.npy')
    a2 = np.load(hyper + env + '_seed'+ str(seed+2) + '_scale1.npy')

    total = np.vstack([a0, a1, a2])
    min_total = np.min(total,axis=0)
    max_total = np.max(total, axis=0)
    avg_total = np.mean(total, axis=0)

    plt.title(env)
    plt.xlabel("steps 1e3")
    plt.ylabel("rewards")
    asmoothed = gaussian_filter1d(min_total, sigma=2)
    bsmoothed = gaussian_filter1d(max_total, sigma=2)
    csmoothed = gaussian_filter1d(avg_total, sigma=2)

    plt.plot(range(len(avg_total)), csmoothed, label="Hyper_SAC_reward_scale1_alpha0.01", color='#CC4F1B')
    plt.fill_between(range(len(min_total)), bsmoothed, asmoothed, facecolor='#FF9848', alpha=0.3)

    plt.legend()
    plt.savefig(env+".png")
    plt.close()  


if __name__ == "__main__":
    	
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_file", type=str)				# hyper net file
    parser.add_argument("--reg_file", type=str)					# regular net file
    parser.add_argument("--env", type=str)					    # env
    args = parser.parse_args()
   # a0 = read_log()
   # plot_reward_scale(args)
    plot(args)
   # plot_hist(args)

    
