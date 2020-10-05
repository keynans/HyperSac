import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
plt.style.use('ggplot')
import matplotlib

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
    b0 = np.load('results/' + env + '_seed'+ str(seed+0) + '_scale1.npy')[:1001]
    b1 = np.load('results/'+  env + '_seed'+ str(seed+1) + '_scale1.npy')
    b2 = np.load('results/'+  env + '_seed'+ str(seed+2) + '_scale1.npy')
  #  b3 = np.load('results/Hyper__HopperBulletEnv-v0_seed3_scale1.npy')
  #  b4 = np.load('results/Hyper__HopperBulletEnv-v0_seed4_scale1.npy')

    total = np.vstack([a0, a1, a2])
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
    plt.savefig(hyper + env +".pdf")
    plt.close()  

def normlaize_plot(args):
    hyper = "Hyper_"
    seed = 0
    fontsize = 20
    matplotlib.rc('font', size=fontsize)
    norm_reg = []
    norm_hyp = []
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    mid = [707, 1001, 1001, 1001,1001]
    envs = ["_HopperBulletEnv-v0", "_Walker2DBulletEnv-v0", "_AntBulletEnv-v0", "_HalfCheetahBulletEnv-v0"]
    envs_name = ["Hopper", "Walker2D", "Ant", "HalfCheetah"]
    for i,env in enumerate(envs):
        last=1001
        a0 = np.load('results/'+ hyper + env + '_seed'+ str(seed+0) + '_scale1.npy')[:last]
        a1 = np.load('results/'+ hyper + env + '_seed'+ str(seed+1) + '_scale1.npy')[:last]
        a2 = np.load('results/'+ hyper + env + '_seed'+ str(seed+2) + '_scale1.npy')[:last]
        #reg
        b0 = np.load('results/' + env + '_seed'+ str(seed+0) + '_scale1.npy')[:1001]
        b1 = np.load('results/'+  env + '_seed'+ str(seed+1) + '_scale1.npy')
        b2 = np.load('results/'+  env + '_seed'+ str(seed+2) + '_scale1.npy')

        total = np.vstack([a0[:mid[i]], a1[:mid[i]], a2[:mid[i]]])
        total1 = np.vstack([a0[mid[i]:], a1[mid[i]:]])
        min_total_hyp = np.concatenate([np.min(total,axis=0),np.min(total1,axis=0)])
        max_total_hyp = np.concatenate([np.max(total,axis=0),np.max(total1,axis=0)])
        avg_total_hyp = np.concatenate([np.mean(total,axis=0),np.mean(total1,axis=0)])
   
        total = np.vstack([b0, b1 ,b2])
        min_total_reg = np.min(total,axis=0)
        max_total_reg = np.max(total, axis=0)
        avg_total_reg = np.mean(total, axis=0)

        base = avg_total_reg[0]
        top = avg_total_reg[-1] - base
        norm_reg.append((avg_total_reg - base) / top)
        norm_hyp.append((avg_total_hyp - base) / top)

    norm_max_hyp = np.max(norm_hyp,axis=0)
    norm_min_hyp = np.min(norm_hyp,axis=0)
    norm_mean_hyp = np.mean(norm_hyp,axis=0)
    norm_max_reg = np.max(norm_reg,axis=0)
    norm_min_reg = np.min(norm_reg,axis=0)
    norm_mean_reg = np.mean(norm_reg,axis=0)
  #  plt.title("normalized reward - SAC",fontsize=fontsize)
    plt.xlabel("steps")
  #  plt.ylabel("norm rewards")
    asmoothed = gaussian_filter1d(norm_min_reg, sigma=2)
    bsmoothed = gaussian_filter1d(norm_max_reg, sigma=2)
    csmoothed = gaussian_filter1d(norm_mean_reg, sigma=2)
    dsmoothed = gaussian_filter1d(norm_min_hyp, sigma=2)
    esmoothed = gaussian_filter1d(norm_max_hyp, sigma=2)
    fsmoothed = gaussian_filter1d(norm_mean_hyp, sigma=2)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.plot(range(0,len(norm_mean_hyp)*1000,1000), csmoothed, label="SAC", color='#CC4F1B',linewidth=2.5)
    ax.fill_between(range(0,len(norm_mean_hyp)*1000,1000), bsmoothed, asmoothed, facecolor='#FF9848', alpha=0.3)
    ax.plot(range(0,len(norm_mean_hyp)*1000,1000), fsmoothed, label="Hyper SAC", color='#1B2ACC',linewidth=2.5)
    ax.fill_between(range(0,len(norm_mean_hyp)*1000,1000), dsmoothed, esmoothed, facecolor='#089FFF', alpha=0.3)
    ax.grid(True)
    text = ax.text(-0.2,1.01, "", transform=ax.transAxes)
    lgd = ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.2),prop={'size': 20})

    plt.savefig("normalize_SAC.pdf",bbox_extra_artists=(lgd,text), bbox_inches='tight')
    plt.close()  


def plot_all():
  fontsize=20
  matplotlib.rc('font', size=fontsize)
  fig, axs = plt.subplots(1, 4,constrained_layout=True, figsize=(20,5), sharex=True)
  #tsize=fontsize)
  envs = ["_HopperBulletEnv-v0", "_Walker2DBulletEnv-v0", "_AntBulletEnv-v0", "_HalfCheetahBulletEnv-v0"]
  envs_name =  ["Hopper", "Walker2D", "Ant", "HalfCheetah"]
  hyper = 'Hyper_'
  seed=0
  mid = [707, 1001, 1001, 1001]
  for i,env in enumerate(envs):
        a0 = np.load('results/'+ hyper + env + '_seed'+ str(seed+0) + '_scale1.npy')
        a1 = np.load('results/'+ hyper + env + '_seed'+ str(seed+1) + '_scale1.npy')
        a2 = np.load('results/'+ hyper + env + '_seed'+ str(seed+2) + '_scale1.npy')
        #reg
        b0 = np.load('results/' + env + '_seed'+ str(seed+0) + '_scale1.npy')
        b1 = np.load('results/'+  env + '_seed'+ str(seed+1) + '_scale1.npy')
        b2 = np.load('results/'+  env + '_seed'+ str(seed+2) + '_scale1.npy')

        total = np.vstack([a0[:mid[i]], a1[:mid[i]], a2[:mid[i]]])
        total1 = np.vstack([a0[mid[i]:], a1[mid[i]:]])
        min_hyp = np.concatenate([np.min(total,axis=0),np.min(total1,axis=0)])
        max_hyp = np.concatenate([np.max(total,axis=0),np.max(total1,axis=0)])
        avg_hyp = np.concatenate([np.mean(total,axis=0),np.mean(total1,axis=0)])

        total = np.vstack([b0, b1,b2])
        min_reg = np.min(total,axis=0)
        max_reg = np.max(total, axis=0)
        avg_reg = np.mean(total, axis=0)

        axs[i].set_title(envs_name[i])
        d = gaussian_filter1d(min_hyp, sigma=2)
        e = gaussian_filter1d(max_hyp, sigma=2)
        f = gaussian_filter1d(avg_hyp, sigma=2)
        l = gaussian_filter1d(min_reg, sigma=2)
        m = gaussian_filter1d(max_reg, sigma=2)
        n = gaussian_filter1d(avg_reg, sigma=2)
        axs[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axs[i].plot(range(0,len(avg_hyp)*1000,1000), f, label="Hyper (Ours)", color='#CC4F1B',linewidth=2.5)
        axs[i].fill_between(range(0,len(avg_hyp)*1000,1000), d, e, facecolor='#FF9848', alpha=0.4)
        axs[i].plot(range(0,len(avg_hyp)*1000,1000), n, label="Standard net", color='#1B2ACC',linewidth=2.5)
        axs[i].fill_between(range(0,len(avg_hyp)*1000,1000), m, l,facecolor='#089FFF', alpha=0.4)
        axs[i].set_ylim(0,3300)
        axs[i].set_xlabel("steps")
  axs[0].set_ylabel("reward")
  for ax in axs.flat:
      ax.xaxis.set_tick_params(labelsize=fontsize)
      ax.xaxis.offsetText.set_fontsize(fontsize-2)
      ax.yaxis.set_tick_params(labelsize=fontsize)
  for ax in axs.flat:
      ax.label_outer() 
  plt.grid(True)
  
  axs[3].legend(loc='lower right',prop={'size': 19})
 # plt.savefig("total_Order_TD3.pdf")
  plt.savefig("total_plots_SAC.pdf")
  plt.close()  


if __name__ == "__main__":
    	
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_file", type=str)				# hyper net file
    parser.add_argument("--reg_file", type=str)					# regular net file
    parser.add_argument("--env", type=str)					    # env
    args = parser.parse_args()
   # a0 = read_log()
   # plot_reward_scale(args)
   # plot(args)
    normlaize_plot(args)
   # plot_hist(args)
   # plot_all()

    
