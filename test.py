
from IPVC import IPVC 
from stable_baselines3 import SAC, DQN
import numpy as np
import matplotlib.pyplot as plt

def plot():

    th_in = [0*np.pi/180, 135*np.pi/180, 270*np.pi/180]

    fig1, axes1 = plt.subplots(nrows=len(th_in),ncols=1)
    fig1.subplots_adjust(hspace=.6)

    fig2, axes2 = plt.subplots(nrows=len(th_in),ncols=1)
    fig2.subplots_adjust(hspace=.6)

    env = IPVC(fscale=fscale, maxsteps=maxsteps, discrete=discrete, ivp_dt=ivp_dt, test=True)

    for theta, ax1, ax2 in zip(th_in,axes1.flatten(),axes2.flatten()):
        

        state = env.reset(theta)

        angle=[]
        tau=[]
        ep_ret=0

        for i in range(env.maxsteps):
            # env.render()
            action, _states = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            tau.append(env.tau)
            th = (next_state[0] + np.pi)%(2*np.pi) - np.pi
            angle.append(th*180/np.pi)
            ep_ret += reward
            state=next_state
            if done:
                print(f"Angle:{theta*180/np.pi:.2f}, Reward:{ep_ret:.2f}")
                break
    
        x= np.array([i for i in range(1,len(angle)+1)])

        ax1.plot(x,tau)

        if theta==135*np.pi/180:
            np.save(f'tau_{theta*180/np.pi}_{env.maxsteps}_{int(discrete)}.npy',tau)

        ax1.set_title(f'Initial Angle:{theta*180/np.pi:.2f} degrees')
        ax1.set_ylabel('Force')
        # env.close()

        ax2.plot(x,angle)
        ax2.set_title(f'Initial Angle:{theta*180/np.pi:.2f} degrees')
        ax2.set_ylabel('Angle')
        ax2.set_ylim(-10,10)

        env.close()
    if discrete:
        fig1.suptitle(f'Discrete action space, Force ={fscale}')
        fig2.suptitle(f'Discrete action space, Force ={fscale}')
    else:
        fig1.suptitle(f'Continuous action space, Force ={fscale}')
        fig2.suptitle(f'Continuous action space, Force ={fscale}')
    fig1.savefig(f'force_{maxsteps}_{fscale}_{int(discrete)}.png')
    fig2.savefig(f'angle_{maxsteps}_{fscale}_{int(discrete)}.png')
    plt.show()


to_plot = True
to_test= True

fscale = 100

#True for DQN, False for SAC
discrete = True
# discrete = False

ivp_dt=0.02
maxsteps = 5000

env = IPVC(fscale=fscale, maxsteps=maxsteps, discrete=discrete, ivp_dt=ivp_dt, test=True)
print(env.test)

# model = DQN.load('/home/vamsi/Desktop/VCP/dqn1_1270000_steps.zip')
# model = DQN.load('/home/vamsi/Desktop/VCP/dqn3_1160000_steps.zip')
model = DQN.load('./dqn4_790000_steps.zip')
# model = SAC.load('./sac1_1420000_steps.zip')

if to_test:
    th = 15*np.pi/180
    ep_ret = 0
    nep=1
    rewards=[]
    for k in range(nep):
        max_ang=0
        state = env.reset(th)
        ep_ret=0
        for i in range(env.maxsteps):
            # env.render()
            action, _states = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            ang = (next_state[0] + np.pi)%(2*np.pi) - np.pi
            if abs(ang)>max_ang and i>500:
                max_ang=abs(ang)
            ep_ret += reward
            state=next_state
            if done:
                rewards.append(ep_ret)
                print(f"Angle:{th*180/np.pi:.3f}, Episode return: {ep_ret:.3f}, Steps:{i+1}, Max deviation: {max_ang*180/np.pi:.2f}")
                th+=15*np.pi/180
                break
    print("Average reward: {:.3f}".format(np.mean(rewards)))
    env.close()

if to_plot:
    plot()