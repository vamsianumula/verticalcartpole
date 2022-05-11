import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
from IPVC import IPVC
from stable_baselines3 import SAC, DQN


#plot1
def plot1(csv):
	data = pd.read_csv(csv)

	x = data.loc[:,'Step'].values
	y = data.loc[:,'Value'].values

	window_width = 100
	cumsum_vec = np.cumsum(np.insert(y, 0, 0)) 
	ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
	plt.figure(1)
	plt.plot(x,y)
	plt.plot(x[window_width-1:],ma_vec)
	plt.legend(['Reward', 'Average reward per 100 episodes'])
	plt.title(f'Learning curve: {name}')
	plt.savefig(f'{name}_1.png')
	plt.xlabel('Timestep')

def plot23(theta=135*np.pi/180):

	state = env.reset(theta)
	angle=[]
	tau=[]
	ycord=[]
	ep_ret=0

	for i in range(maxsteps):
	    # env.render()
	    action, _states = model.predict(state, deterministic=True)
	    next_state, reward, done, info = env.step(action)
	    tau.append(env.tau)
	    th = (next_state[0] + np.pi)%(2*np.pi) - np.pi
	    angle.append(th*180/np.pi)
	    ycord.append(next_state[1])
	    ep_ret += reward
	    state=next_state
	    if done:
	        print(f"Angle:{theta*180/np.pi:.2f}, Reward:{ep_ret:.2f}")
	        break

	x= np.array([i for i in range(1,len(angle)+1)])

	step = 200
	plt.figure(2)
	plt.plot(x[:step],tau[:step])
	plt.title('Force vs timesteps before convergence')
	plt.ylabel('Force (in N)')
	plt.xlabel('Timestep')
	plt.savefig(f'{name}_2a.png')

	plt.figure(3)
	plt.plot(x[:step], angle[:step])
	plt.title('Angle vs timesteps before convergence')
	plt.ylabel('Angle (in degrees)')
	plt.xlabel('Timestep')
	plt.savefig(f'{name}_2b.png')

	plt.figure(4)

	plt.plot(x[step:],tau[step:])
	plt.title('Force vs timesteps after convergence')
	plt.ylabel('Force (in N)')
	plt.xlabel('Timestep')
	plt.savefig(f'{name}_3a.png')

	plt.figure(5)
	plt.plot(x[step:],angle[step:])
	plt.title('Angle vs timesteps after convergence')
	plt.ylabel('Angle (in degrees)')
	plt.xlabel('Timestep')
	plt.savefig(f'{name}_3b.png')

	plt.figure(6)
	plt.plot(x,ycord)
	plt.title('Y coordinate vs timesteps')
	plt.ylabel('Y- coordinate of the cart (in m)')
	plt.xlabel('Timestep')
	plt.savefig(f'{name}_4.png')


fscale = 100
ivp_dt=0.02
maxsteps = 5000

#Uncomment below for DQN1
# discrete=True
# env = IPVC(fscale=fscale, maxsteps=maxsteps, discrete=discrete, ivp_dt=ivp_dt, test=True)
# name="DQN1"
# model = DQN.load('/home/vamsi/Desktop/VCP/dqn1_1270000_steps.zip')
# plot1('dqn1.csv')

#Uncomment below for DQN3
# discrete=True
# env = IPVC(fscale=fscale, maxsteps=maxsteps, discrete=discrete, ivp_dt=ivp_dt, test=True)
# name="DQN3"
# model = DQN.load('/home/vamsi/Desktop/VCP/dqn3_1160000_steps.zip')
# plot1('dqn3.csv')

#Uncomment below for DQN3
discrete=True
env = IPVC(fscale=fscale, maxsteps=maxsteps, discrete=discrete, ivp_dt=ivp_dt, test=True)
name="DQN3"
model = DQN.load('/home/vamsi/Desktop/VCP/dqn4_790000_steps.zip')
plot1('dqn4.csv')

#Uncomment below for SAC
# discrete = False
# env = IPVC(fscale=fscale, maxsteps=maxsteps, discrete=discrete, ivp_dt=ivp_dt, test=True)
# name = "SAC"
# model = SAC.load('/home/vamsi/Desktop/VCP/sac1_1420000_steps.zip')
# plot1('sac1.csv')

plot23()

# plt.show()



