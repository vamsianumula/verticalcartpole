import numpy as np
from scipy.integrate import solve_ivp
import gym
from gym import spaces
import matplotlib.pyplot as plt
import time

class IPVC(gym.Env):
    metadata = {
        'render.modes': ['human','rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,fscale=100, maxsteps=1000, discrete=False, ivp_dt=0.02, test=False):
        super(IPVC,self).__init__()
        self.viewer = None
        self.counter = 0
        self.maxsteps = maxsteps
        self.mp = 0.25
        self.Jp = 0.0075
        self.lp = 0.6
        self.mc = 1
        self.g = 9.81
        self.discrete= discrete
        self.fscale = fscale
        
        self.dt= ivp_dt
        self.test=test
        self.state = np.array([0,0,0,0])
        if self.discrete:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(low=-1*np.ones((1,)),high=1*np.ones((1,)),dtype=np.float32)
        self.observation_space = spaces.Box(low=-100*np.ones((4,)),high=100*np.ones((4,)),dtype=np.float32)
    
    def reset(self,th_init=135*np.pi/180):
        self.counter = 0
        self.th_init= th_init
        self.state = np.array([np.pi+np.random.normal(0,1.8),0,np.random.normal(0,1),0])

        if self.th_init is not None:
            self.state[0] = self.th_init
                    
        if self.test == True:
            self.state[2]=0

        self.state[0] = self.state[0]%(2*np.pi)
        obs = self.state
        return obs

    def __f(self,t,y):
        mp, Jp, lp, mc, g = self.mp, self.Jp, self.lp, self.mc, self.g
        q, yc, qd, ycd = y[0:4]
        M = np.array([[Jp + lp ** 2 * mp / 4,-mp * lp * np.sin(q) / 2],[-mp * lp * np.sin(q) / 2,mc + mp]])
        C = np.array([[0],[-mp * lp * qd ** 2 * np.cos(q) / 2]])
        G = np.array([[-mp * g * lp * np.sin(q) / 2],[mc * g + mp * g]])
        tau = self.tau.reshape((2,1))+np.array([0,(mp+mc)*g]).reshape((2,1))
        Minv = np.linalg.inv(M)
        acc = Minv@(tau-C-G)
        dy = np.array([y[2],y[3],acc[0],acc[1]],dtype=np.float32)
        return dy

    def step(self,action):
        next_state, reward, done, info = self.state, 0, False, {'Terminal':''}
        self.counter += 1
        if self.counter>self.maxsteps:
            done, info['Terminal'] = True, 'Timeout'
            return next_state, reward, done, info
        if self.discrete:
            if action==1:
                self.tau = np.array([0,self.fscale])
            else:
                self.tau = np.array([0,-self.fscale])
        else:
            self.tau = np.array([0,self.fscale*action[0]])

        self.state[0] = self.state[0]%(2*np.pi)

        sol = solve_ivp(self.__f,[0,self.dt],self.state,rtol=1e-8,atol=1e-8)

        next_state = sol.y[:,-1]
        next_state[0] = next_state[0]%(2*np.pi)
        self.state = next_state
        
        reward = 0.1*(next_state[0]-np.pi)**2-0.005*abs(next_state[0]-np.pi)*next_state[2]**2-0.05*abs(next_state[1])

        if abs(next_state[1])>1.4:# 0.2
            reward, done, info['Terminal'] = -100, True, 'Limit'
        if self.counter==self.maxsteps:
            done, info['Terminal'] = True, 'Timeout'
        return next_state, reward, done, info

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(300,600)
            self.viewer.set_bounds(-1.0,1.0,-2.0,2.0)#left,right,bottom,top

            gnd = rendering.Line((0,-1.45),(0,1.45))
            gnd.set_color(0.5,0.5,0.5)
            upstopper = rendering.Line((-0.2,1.45),(0.2,1.45))
            upstopper.set_color(0.2,0.2,0.2)
            dnstopper = rendering.Line((-0.2,-1.45),(0.2,-1.45))
            dnstopper.set_color(0.2,0.2,0.2)

            cart = rendering.make_polygon([(-0.1,-0.1),(0.1,-0.1),(0.1,0.1),(-0.1,0.1)],filled=True)
            cart.set_color(1,0.5,0.5)
            self.cartt = rendering.Transform(translation=(0,0))
            cart.add_attr(self.cartt)

            pend = rendering.make_capsule(0.6,0.02)
            pend.set_color(0.5,0.5,1.0)
            self.pendt = rendering.Transform(rotation=self.state[0]+np.pi/2)
            pend.add_attr(self.pendt)

            self.viewer.add_geom(gnd)
            self.viewer.add_geom(upstopper)
            self.viewer.add_geom(dnstopper)
            self.viewer.add_geom(cart)
            self.viewer.add_geom(pend)
        self.cartt.set_translation(0,self.state[1])
        self.pendt.set_translation(0,self.state[1])
        self.pendt.set_rotation(self.state[0]+np.pi/2)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None