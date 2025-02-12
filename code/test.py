import gymnasium as gym
env = gym.make('HalfCheetah-v4')
print(env)
observation,info =env.reset()
action =env.action_space.sample()
observation,reward, terminated,truncated, info= env.step(action)
observation,info =env.reset()
env.close()