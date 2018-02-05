from torch.autograd import Variable
from madrl_environments.pursuit import MAWaterWorld_mod
from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
from params import scale_reward


# do not render the scene
e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = MAWaterWorld_mod(n_pursuers=2, n_evaders=50,
                         n_poison=50, obstacle_radius=0.04,
                         food_reward=food_reward,
                         poison_reward=poison_reward,
                         encounter_reward=encounter_reward,
                         n_coop=n_coop,
                         sensor_range=0.2, obstacle_loc=None, )

vis = visdom.Visdom(port=8097)  # 5274
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.seed(1234)
n_agents = world.n_pursuers
n_states = 213
n_actions = 2
capacity = 1000000
batch_size = 1000

n_episode = 20000
max_steps = 1000
episodes_before_train = 100    # 100

win = None
param = None
grad_critics1 = None
grad_critics2 = None
grad_actors1 = None
grad_actors2 = None

snapshot_path = "/home/jadeng/dev/snapshot/"
snapshot_name = "maddpg_latest_episode_"
path = snapshot_path + snapshot_name + '800'

# maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train, load_models=path)

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train, load_models=None)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

for i_episode in range(n_episode):
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    av_critics_grad = np.zeros((2, 8))   # according to critic network
    av_actors_grad = np.zeros((2, 6))    # according to actor network
    n = 0
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        #if i_episode % 100 == 0 and e_render:
         #   world.render()
        world.render()
        obs = Variable(obs).type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = world.step(action.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        # c_loss, a_loss = maddpg.update_policy()
        # print(a_loss)     # NoneType
        # print('length of a_loss: ', len(a_loss))
        # print('length of c_loss: ', len(c_loss))

        critics_grad, actors_grad = maddpg.update_policy()

        if maddpg.episode_done > maddpg.episodes_before_train:
            # print(critics_grad)
            # print(actors_grad)
            av_critics_grad += np.array(critics_grad)
            av_actors_grad += np.array(actors_grad)
            n += 1

    if n != 0:
        av_critics_grad = av_critics_grad / n
        av_actors_grad = av_actors_grad / n

    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)
    # print('Average critics grad: ', av_critics_grad)
    # print('Average actors grad: ', av_actors_grad)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on WaterWorld\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % n_agents +
              ', coop=%d' % n_coop +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
              'food=%f, poison=%f, encounter=%f' % (
                  food_reward,
                  poison_reward,
                  encounter_reward))

    # plot of reward
    if win is None:
        win = vis.line(X=np.arange(i_episode, i_episode+1),
                       Y=np.array([np.append(total_reward, rr)]),
                       opts=dict(
                           ylabel='Reward',
                           xlabel='Episode',
                           title='MADDPG on WaterWorld_mod\n' +
                           'agent=%d' % n_agents +
                           ', coop=%d' % n_coop +
                           ', sensor_range=0.2\n' +
                           'food=%f, poison=%f, encounter=%f' % (
                               food_reward,
                               poison_reward,
                               encounter_reward),
                           legend=['Total'] +
                           ['Agent-%d' % i for i in range(n_agents)]))
    else:
        vis.line(X=np.array([np.array(i_episode).repeat(n_agents+1)]),
                 Y=np.array([np.append(total_reward, rr)]),
                 win=win,
                 update='append')

    # plot of exploration rate
    if param is None:
        param = vis.line(X=np.arange(i_episode, i_episode+1),
                         Y=np.array([maddpg.var[0]]),
                         opts=dict(
                             ylabel='Var',
                             xlabel='Episode',
                             title='MADDPG on WaterWorld: Exploration',
                             legend=['Variance']))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([maddpg.var[0]]),
                 win=param,
                 update='append')

    # plot of agent1 gradient of critics net
    if grad_critics1 is None:
        grad_critics1 = vis.line(X=np.arange(i_episode, i_episode+1),
                        Y=np.array([av_critics_grad[0]]),
                        opts=dict(
                            ylabel='Average Gradient',
                            xlabel='Episode',
                            title='Norm of Gradient for critics network of agent 1',
                            legend=['Grad-%d' % i for i in range(8)]))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([av_critics_grad[0]]),
                 win=grad_critics1,
                 update='append')

    # plot of agent2 gradient of critics net
    if grad_critics2 is None:
        grad_critics2 = vis.line(X=np.arange(i_episode, i_episode + 1),
                        Y=np.array([av_critics_grad[1]]),
                        opts=dict(
                            ylabel='Average Gradient',
                            xlabel='Episode',
                            title='Norm of Gradient for critics network of agent 2',
                            legend=['Grad-%d' % i for i in range(8)]))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([av_critics_grad[1]]),
                 win=grad_critics2,
                 update='append')

    # plot of agent1 gradient of actors net
    if grad_actors1 is None:
        grad_actors1 = vis.line(X=np.arange(i_episode, i_episode + 1),
                        Y=np.array([av_actors_grad[0]]),
                        opts=dict(
                            ylabel='Average Gradient',
                            xlabel='Episode',
                            title='Norm of Gradient for actors network of agent 1',
                            legend=['Grad-%d' % i for i in range(6)]))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([av_critics_grad[0]]),
                 win=grad_actors1,
                 update='append')

    # plot of agent2 gradient of actors net
    if grad_actors2 is None:
        grad_actors2 = vis.line(X=np.arange(i_episode, i_episode + 1),
                        Y=np.array([av_actors_grad[1]]),
                        opts=dict(
                            ylabel='Average Gradient',
                            xlabel='Episode',
                            title='Norm of Gradient for actors network of agent 2',
                            legend=['Grad-%d' % i for i in range(6)]))
    else:
        vis.line(X=np.array([i_episode]),
                 Y=np.array([av_critics_grad[1]]),
                 win=grad_actors2,
                 update='append')

    # to save models every 200 episodes
    if i_episode != 0 and i_episode % 200 == 0:
        print('Save models!')
        states = {'models': maddpg.models,
                  'critic_optimizer': maddpg.critic_optimizer,
                  'actor_optimizer': maddpg.actor_optimizer,
                  'critics_target': maddpg.critics_target,
                  'actors_target': maddpg.actors_target,
                  'memory': maddpg.memory,
                  'var': maddpg.var}
        th.save(states, snapshot_path + snapshot_name + str(i_episode))

world.close()





