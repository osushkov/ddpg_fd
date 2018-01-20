
def _run_episode(env, agent, max_steps_per_episode, observers, episode):
    obs = env.reset()

    cur_step = 0
    while True:
        action = agent.act(obs)
        new_obs, reward, done, _ = env.step(action)

        agent.feedback(new_obs, reward, done)

        for observer in observers:
            observer(env, agent, episode, cur_step, obs, action, reward)

        obs = new_obs
        cur_step += 1

        if done:
            break

        if max_steps_per_episode is not None and cur_step >= max_steps_per_episode:
            break


def run_loop(env, agent, num_episodes, max_steps_per_episode=None, observers=[]):
    for episode in range(num_episodes):
        agent.initialize_episode(episode)
        _run_episode(env, agent, max_steps_per_episode, observers, episode)
