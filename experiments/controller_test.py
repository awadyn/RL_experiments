import itr_dvfs_controller_0 as ec

env = ec.EnergyCorridor({"df": df_test, "step_count": 0, "reset_count": 0, "success_count": 0})

env.training = False
env.testing = True

fig = go.Figure()
plot_dvfss = []
plot_itrs = []
plot_rewards = []
for (i, d) in zip(env.itrs, env.dvfss):
	if d < 10000:
		plot_dvfss.append(d)
		plot_itrs.append(i)
		plot_rewards.append(env.reward_space[(i,d)])
fig.add_trace(go.Scatter(x=plot_itrs, y=plot_dvfss, text=plot_rewards, mode='markers'))
fig.add_trace(go.Scatter(x=[list(env.goal_key)[0]], y=[list(env.goal_key)[1]], marker_size=20, marker_color = "yellow"))	
fig.update_layout(title="QPS: " + str(env.qps) + " - Gamma: " + str(algo.config['gamma']) + " - lr: " + str(algo.config['lr']))

for key in env.key_space:
	obs = env.reset(start_key = key)
	done = False
	for i in range(20):
		action = algo.compute_single_action(obs)
		obs, reward, done, info = env.step(action)

		if done:
			break

	trace_name = str(key[0]) + " , " + str(key[1])
	if key[1] < 10000:
		if not done:
			fig.add_trace(go.Scatter(x=env.itrs_visited, y=env.dvfss_visited, name=trace_name, marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous")))
			print()
			print("NOT DONE")
			print()
		else:
			fig.add_trace(go.Scatter(x=env.itrs_visited, y=env.dvfss_visited, name=trace_name, marker= dict(size=10,symbol= "arrow-bar-up", angleref="previous"), marker_color="black"))
	print("itrs_visited: ", env.itrs_visited)
	print("dvfss_visited: ", env.dvfss_visited)



fig.show()

