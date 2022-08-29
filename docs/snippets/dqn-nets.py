from playground.agents import DeepQNetworkAgent

online_net = DeepQNetworkAgent(env, net)
target_net = DeepQNetworkAgent(env, net)

target_net.load_state_dict(online_net.state_dict())