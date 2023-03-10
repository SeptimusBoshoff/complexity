using DrWatson
@quickactivate "dare"

include(srcdir("nodeconstructor.jl"))
include(srcdir("env.jl"))
include(srcdir("agent_ddpg.jl"))
include(srcdir("data_hook.jl"))


function reward(env)
    #implement your reward function here

    P_required = 466 # W
    V_required = 230 # V

    u_l1_index = findfirst(x -> x == "u_load1", env.state_ids)

    u_l1 = env.state[u_l1_index]

    P_load = (env.norm_array[u_l1_index] * u_l1)^2 / 14
    
    # P_diff = -abs(P_required - P_load) 
    # reward = exp(P_diff/130) - 1

    reward = -(abs(V_required - (env.norm_array[u_l1_index] * u_l1))/300)

    return reward
end

env_cuda = false
agent_cuda = false

CM = [ 0. 0. 1.
        0. 0. 2
        -1. -2. 0.]

parameters = Dict{Any, Any}(
    "source" => Any[
                    Dict{Any, Any}("L1"=>0.0023, "R_C"=>0.4, "C"=>1.0e-5, "R1"=>0.4, "fltr"=>"LC"),
                    Dict{Any, Any}("v_rip"=>0.015, "L1"=>0.0023, "vdc"=>700, "R1"=>0.4, "i_rip"=>0.12, "pwr"=>10000.0, "fltr"=>"L")
                    ],
    "load"   => Any[
                    Dict{Any, Any}("R"=>14, "impedance"=>"R")
                    #Dict{Any, Any}("C"=>0.381, "L"=>0.2, "R"=>14, "impedance"=>"RLC")
                    ],
    "cable"  => Any[
                    Dict{Any, Any}("C"=>4.0e-7, "L"=>0.000264, "R"=>0.722),
                    Dict{Any, Any}("C"=>4.0e-7, "L"=>0.000264, "R"=>0.722)
                    ],
    "grid"   => Dict{Any, Any}("fs"=>10000.0, "phase"=>1, "v_rms"=>230)
)


env = SimEnv(num_sources = 2, num_loads = 1, CM = CM, parameters = parameters, reward_function = reward, maxsteps=600, use_gpu=env_cuda)

ns = length(env.sys_d.A[1,:])
na = length(env.sys_d.B[1,:])

agent = create_agent_ddpg(na = na, ns = ns, use_gpu = env_cuda)

hook = data_hook(save_best_NNA = true, plot_rewards = true)

run(agent, env, StopAfterEpisode(80), hook)





# PLOT rewards in 3D plot over every episode

plot_rewards_3d(hook)



# PLOT a test run with the best behavior_actor NNA so far

plot_best_results(;agent = agent, env = env, hook = hook, state_ids_to_plot = ["u_f1", "u_1", "u_2", "u_load1"])

