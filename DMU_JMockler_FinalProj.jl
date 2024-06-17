using Printf
using LinearAlgebra: I, norm
import POMDPTools
using SparseArrays
using Statistics: mean
using BenchmarkTools
using Plots
using StatsBase
using CommonRLInterface
using StaticArrays
using Compose
using Distributions
import ColorSchemes

# Create MDP Model

# Add a github comment

# Make another comment

mutable struct Traffic_model <: AbstractEnv
    s::SVector{3, Int}
end

function CommonRLInterface.reset!(m::Traffic_model)
    m.s = SA[2, 2, 0]
end

# Additional commands
    CommonRLInterface.actions(m::Traffic_model) = (:NS, :EW)
    CommonRLInterface.observe(m::Traffic_model) = m.s
    CommonRLInterface.terminated(m::Traffic_model) = false
    CommonRLInterface.observations(env::Traffic_model) = [SA[x, y, i] for x in 0:50, y in 0:50, i in 0:1]

function CommonRLInterface.act!(m::Traffic_model, a)
    
    # Volumetric Flowrates
    NS_ϵ = Poisson(2)
    EW_ϵ = Poisson(1)

    if a == :NS #NS Green (EW Red)
        r = -sum(m.s[1:2])
        if m.s[1] == 50 || m.s[2] == 50
            r = -1000
        end
        
        # If we're in a NS green, then flow thru
        if m.s[3] == 0
            # Perform the potential flow thru
            deltaN, deltaS = -5+rand(NS_ϵ), -5+rand(NS_ϵ)
            deltaE, deltaW = rand(EW_ϵ), rand(EW_ϵ)
            deltaNS = deltaN + deltaS
            deltaEW = deltaE + deltaW
            
            # If we go negative, set to zero
            if (m.s[1]+deltaNS) < 0
                deltaNS = -m.s[1]
            end

            # If we go above 50, set to 50
            if (m.s[1]+deltaNS) > 50
                deltaNS = 0
            end
            if (m.s[2]+deltaEW) > 50
                deltaEW = 0
            end
            
            # Delta TL -> 0 (no change to light)
            light_delta = 0
        end

        # If we're in a EW green, then all red 
        # and transition the light to NS
        if m.s[3] == 1
            deltaN, deltaS = 1*rand(NS_ϵ), 1*rand(NS_ϵ)
            deltaE, deltaW = 1*rand(EW_ϵ), 1*rand(EW_ϵ)
            deltaNS, deltaEW = deltaN + deltaS, deltaE + deltaW

            if (m.s[1]+deltaNS) > 50
                deltaNS = 0
            end
            if (m.s[2]+deltaEW) > 50
                deltaEW = 0
            end

            light_delta = -1
        end

        # Perform time step
        m.s = m.s + [deltaNS,deltaEW,light_delta]
    
    elseif a == :EW #NS Red (EW Green)
        r = -sum(m.s[1:2])
        if m.s[1] == 50 || m.s[2] == 50
            r = -1000
        end

        # If we're in a EW green, then flow thru
        if m.s[3] == 1
            # Perform the potential flow thru
            deltaE, deltaW = -3+rand(EW_ϵ), -3+rand(EW_ϵ)
            deltaN, deltaS = rand(NS_ϵ), rand(NS_ϵ)
            deltaNS = deltaN + deltaS
            deltaEW = deltaE + deltaW

            # If we go negative, set to zero
            if (m.s[2]+deltaEW) < 0
                deltaEW = -m.s[2]
            end

            # If we hit 50, ignore
            if (m.s[1]+deltaNS) > 50
                deltaNS = 0
            end
            if (m.s[2]+deltaEW) > 50
                deltaEW = 0
            end
            
            # Delta TL -> 0 (no change to light)
            light_delta = 0
        end

        # If we're in a NS green, then all red 
        # and transition the light to EW
        if m.s[3] == 0
            deltaN, deltaS = 1*rand(NS_ϵ), 1*rand(NS_ϵ)
            deltaE, deltaW = 1*rand(EW_ϵ), 1*rand(EW_ϵ)
            deltaNS, deltaEW = deltaN + deltaS, deltaE + deltaW

            if (m.s[1]+deltaNS) > 50
                deltaNS = 0
            end
            if (m.s[2]+deltaEW) > 50
                deltaEW = 0
            end

            light_delta = 1
        end

        m.s = m.s + [deltaNS,deltaEW,light_delta]

    end

    return r

end

# Optional functions can be added like this:
CommonRLInterface.clone(m::Traffic_model) = Traffic_model(m.s)



# -------- Script Controls -------------
max_steps = 200
n_episodes_max = 300
heuristic_cycles = floor(0.2*n_episodes_max)
n_episodes_qlearn = n_episodes_max
n_eps_eval = 500
ϵM, ϵQ = 0.1, 0.2
γM, γQ = 0.99, 1
α=0.01
pp = 1
cycle_times = 500


# -------- Max-Likelihood -------------
function max_like_episode!(N, ρ, env, index, i; ϵ)
### Performs the RL algorithm inner-loop
### i.e. does a single episode

    # calculate MDP model
    A = collect(actions(env))
    sz = length(index) # number of states
    sums = Dict(a => sum(N[a], dims=2).+1e-6 for a in A)
    T = Dict(a => N[a]./sums[a] for a in A)
    R = Dict(a => ρ[a]./sums[a] for a in A)
    
    # solve with VI
    V = zeros(sz)
    oldV = ones(sz)
    
    while maximum(abs, V-oldV) > 0.001
        oldV[:] = V
        V[:] = max.((R[a] .+ γM*T[a]*V for a in A)...)
    end
    
    Q = Dict(a => R[a] .+ γM*T[a]*V for a in A)
    
    policy(s) = A[argmax([Q[a][index[s]] for a in A])]
    
    function greedy_eps_policy(env, ϵ)
        if rand() < ϵ
            return rand(actions(env))
        else
            return policy(observe(env))
        end
    end

    function heuristic(s)
        if s[1] < 2
            return :EW
        elseif s[2] < 2
            return :NS
        else
            if s[3] == 0
                return :NS
            else
                return :EW
            end
        end
    end

    # setup
    s = observe(env)

    if i == heuristic_cycles
        print("\nGreedy eps starts\n")
    end
    
    n = 0
    while n < max_steps
        if i < heuristic_cycles
            a = heuristic(s)
        else
            a = greedy_eps_policy(env, ϵ)
        end
        r = act!(env, a)
        sp = observe(env)
        N[a][index[s], index[sp]] += 1
        ρ[a][index[s]] += r
        s = sp
        n += 1
    end
    
    # Eval a run to plot Training
    # Pass back mean and std of 10 trials
    cnd = false
    @show (r_cycle, rs) = evaluateM(env, Q, cnd)

    return (Q, r_cycle, rs)
end

function max_like!(env)
### Performs the RL algorithm outer-loop

    # Do all preliminaries
    sz = length(observations(env))
    n = Dict(a => spzeros(sz,sz) for a in actions(env))
    ρ = Dict(a => spzeros(sz) for a in actions(env))
    Q = Dict((s, a) => 0.0 for s in observations(env), a in actions(env))
    index = Dict(s => i for (i, s) in enumerate(observations(env)))
    
    # Initialize to run the training
    episodes = []
    train_rewards = Float64[]
    train_rewards_std = Float64[]
    
    # Train the model! Save off the evaluations
    for i in 1:n_episodes_max
        reset!(env)
        (Q, r, rs) = max_like_episode!(n, ρ, env, index, i; ϵ=max(ϵM, 1-i/n_episodes_max))
        push!(episodes, Q)
        push!(train_rewards, r)
        push!(train_rewards_std, rs) #SEM
        print("Episode complete\n")
    end

    # Create and save off a plot (FIG3)
    training = plot(1:n_episodes_max, train_rewards,
        lw = 1.5, markershape = :circle, ms=2, ma=0.5,
        ribbon=train_rewards_std, fillalpha=.25, 
        legend = false)
    plot!(training, title = "Maximum Likelihood Model Training", xlabel = "Training Episode", ylabel = "Undiscounted Reward / Episode")
    savefig(training, "training_plot.png")

    # Express as a %'age
    train_percent = (train_rewards .- maximum(train_rewards))/abs(minimum(train_rewards))
    trb = train_rewards_std./abs.(train_rewards)
    train_percent = plot(1:n_episodes_max, train_percent, 
        lw = 2, markershape = :circle,
        ribbon=trb, fillalpha=.25, 
        legend = false)
    plot!(train_percent, title = "Maximum Likelihood Model Training", xlabel = "Training Episode", ylabel = "Normalized Undiscounted Reward / Episode")
    savefig(train_percent, "training_plot_percentage.png")

    # Return the Q's
    return episodes
end


function evaluateM(env, Q, final,  γ=1.0)
    A = actions(env)
    index = Dict(s => i for (i, s) in enumerate(observations(env)))
    policy(s) = A[argmax([Q[a][index[s]] for a in A])]
    r = Float64[]
    if final == true
        num, max_steps = 50, cycle_times
    else
        num, max_steps = 5, 100
    end
    for _ = 1:num #Run for 5 episodes to get an avg
        reset!(env)
        s = observe(env)
        rwd = 0.0
        t = 0
        while !terminated(env) && t < max_steps
            a = policy(s)
            rwd += act!(env, a)
            s = observe(env)
            t += 1
        end
        push!(r, rwd)
    end
    return (mean(r), std(r))
end

function rollout_heuristic(env)
rwds = Float64[]
for _ = 1:50
    t, tcrit, rcycle= 0, 10, 0.
    reset!(env)
    for _ in 1:cycle_times
        t += 1
        if t <= tcrit
            r = act!(env, :NS)
        else
            r = act!(env, :EW)
        end
        rcycle += r
        # Reset the loop
        if t >= 18
            t = 0
        end

    end
    push!(rwds, rcycle)
end

return (mean(rwds), std(rwds))

end


# -------- Script -------------
function RL_train_test()
### This script trains and tests an RL model of the 
### signal timing design of the intersection model

# Construct and train the model
env = convert(AbstractEnv, Traffic_model([5, 5, 0]))
print("Begin!\n")
Max_like_eps = max_like!(env)
print("End")

# Define the policies for using the Q constructs
A = actions(env)
index = Dict(s => i for (i, s) in enumerate(observations(env)))
# Policy is the argmax of Q(s, a)
function policy(Q, s)
    return A[argmax([Q[a][index[s]] for a in A])]
end

# Plot a few training flow curves of Q (FIG 4)
NS_flows1, EW_flows1 = Float64[], Float64[]
NS_flows2, EW_flows2 = Float64[], Float64[]
NS_flows3, EW_flows3 = Float64[], Float64[]
NS_flows4, EW_flows4 = Float64[], Float64[]

Qs = [Max_like_eps[floor(Int64, 0.25*n_episodes_max)], 
Max_like_eps[floor(Int64, 0.5*n_episodes_max)],
Max_like_eps[floor(Int64, 0.75*n_episodes_max)],
Max_like_eps[floor(Int64, 0.9*n_episodes_max)]]

# Iterate to make the training subplots
for i = 1:4
reset!(env)
Qiter = Qs[i]
    for _ in 1:150
        s = observe(env)
        a = policy(Qiter, s)
        r = act!(env, a)
        if i == 1
            push!(NS_flows1, (s[1]))
            push!(EW_flows1, (s[2]))
        elseif i == 2
            push!(NS_flows2, (s[1]))
            push!(EW_flows2, (s[2]))
        elseif i == 3
            push!(NS_flows3, (s[1]))
            push!(EW_flows3, (s[2]))
        elseif i == 4
            push!(NS_flows4, (s[1]))
            push!(EW_flows4, (s[2]))
        end
    end
end

# Construct and save the policy subplots
    p = plot(title="4-Way Intersection Queue Lengths", xlim=(50, 150), ylim=(0, 50))
    plot!(p, 1:150, NS_flows1, label = "Total NS Lengths")
    plot!(p, 1:150, EW_flows1, label = "Total EW Lengths")
    plot!(p, xlabel = "Time Steps", ylabel = "Total Queue Lengths, Vehicle Counts")
    savefig(p, "SubflowQ1.png")

    p = plot(title="4-Way Intersection Queue Lengths", xlim=(50, 150), ylim=(0, 50))
    plot!(p, 1:150, NS_flows2, label = "Total NS Lengths")
    plot!(p, 1:150, EW_flows2, label = "Total EW Lengths")
    plot!(p, xlabel = "Time Steps", ylabel = "Total Queue Lengths, Vehicle Counts")
    savefig(p, "SubflowQ2.png")

    p = plot(title="4-Way Intersection Queue Lengths", xlim=(50, 150), ylim=(0, 50))
    plot!(p, 1:150, NS_flows3, label = "Total NS Lengths")
    plot!(p, 1:150, EW_flows3, label = "Total EW Lengths")
    plot!(p, xlabel = "Time Steps", ylabel = "Total Queue Lengths, Vehicle Counts")
    savefig(p, "SubflowQ3.png")

    p = plot(title="4-Way Intersection Queue Lengths", xlim=(50, 150), ylim=(0, 50))
    plot!(p, 1:150, NS_flows4, label = "Total NS Lengths")
    plot!(p, 1:150, EW_flows4, label = "Total EW Lengths")
    plot!(p, xlabel = "Time Steps", ylabel = "Total Queue Lengths, Vehicle Counts")
    savefig(p, "SubflowQ4.png")

# Take the final Q value for policy evaluation
Qmax = Max_like_eps[end]

# Construct flow diagrams using the policy! (FIG 5)
states = [observe(env)]
NS_flows_Learn, EW_flows_Learn = Float64[], Float64[]
rcycle = 0.
reset!(env)
for _ in 1:cycle_times
    s = observe(env)
    @show a = policy(Qmax, s)
    r = act!(env, a)
    rcycle += r
    push!(states, observe(env))
    push!(NS_flows_Learn, (s[1]))
    push!(EW_flows_Learn, (s[2]))
end

@show observe(env)
@show rcycle

NS_flows_conv = Float64[]
EW_flows_conv = Float64[]
t, tcrit, rcycle= 0, 10, 0.
reset!(env)
for _ in 1:cycle_times
    t += 1
    
    if t <= tcrit
        r = act!(env, :NS)
        print("\n :NS")
    else
        r = act!(env, :EW)
        print("\n :EW")
    end
    rcycle += r
    s = observe(env)
    push!(states, observe(env))
    push!(NS_flows_conv , (s[1]))
    push!(EW_flows_conv , (s[2]))

    # Reset the loop
    if t >= 18
        t = 0
    end

end
@show observe(env)
@show rcycle
    
# Plot the flow diagrams (FIG 5)
p = plot(title="Learned vs Conventional Signals: North/Southbound", xlim=(50, 150))
plot!(p, 1:cycle_times, NS_flows_Learn, lw = 1.5, ls = :solid, color = 1, label = "Reinforcement Learned")
plot!(p, 1:cycle_times, NS_flows_conv, lw = 1, ls = :dash, color = 1, label = "Conventional Planning")
savefig(p, "LearnedvsConv_Flows_NS.png")

p = plot(title="Learned vs Conventional Signals: East/Westbound", xlim=(50, 150))
plot!(p, 1:cycle_times, EW_flows_Learn, lw = 1.5, ls = :solid, color = 2, label = "Reinforcement Learned")
plot!(p, 1:cycle_times, EW_flows_conv, lw = 1, ls = :dash, color = 2, label = "Conventional Planning")
savefig(p, "LearnedvsConv_Flows_EW.png")

# Estimate their efficicacy
@show (mean_conv, std_conv) = rollout_heuristic(env)
cnd = true
@show (mean_RL, std_RL) = evaluateM(env, Qmax, cnd)

end

RL_train_test()


