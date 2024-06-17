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

mutable struct LQREnv <: AbstractEnv
    s::SVector{5, Int}
end

function CommonRLInterface.reset!(m::LQREnv)
    m.s = SA[0, 0, 0, 0, 0]
end

CommonRLInterface.actions(m::LQREnv) = (:NS, :EW)
CommonRLInterface.observe(m::LQREnv) = m.s
CommonRLInterface.terminated(m::LQREnv) = false

function CommonRLInterface.act!(m::LQREnv, a)
    
    # Volumetric Flowrates
    NS_ϵ = Poisson(2)
    EW_ϵ = Poisson(1)

    if a == :NS #NS Green (EW Red)
        r = -sum(m.s[1:4])
        
        # If we're in a NS green, then flow thru
        if m.s[5] == 0
            # Perform the potential flow thru
            deltaN, deltaS = -5+rand(NS_ϵ), -5+rand(NS_ϵ)
            deltaE, deltaW = rand(EW_ϵ), rand(EW_ϵ)
            
            # If we go negative, set to zero
            if (m.s[1]+deltaN) < 0
                deltaN = -m.s[1]
            end
            if (m.s[2]+deltaS) < 0
                deltaS = -m.s[2]
            end
            
            # Delta TL -> 0 (no change to light)
            light_delta = 0
        end

        # If we're in a EW green, then all red 
        # and transition the light to NS
        if m.s[5] == 1
            deltaN, deltaS = 2*rand(NS_ϵ), 2*rand(NS_ϵ)
            deltaE, deltaW = 2*rand(EW_ϵ), 2*rand(EW_ϵ)
            light_delta = -1
        end

        # Perform time step
        m.s = m.s + [deltaN,deltaS,deltaE,deltaW,light_delta]
    
    elseif a == :EW #NS Red (EW Green)
        r = -sum(m.s[1:4])

        # If we're in a EW green, then flow thru
        if m.s[5] == 1
            # Perform the potential flow thru
            deltaE, deltaW = -3+rand(EW_ϵ), -3+rand(EW_ϵ)
            deltaN, deltaS = rand(NS_ϵ), rand(NS_ϵ)

            # If we go negative, set to zero
            if (m.s[3]+deltaE) < 0
                deltaE = -m.s[3]
            end
            if (m.s[4]+deltaW) < 0
                deltaW = -m.s[4]
            end
            
            # Delta TL -> 0 (no change to light)
            light_delta = 0
        end

        # If we're in a NS green, then all red 
        # and transition the light to EW
        if m.s[5] == 0
            deltaN, deltaS = 2*rand(NS_ϵ), 2*rand(NS_ϵ)
            deltaE, deltaW = 2*rand(EW_ϵ), 2*rand(EW_ϵ)
            light_delta = 1
        end

        m.s = m.s + [deltaN,deltaS,deltaE,deltaW,light_delta]

    end

    return r
end

# Optional functions can be added like this:
CommonRLInterface.clone(m::LQREnv) = LQREnv(m.s)

function run_steps()

# Useful functions!
# ---------------------- 
# observe(env) # current state
# actions(env) # action space
# terminated(env) # bool check if final state
# act!(env, :up) # returns reward of the env
# HW4.render(env) # plots gridworld
# observations(env) # all states in env
# ---------------------- 
env = convert(AbstractEnv, LQREnv([5, 5, 2, 2, 0]))
print("\nBegin!\n")
# Let's run a 1000-time step policy!
# Consider 20 sec as the cycle
# The estimates are 2/1 so give 13 sec to NS, 7 sec to EW
states = [observe(env)]
NS_flows = Float64[]
EW_flows = Float64[]

t = 0
tcrit = 10
for _ in 1:1000
    t += 1
    
    if t <= tcrit
        act!(env, :NS)
    else
        act!(env, :EW)
    end
    s = observe(env)
    push!(states, observe(env))
    push!(NS_flows, (s[1]+s[2]))
    push!(EW_flows, (s[3]+s[4]))

    # Reset the loop
    if t >= 18
        t = 0
    end

end
@show observe(env)
print("End")

# Plot the flow diagrams
p = plot(title="4-Way Intersection Queue Lengths", xlim=(0, 100), ylim=(0, 100))
plot!(p, 1:1000, NS_flows, label = "Total NS Lengths")
plot!(p, 1:1000, EW_flows, label = "Total EW Lengths")
plot!(p, xlabel = "Time Steps", ylabel = "Total Queue Lengths, Vehicle Counts")
savefig(p, "Conventional_flow_diagram.png")

p2 = plot(title="4-Way Intersection Queue Lengths", xlim=(900, 1000))
plot!(p2, 1:1000, NS_flows, label = "Total NS Lengths")
plot!(p2, 1:1000, EW_flows, label = "Total EW Lengths")
plot!(p2, xlabel = "Time Steps", ylabel = "Total Queue Lengths, Vehicle Counts")
savefig(p2, "Conventional_flow_diagram_t900_1000.png")

end

run_steps()

