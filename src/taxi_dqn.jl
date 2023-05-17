using Plots
using Statistics
using LinearAlgebra

include("dqn.jl")

# Define dynamics model
function dynamics(x, y, θ, ϕ; dt=0.05, v=5, L=5)
    """
        x: crosstrack
        y: downtrack
        θ: heading
        ϕ: steering angle
    """

    # ẋ = v * sind(θ)
    # ẏ = v * cosd(θ)
    # θ̇ = (v / L) * tand(ϕ)

    # x′ = x + ẋ * dt
    # y′ = y + ẏ * dt
    # θ′ = θ + rad2deg(θ̇) * dt

    θ̇ = (v / L) * tand(ϕ)
    θ′ = θ + rad2deg(θ̇) * dt

    ẋ = v * sind(θ′)
    ẏ = v * cosd(θ′)

    x′ = x + ẋ * dt
    y′ = y + ẏ * dt

    return x′, y′, θ′
end

function next_position(x, y, θ, ϕ; control_every=10, v=5, L=5)
    for _ in 1:control_every
        x, y, θ = dynamics(x, y, θ, ϕ, v=v, L=L)
    end
    return x, y, θ
end

function sim_episode(mdp, policy; n_steps=50)
    cs = zeros(n_steps+1)
    ds = zeros(n_steps+1)
    hs = zeros(n_steps+1)
    rs = zeros(n_steps+1)

    s₀ = rand.(mdp.s₀dists)
    cs[1] = s₀[1]
    ds[1] = 0.0
    hs[1] = s₀[2]
    rs[1] = mdp.reward([s₀[1], s₀[2]])

    for step in 1:n_steps
        ϕ = mdp.actions[argmax(policy([cs[step], hs[step]]))]
        x′, y′, θ′ = next_position(cs[step], ds[step], hs[step], ϕ)
        r = mdp.reward([x′, θ′])
        cs[step+1] = x′
        ds[step+1] = y′
        hs[step+1] = θ′
        rs[step+1] = r
    end
    return cs, ds, hs, rs
end

# Define MDP components
function taxi_mdp(n_actions; λₚ=-1.0, λₕ=-1.0)
    s₀dists = Uniform.([-10.0, -1.0], [10.0, 1.0])
    actions = collect(range(-10.0, stop=10.0, length=n_actions))
    reward(s) = λₚ * abs(s[1]) + λₕ * abs(s[2])
    function gen(s, a) 
        x′, y′, θ′ = next_position(s[1], s[2], 0.0, a)
        r = reward([x′, θ′])
        return [x′, θ′], r
    end
    a2ind = Dict()
    for (i, a) in enumerate(actions)
        a2ind[a] = i
    end
    return MDP(s₀dists, actions, reward, gen, a2ind)
end

# Create evaluation functions
rectangle(w, h, x, y) = Shape(x .+ [0, w, w, 0], y .+ [0, 0, h, h])
function plot_runway(; downtrack=150.0, green_width=2.0)
    p = plot(rectangle(downtrack, green_width, 0, -green_width-10), legend=false, color=:green,
    xlims=(0.0, downtrack), ylims=(-green_width-10, 10+green_width))
    plot!(p, rectangle(downtrack, green_width, 0, 10), legend=false, color=:green)
    plot!(p, rectangle(downtrack, 20, 0, -10), legend=false, color=:gray, opacity=0.5)
    plot!(p, [0, downtrack], [0, 0], linestyle=:dash, color=:black, size=(800, 300))
    return p
end

function plot_results(crosstracks, downtracks, episode_number, save_folder)
    p = plot_runway()
    for (cs, ds) in zip(crosstracks, downtracks)
        plot!(p, ds, cs, color=:blue)
    end
    savefig("$(save_folder)$(episode_number).png")
end

function eval(mdp, policy, ep_num, save_folder; n_eps=10, n_steps=50, plt_every=100)
    crosstracks = []
    downtracks = []
    headings = []
    rewards = []
    for episode in 1:n_eps
        cs, ds, hs, rs = sim_episode(mdp, policy, n_steps=n_steps)
        push!(crosstracks, cs)
        push!(downtracks, ds)
        push!(headings, hs)
        push!(rewards, mean(rs))
    end
    r_ave = mean(rewards)

    if ep_num % plt_every == 0
        plot_results(crosstracks, downtracks, ep_num, save_folder)
    end

    return r_ave
end

# Define networks
function build_model(layer_sizes, act)
    # ReLU except last layer identity
    layers = Any[Dense(layer_sizes[i], layer_sizes[i+1], act) for i = 1:length(layer_sizes)-2]
    push!(layers, Dense(layer_sizes[end-1], layer_sizes[end]))
    return Chain(layers...)
end

function taxi_dqn(hidden_sizes, n_actions)
    policy = build_model([2; hidden_sizes; n_actions], relu)
    target = deepcopy(policy)
    replay_buffer = Vector{NamedTuple}()
    return DQN(policy, target, replay_buffer)
end

# Set up to attempt training
n_actions = 5
mdp = taxi_mdp(n_actions, λₕ=-0.5)
dqn = taxi_dqn([10, 10], n_actions)

h = Hyperparameters(buffer_size=5000, save_folder="src/results/", batch_size=256, n_grad_steps=20)

train(dqn, mdp, h, eval)

nothing

# random_policy(s) = rand(5)
# left_policy(s) = [1.0, 0.0, 0.0, 0.0, 0.0]
# right_policy(s) = [0.0, 0.0, 0.0, 0.0, 1.0]
# straight_policy(s) = [0.0, 0.0, 1.0, 0.0, 0.0]
# eval(mdp, random_policy, 0, "src/", n_eps=1)