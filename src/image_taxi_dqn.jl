using Flux
using BSON
# using NeuralVerification
using Plots

# Load in generative model
model = BSON.load("models/supervised_mlp.bson")[:model]
# Plot test image
test = (model([0.0, 0.0, 0.0, 0.0]) .+ 1) ./ 2
test_im = reshape(test, 16, 8)
plot(Gray.(test_im'))

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

function sim_episode(pomdp, policy; n_steps=50)
    cs = zeros(n_steps+1)
    ds = zeros(n_steps+1)
    hs = zeros(n_steps+1)
    rs = zeros(n_steps+1)

    s₀ = rand.(pomdp.s₀dists)
    cs[1] = s₀[1]
    ds[1] = 0.0
    hs[1] = s₀[2]
    rs[1] = pomdp.reward([s₀[1], s₀[2]]) 

    for step in 1:n_steps
        o = pomdp.obs([cs[step], hs[step]])
        ϕ = pomdp.actions[argmax(policy(o))]
        x′, y′, θ′ = next_position(cs[step], ds[step], hs[step], ϕ)
        r = pomdp.reward([x′, θ′])
        cs[step+1] = x′
        ds[step+1] = y′
        hs[step+1] = θ′
        rs[step+1] = r
    end
    return cs, ds, hs, rs
end

# Define POMDP
function taxi_pomdp(n_actions; λₚ=-1.0, λₕ=-1.0)
    s₀dists = Uniform.([-10.0, -1.0], [10.0, 1.0]) #Two uniform distros. One between -10/10 and the other between -1/1
    #Captures 2-d state space
    actions = collect(range(-5.0, stop=5.0, length=n_actions))
    reward(s) = λₚ * abs(s[1]) + λₕ * abs(s[2])
    obs(s) = s
    function gen(s, a) 
        x′, _, θ′ = next_position(s[1], 0.0, s[2], a)
        r = reward([x′, θ′])
        return [x′, θ′], obs([x′, θ′]), r
    end
    a2ind = Dict()
    for (i, a) in enumerate(actions)
        a2ind[a] = i
        a2ind[Float32.(a)] = i
    end
    return POMDP(s₀dists, actions, reward, obs, gen, a2ind)
end

function image_taxi_pomdp(n_actions; λₚ=-1.0, λₕ=-1.0)
    s₀dists = Uniform.([-10.0, -1.0], [10.0, 1.0]) #Two uniform distros. One between -10/10 and the other between -1/1
    #Captures 2-d state space
    actions = collect(range(-5.0, stop=5.0, length=n_actions))
    reward(s) = λₚ * abs(s[1]) + λₕ * abs(s[2])
    function obs(s)
        z = Float32.(rand(Uniform(-0.8, 0.8), 2))
        x = [z; s[1] / 6.366468343804353; s[2] / 17.248858791583547]
        o = model(x)
        return o
    end
    function gen(s, a) 
        x′, _, θ′ = next_position(s[1], 0.0, s[2], a)
        r = reward([x′, θ′])
        return [x′, θ′], obs([x′, θ′]), r
    end
    a2ind = Dict()
    for (i, a) in enumerate(actions)
        a2ind[a] = i
        a2ind[Float32.(a)] = i
    end
    return POMDP(s₀dists, actions, reward, obs, gen, a2ind)
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

function eval(pomdp, policy, ep_num, save_folder; n_eps=50, n_steps=50, plt_every=20)
    crosstracks = []
    downtracks = []
    headings = []
    rewards = []
    for episode in 1:n_eps
        cs, ds, hs, rs = sim_episode(pomdp, policy, n_steps=n_steps)
        push!(crosstracks, cs)
        push!(downtracks, ds)
        push!(headings, hs)
        push!(rewards, sum(rs))
    end
    r_ave = mean(rewards)
    r_std = std(rewards)

    if ep_num % plt_every == 0
        plot_results(crosstracks, downtracks, ep_num, save_folder)
    end

    return r_ave, r_std
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

function image_taxi_dqn(hidden_sizes, n_actions)
    policy = build_model([128; hidden_sizes; n_actions], relu)
    target = deepcopy(policy)
    replay_buffer = Vector{NamedTuple}()
    return DQN(policy, target, replay_buffer)
end

n_actions = 5
image_pomdp = image_taxi_pomdp(n_actions, λₚ=-10.0)
image_dqn = image_taxi_dqn([16, 8], n_actions)

pomdp = taxi_pomdp(n_actions, λₚ=-10.0)
dqn = taxi_dqn([16, 8], n_actions)

image_h = Hyperparameters(buffer_size=5000, save_folder="src/image_results/", batch_size=64, 
    n_grad_steps=20, ϵ=0.3, n_eps=300, learning_rate=1e-3)
h = Hyperparameters(buffer_size=5000, save_folder="src/results/", batch_size=64, 
    n_grad_steps=20, ϵ=0.3, n_eps=300, learning_rate=1e-3)

r_average_image, r_std_image = train(image_dqn, image_pomdp, image_h, eval)
r_average, r_std = train(dqn, pomdp, h, eval)

p = plot(collect(100:300), r_average[100:300], 
    fillrange=(r_average[100:300]-r_std[100:300],r_average[100:300]+r_std[100:300]), 
    fillalpha=0.35, c=1, label=std, legend=false, title="Reward vs Episodes", xlabel="# Episodes", ylabel="Reward")
plot!(p, collect(100:300), r_average_image[100:300], 
    fillrange=(r_average_image[100:300]-r_std_image[100:300],r_average_image[100:300]+r_std_image[100:300]), 
    fillalpha=0.35, c=2, label=std, legend=false, title="Reward vs Episodes", xlabel="# Episodes", ylabel="Reward")

image_dqn.policy(model([0.0, 0.0, 0.0 / 6.366468343804353, 0.0 / 17.248858791583547]))