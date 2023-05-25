using Distributions
using Random
using Flux
using Flux: update!
using ProgressBars
using Parameters
using Plots

struct MDP
    s₀dists::Vector{Distribution}
    actions::Vector
    reward::Base.Callable
    gen::Base.Callable
    a2ind::Dict
end

struct POMDP
    s₀dists::Vector{Distribution} #State space
    actions::Vector #Action Space
    reward::Base.Callable #Reward Function 
    obs::Base.Callable #Observation Space
    gen::Base.Callable #Transtion Function 
    a2ind::Dict
end


mutable struct DQN
    policy
    target
    replay_buffer::Vector{NamedTuple}
end

@with_kw struct Hyperparameters
    buffer_size::Int64 = 5000 # Size of replay_buffer
    n_eps::Int64 = 5000 # Total number of episodes
    n_steps::Int64 = 50 # Number of steps per episode
    n_grad_steps::Int64 = 5 # Number of gradient steps per update
    update_target_every::Int64 = 100 # How many steps between target updates
    batch_size::Int64 = 64 # Number of samples from the replay buffer
    ϵ::Float64 = 0.3 # ϵ for ϵ-greedy exploration
    learning_rate::Float64 = 1e-3 # learning rate for ADAM optimizer
    save_folder::String = "results/"
end

function to_buffer!(s, o, a, r, s′, o′, replay_buffer, buffer_size)
    """
    Adds experience tuple to replay buffer while maintaining correct buffer size
    """
    length(replay_buffer) == buffer_size ? popfirst!(replay_buffer) : nothing
    push!(replay_buffer, (s=Float32.(s), o=Float32.(o), a=Float32(a), r=Float32(r), s′=Float32.(s′), o′=Float32.(o′)))
end

function sample_batch(replay_buffer, target, a2ind, batch_size)
    """
    Samples and creates training batch
    """
    # Sample experience tuples
    experiences = replay_buffer[randperm(length(replay_buffer))[1:batch_size]]
    # Create S
    O = hcat([e.o for e in experiences]...)
    A = hcat([a2ind[e.a] for e in experiences]...)
    # Create y
    O′ = hcat([e.o′ for e in experiences]...)
    r = hcat([e.r for e in experiences]...)
    y = r + maximum(target(O′), dims=1)
    # Return data
    return O, A, y
end

function dqn_loss(policy, O, A, y)
    Q = policy(O)
    ŷ = [Q[a, i] for (i, a) in enumerate(A)]
    return Flux.Losses.mse(ŷ, y)
end

function train(dqn::DQN, pomdp::POMDP, h::Hyperparameters, eval)
    # Gather parameters and initialize optimizer
    θ = Flux.params(dqn.policy)
    opt = ADAM(h.learning_rate)

    # Initialize some counters and storage
    target_step_counter = 0
    episodes = ProgressBar(1:h.n_eps)
    r_average = zeros(h.n_eps)
    r_stdev = zeros(h.n_eps)

    for episode in episodes
        # Reset to sampled initial state
        s = Float32.(rand.(pomdp.s₀dists))
        o = pomdp.obs(s)
        for step in 1:h.n_steps
            if length(dqn.replay_buffer) < h.buffer_size
                # Exploration phase
                a = rand(pomdp.actions)
            else
                # Sample an action with ϵ-greedy exploration
                a = rand() < h.ϵ ? rand(pomdp.actions) : pomdp.actions[argmax(dqn.policy(o))]
            end
            # Simulation step
            s′, o′, r = pomdp.gen(s, a)
            # Add experience tuple to replay buffer
            to_buffer!(s, o, a, r, s′, o′, dqn.replay_buffer, h.buffer_size)
            # Update current state
            s = s′
            o = o′
            # If enough data is in the replay buffer, perform some training steps
            if length(dqn.replay_buffer) == h.buffer_size
                # Get training batch
                O, A, y = sample_batch(dqn.replay_buffer, dqn.target, pomdp.a2ind, h.batch_size)
                # Train
                for _ = 1:h.n_grad_steps
                    loss, back = Flux.pullback(() -> dqn_loss(dqn.policy, O, A, y), θ)
                    update!(opt, θ, back(1.0f0))
                    # println(loss)
                end
                # println()
                # Update target if time
                target_step_counter += 1
                if target_step_counter == h.update_target_every
                    dqn.target = deepcopy(dqn.policy)
                    target_step_counter = 0
                end
            end
        end
        r_ave, r_std = eval(pomdp, dqn.policy, episode, h.save_folder)
        set_postfix(episodes, R="$r_ave", StDev="$r_std")
        r_average[episode] = r_ave
        r_stdev[episode] = r_std
    end
    return r_average, r_stdev
end



