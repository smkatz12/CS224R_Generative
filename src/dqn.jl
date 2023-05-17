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

function to_buffer!(s, a, r, s′, replay_buffer, buffer_size)
    """
    Adds experience tuple to replay buffer while maintaining correct buffer size
    """
    length(replay_buffer) == buffer_size ? pop!(replay_buffer) : nothing
    push!(replay_buffer, (s=Float32.(s), a=Int32(a), r=Float32(r), s′=Float32.(s′)))
end

function sample_batch(replay_buffer, target, a2ind, batch_size)
    """
    Samples and creates training batch
    """
    # Sample experience tuples
    experiences = replay_buffer[randperm(length(replay_buffer))[1:batch_size]]
    # Create S
    S = hcat([e.s for e in experiences]...)
    A = hcat([a2ind[e.a] for e in experiences]...)
    # Create y
    S′ = hcat([e.s′ for e in experiences]...)
    r = hcat([e.r for e in experiences]...)
    y = r + maximum(target(S′), dims=1)
    # Return data
    return S, A, y
end

function dqn_loss(policy, S, A, y)
    Q = policy(S)
    ŷ = [Q[a, i] for (i, a) in enumerate(A)]
    return Flux.Losses.mse(ŷ, y)
end

function train(dqn::DQN, mdp::MDP, h::Hyperparameters, eval)
    # Gather parameters and initialize optimizer
    θ = Flux.params(dqn.policy)
    opt = ADAM(1e-3)

    # Initialize some counters and storage
    target_step_counter = 0
    episodes = ProgressBar(1:h.n_eps)
    r_average = zeros(h.n_eps)

    for episode in episodes
        # Reset to sampled initial state
        s = Float32.(rand.(mdp.s₀dists))
        for step in 1:h.n_steps
            # Sample an action with ϵ-greedy exploration
            a = rand() < h.ϵ ? rand(mdp.actions) : mdp.actions[argmax(dqn.policy(s))]
            # Simulation step
            s′, r = mdp.gen(s, a)
            # Add experience tuple to replay buffer
            to_buffer!(s, a, r, s′, dqn.replay_buffer, h.buffer_size)
            # Update current state
            s = s′
            # If enough data is in the replay buffer, perform some training steps
            if length(dqn.replay_buffer) == h.buffer_size
                # Get training batch
                S, A, y = sample_batch(dqn.replay_buffer, dqn.target, mdp.a2ind, h.batch_size)
                # Train
                for _ = 1:h.n_grad_steps
                    loss, back = Flux.pullback(() -> dqn_loss(dqn.policy, S, A, y), θ)
                    update!(opt, θ, back(1.0f0))
                    # println(loss)
                end
                # println()
                # Update target if time
                target_step_counter += 1
                if target_step_counter == h.update_target_every
                    dqn.target = deepcopy(dqn.policy)
                end
            end
        end
        r_ave = eval(mdp, dqn.policy, episode, h.save_folder)
        set_postfix(episodes, R="$r_ave")
        r_average[episode] = r_ave
    end
    return r_average
end



