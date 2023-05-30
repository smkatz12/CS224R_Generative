using NeuralVerification
using LazySets
using DataStructures
using LinearAlgebra

function priority_optimization(network, lbs, ubs, optimize_reach, evaluate_objective; 
        n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits = 0)
    
    start_cell = Hyperrectangle(low = lbs, high = ubs)
    initial_cells = split_multiple_times(start_cell, initial_splits)

    # Create your queue, then add your original new_cells 
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first 
    for cell in initial_cells
        enqueue!(cells, cell, optimize_reach(forward_network(solver, network, cell)))
    end

    best_lower_bound = -Inf
    best_x = nothing

    # For n_steps dequeue a cell, split it, and then 
    for i = 1:n_steps
        cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
        dequeue!(cells)
        
        # Early stopping
        if early_stop
            if i % stop_freq == 0
                lower_bound = evaluate_objective(network, cell.center)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = cell.center
                end
                # println("i: ", i)
                # println("lower bound: ", lower_bound)
                # println("best lower bound: ", best_lower_bound)
                # println("value: ", value)
                if (value .- lower_bound) ≤ stop_gap
                    return best_x, best_lower_bound, value
                end
                # println("max radius: ", max(radius(cell)))
            end
        end

        new_cells = split_cell(cell)

        # Enqueue each of the new cells
        for new_cell in new_cells
            # If you've made the max objective cell tiny
            # break (otherwise we end up with zero radius cells)
            if max(radius(new_cell) < sqrt(eps()))
                # Return a concrete value and the upper bound from the parent cell
                # that was just dequeued, as it must have higher value than all other cells
                # that were on the queue, and they constitute a tiling of the space
                lower_bound = evaluate_objective(network, cell.center)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = cell.center
                end
                return best_x, best_lower_bound, value 
            end

            new_value = optimize_reach(forward_network(solver, network, new_cell))
            enqueue!(cells, new_cell, new_value)
        end
    end

    # The largest value in our queue is the approximate optimum 
    cell, value = peek(cells)
    lower_bound = evaluate_objective(network, cell.center)

    if lower_bound > best_lower_bound
        best_lower_bound = lower_bound
        best_x = cell.center
    end
    return best_x, best_lower_bound, value
end

""" Helpers """

elem_basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

function split_multiple_times(cell, n)
    q = Queue{Hyperrectangle}()
    enqueue!(q, cell)
    for i = 1:n
        new_cells = split_cell(dequeue!(q))
        enqueue!(q, new_cells[1])
        enqueue!(q, new_cells[2])
    end
    return q
end

function split_cell(cell::Hyperrectangle)
    lbs, ubs = low(cell), high(cell)
    largest_dimension = argmax(ubs .- lbs)
    # have a vector [0, 0, ..., 1/2 largest gap at largest dimension, 0, 0, ..., 0]
    delta = elem_basis(largest_dimension, length(lbs)) * 0.5 * (ubs[largest_dimension] - lbs[largest_dimension])
    cell_one = Hyperrectangle(low=lbs, high=(ubs .- delta))
    cell_two = Hyperrectangle(low=(lbs .+ delta), high=ubs)
    return [cell_one, cell_two]
end

""" Problem Specific """
function concat_networks(generator, controller)
    gen_ps = Flux.params(generator)
    cont_ps = Flux.params(controller)
    tmp_sizes = [size(p, 1) for p in gen_ps][1:2:end]
    generator_sizes = [4; tmp_sizes]
    tmp_sizes = [size(p, 1) for p in cont_ps][1:2:end]
    controller_sizes = [128; tmp_sizes]

    layers = Any[Dense(generator_sizes[i], generator_sizes[i+1], relu) for i = 1:length(generator_sizes)-2]
    push!(layers, Dense(generator_sizes[end-1], generator_sizes[end]))
    for i = 1:length(controller_sizes)-2
        push!(layers, Dense(controller_sizes[i], controller_sizes[i+1], relu))
    end
    push!(layers, Dense(controller_sizes[end-1], controller_sizes[end]))
    concat_model = Chain(layers)

    new_params = []
    for p in gen_ps
        push!(new_params, p)
    end
    for p in cont_ps
        push!(new_params, p)
    end

    Flux.loadparams!(concat_model, new_params)

    return concat_model
end

function get_max(network, lbs, ubs)
    tmp = NeuralVerification.compute_output(network, lbs)
    N = length(tmp)
    values = zeros(N)

    # Define functions
    for i = 1:N
        coeffs = zeros(N)
        coeffs[i] = 1.0
        evaluate_objective_max(network, x) = dot(coeffs, NeuralVerification.compute_output(network, x))
        optimize_reach_max(reach) = ρ(coeffs, reach)
        x, under, values[i] = priority_optimization(network, lbs, ubs, optimize_reach_max, evaluate_objective_max)
    end

    return maximum(values)
end