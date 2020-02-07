using Gen
using PyPlot
using GenSMC: selection_sir_gf

# example of using SIR for inference over continuous random variables in a
# custom reversible jump MCMC scheme (SPRJ-MCMC)

# Example from Section 4 of Reversible jump Markov chain Monte Carlo
# computation and Bayesian model determination 

########################
# custom distributions #
########################

# minimum of k draws from uniform_continuous(lower, upper)

# we can sequentially sample the order statistics of a collection of K uniform
# continuous samples on the interval [a, b], by:
# x1 ~ min_uniform_continuous(a, b, K)
# x2 | x1 ~ min_uniform_continuous(x1, b, K-1)
# ..
# xK | x1 .. x_{K-1} ~ min_uniform_continuous(x_{K-1}, b, 1)

struct MinUniformContinuous <: Distribution{Float64} end
const min_uniform_continuous = MinUniformContinuous()

function Gen.logpdf(::MinUniformContinuous, x::Float64, lower::T, upper::U, k::Int) where {T<:Real,U<:Real}
    if x > lower && x < upper
        (k-1) * log(upper - x) + log(k) - k * log(upper - lower)
    else
        -Inf
    end
end

function Gen.random(::MinUniformContinuous, lower::T, upper::U, k::Int) where {T<:Real,U<:Real}
    # inverse CDF method
    p = rand()
    upper - (upper - lower) * (1. - p)^(1. / k)
end


# piecewise homogenous Poisson process 

# n intervals - n + 1 bounds
# (b_1, b_2]
# (b_2, b_3]
# ..
# (b_n, b_{n+1}]

function compute_total(bounds, rates)
    num_intervals = length(rates)
    if length(bounds) != num_intervals + 1
        error("Number of bounds does not match number of rates")
    end
    total = 0.
    bounds_ascending = true
    for i=1:num_intervals
        lower = bounds[i]
        upper = bounds[i+1]
        rate = rates[i]
        len = upper - lower
        if len <= 0
            bounds_ascending = false
        end
        total += len * rate
    end
    (total, bounds_ascending)
end

struct PiecewiseHomogenousPoissonProcess <: Distribution{Vector{Float64}} end
const piecewise_poisson_process = PiecewiseHomogenousPoissonProcess()

function Gen.logpdf(::PiecewiseHomogenousPoissonProcess, x::Vector{Float64}, bounds::Vector{Float64}, rates::Vector{Float64})
    cur = 1
    upper = bounds[cur+1]
    lpdf = 0.
    for xi in sort(x)
        if xi < bounds[1] || xi > bounds[end]
            error("x ($xi) lies outside of interval")
        end
        while xi > upper 
            cur += 1
            upper = bounds[cur+1]
        end
        lpdf += log(rates[cur])
    end
    (total, bounds_ascending) = compute_total(bounds, rates)
    if bounds_ascending
        lpdf - total
    else
        -Inf
    end
end

function Gen.random(::PiecewiseHomogenousPoissonProcess, bounds::Vector{Float64}, rates::Vector{Float64})
    x = Vector{Float64}()
    num_intervals = length(rates)
    for i=1:num_intervals
        lower = bounds[i]
        upper = bounds[i+1]
        rate = (upper - lower) * rates[i]
        n = random(poisson, rate)
        for j=1:n
            push!(x, random(uniform_continuous, lower, upper))
        end
    end
    x
end


#########
# model #
#########

const K = :k
const EVENTS = :events
const CHANGEPT = :changept
const RATE = :rate

@gen function model(T::Float64)

    # prior on number of change points
    k = @trace(poisson(3.), K)

    # prior on the location of (sorted) change points
    change_pts = Vector{Float64}(undef, k)
    lower = 0.
    for i=1:k
        # TODO: this violates the constant support requirement...
        # this causes problems for SIR selection over it..
        cp = @trace(min_uniform_continuous(lower, T, k-i+1), (CHANGEPT, i))
        change_pts[i] = cp
        lower = cp
    end

    # k + 1 rate values
    # h$i is the rate for cp$(i-1) to cp$i where cp0 := 0 and where cp$(k+1) := T
    alpha = 1.
    beta = 200.
    rates = Float64[@trace(Gen.gamma(alpha, 1. / beta), (RATE, i)) for i=1:k+1]

    # poisson process
    bounds = vcat([0.], change_pts, [T])
    @trace(piecewise_poisson_process(bounds, rates), EVENTS)
end

function render(trace; ymax=0.02)
    T = get_args(trace)[1]
    k = trace[:k]
    bounds = vcat([0.], sort([trace[(CHANGEPT, i)] for i=1:k]), [T])
    rates = [trace[(RATE, i)] for i=1:k+1]
    for i=1:length(rates)
        lower = bounds[i]
        upper = bounds[i+1]
        rate = rates[i]
        plot([lower, upper], [rate, rate], color="black", linewidth=2)
    end
    points = trace[EVENTS]
    scatter(points, -rand(length(points)) * (ymax/5.), color="black", s=5)
    ax = gca()
    xlim = [0., T]
    plot(xlim, [0., 0.], "--")
    ax[:set_xlim](xlim)
    ax[:set_ylim](-ymax/5., ymax)
end

function show_prior_samples()
    figure(figsize=(16,16))
    T = 40000.
    for i=1:16
        println("simulating $i")
        subplot(4, 4, i)
        (trace, ) = generate(model, (T,), EmptyChoiceMap())
        render(trace; ymax=0.015)
    end
    tight_layout(pad=0)
    savefig("prior_samples.pdf")
end


#############
# rate move #
#############

@gen function rate_proposal(trace, i::Int)
    cur_rate = trace[(RATE, i)]
    @trace(uniform_continuous(cur_rate/2., cur_rate*2.), (RATE, i))
end

function rate_move(trace)
    k = trace[K]
    i = uniform_discrete(1, k+1)
    trace, acc = mh(trace, rate_proposal, (i,))
    @assert trace[K] == k # check the invariant
    trace, acc
end

#################
# position move #
#################

@gen function position_proposal(trace, i::Int)
    k = trace[K]
    lower = (i == 1) ? 0. : trace[(CHANGEPT, i-1)]
    upper = (i == k) ? T : trace[(CHANGEPT, i+1)]
    @trace(uniform_continuous(lower, upper), (CHANGEPT, i))
end

function position_move(trace)
    k = trace[K]
    @assert k > 0
    i = uniform_discrete(1, k)
    trace, acc = mh(trace, position_proposal, (i,))
    @assert trace[K] == k # check the invariant
    trace, acc
end

######################
# birth / death move #
######################

const CHOSEN = :chosen
const IS_BIRTH = :is_birth
const NEW_CHANGEPT = :new_changept

# TODO add to Gen, this is a common pattern
function Gen.update(trace::Trace, constraints::ChoiceMap)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    update(trace, args, argdiffs, constraints)
end

function birth_selection(i::Int)
    select((RATE, i), (RATE, i+1))
end

function death_selection(i::Int)
    select((RATE, i))
end

@gen function birth_death_proposal(trace, num_particles::Int)
    T = get_args(trace)[1]
    k = trace[K]

    # if k = 0, then always do a birth move
    # if k > 0, then randomly choose a birth or death move
    isbirth = (k == 0) ? true : @trace(bernoulli(0.5), IS_BIRTH)

    if isbirth
        # pick the segment in which to insert the new changepoint
        # changepoints before move:  | 1     2    3 |
        # new changepoint (i = 2):   |    *         |
        # changepoints after move:   | 1  2  3    4 |
        i = @trace(uniform_discrete(1, k+1), CHOSEN)
        lower = (i == 1) ? 0. : trace[(CHANGEPT, i-1)]
        upper = (i == k+1) ? T : trace[(CHANGEPT, i)]

        # update the structure
        constraints = choicemap()
        constraints[K] = k + 1
        constraints[(CHANGEPT, i)] = @trace(uniform_continuous(lower, upper), NEW_CHANGEPT)
        for j=i+1:k+1
            # shift up changepoints
            constraints[(CHANGEPT, j)] = trace[(CHANGEPT, j-1)]
        end
        for j=i+2:k+2
            # shift up rates
            constraints[(RATE, j)] = trace[(RATE, j-1)]
        end
        tmp_trace, = update(trace, constraints)
        selection = birth_selection(i)

    else
        # pick the changepoint to be deleted
        # changepoints before move:     | 1  2  3    4 |
        # deleted changepoint (i = 2):  |    *         |
        # changepoints after move:      | 1     2    3 |
        i = @trace(uniform_discrete(1, k), CHOSEN)

        # update the structure
        constraints = choicemap()
        constraints[K] = k - 1
        for j=i:k-1
            # shift down changepoints
            constraints[(CHANGEPT, j)] = trace[(CHANGEPT, j+1)]
        end
        for j=i+1:k
            # shift down rates
            constraints[(RATE, j)] = trace[(RATE, j+1)]
        end
        tmp_trace, = update(trace, constraints)
        selection = death_selection(i)
    end

    # propose values for continuous random variables given new structure
    @trace(selection_sir_gf(
            tmp_trace, selection,
            num_particles), :sir)

    # a trace with the structure changed, but either unchanged or randomly
    # chosen continuous variables
    tmp_trace
end

# it is an involution because:
# - it switches back and forth between birth move and death move
# - it maintains fwd_choices[CHOSEN] constant (applying it twice will first insert a new
#   changepoint and then remove that same changepoint)
# - new_rates, curried on cp_new, cp_prev, and cp_next, is the inverse of new_rates_inverse.

function birth_death_involution(trace, fwd_choices::ChoiceMap, fwd_ret, proposal_args::Tuple)
    model_args = get_args(trace)
    tmp_trace = fwd_ret
    T = model_args[1]

    bwd_choices = choicemap()

    # current number of changepoints
    k = trace[K]
    
    # if k == 0, then we can only do a birth move
    isbirth = (k == 0) || fwd_choices[IS_BIRTH]

    # if we are a birth move, the inverse is a death move
    if k > 1 || isbirth
        bwd_choices[IS_BIRTH] = !isbirth
    end
    
    # the changepoint to be added or deleted
    i = fwd_choices[CHOSEN]
    bwd_choices[CHOSEN] = i

    # populate backward choices with values of continuous random choices
    if isbirth
        set_submap!(bwd_choices, :sir,  
            get_selected(get_choices(trace), death_selection(i)))
    else
        set_submap!(bwd_choices, :sir,  
            get_selected(get_choices(trace), birth_selection(i)))
        bwd_choices[NEW_CHANGEPT] = trace[(CHANGEPT, i)]
    end

    # update the continuous random variables
    constraints = get_submap(fwd_choices, :sir)
    new_trace, = update(tmp_trace, constraints)

    weight = get_score(new_trace) - get_score(trace)
    (new_trace, bwd_choices, weight)
end

function birth_death_move(trace, num_particles::Int)
    mh(trace, birth_death_proposal, (num_particles,),
        birth_death_involution, check_round_trip=false)
end

function mcmc_step(trace)
    (trace, _) = rate_move(trace)
    if trace[K] > 0
        (trace, _) = position_move(trace)
        @assert trace[K] > 0 # check invariant
    end
    (trace, _) = birth_death_move(trace, 50)
    trace
end

function do_mcmc(T, num_steps::Int)
    (trace, _) = generate(model, (T,), observations)
    for iter=1:num_steps
        k = trace[K]
        if iter % 100 == 0
            println("iter $iter of $num_steps, k: $k")
        end
        trace = mcmc_step(trace)
    end
    trace
end


########################
# inference experiment #
########################

import Random
Random.seed!(1)

# load data set
import CSV
function load_data_set()
    df = CSV.read("$(@__DIR__)/coal.csv")
    dates = df[1]
    dates = dates .- minimum(dates)
    dates * 365.25 # convert years to days
end

const points = load_data_set()
const T = maximum(points)
const observations = choicemap()
observations[EVENTS] = points

function show_posterior_samples()
    figure(figsize=(16,16))
    for i=1:16
        println("replicate $i")
        subplot(4, 4, i)
        trace = do_mcmc(T, 100) # was 200
        render(trace; ymax=0.015)
    end
    tight_layout(pad=0)
    savefig("posterior_samples.pdf")
end

function get_rate_vector(trace, test_points)
    k = trace[K]
    cps = [trace[(CHANGEPT, i)] for i=1:k]
    hs = [trace[(RATE, i)] for i=1:k+1]
    rate = Vector{Float64}()
    cur_h_idx = 1
    cur_h = hs[cur_h_idx]
    next_cp_idx = 1
    upper = (next_cp_idx == k + 1) ? T : cps[next_cp_idx]
    for x in test_points
        while x > upper
            next_cp_idx += 1
            upper = (next_cp_idx == k + 1) ? T : cps[next_cp_idx]
            cur_h_idx += 1
            cur_h = hs[cur_h_idx]
        end
        push!(rate, cur_h)
    end
    rate
end

# compute posterior mean rate curve

function plot_posterior_mean_rate()
    test_points = collect(1.0:10.0:T)
    rates = Vector{Vector{Float64}}()
    num_samples = 0
    num_steps = 400 #8000
    for reps=1:20
        (trace, _) = generate(model, (T,), observations)
        for iter=1:num_steps
            if iter % 100 == 0
                println("iter $iter of $num_steps, k: $(trace[K])")
            end
            trace = mcmc_step(trace)
            if iter > 200#4000
                num_samples += 1
                rate_vector = get_rate_vector(trace, test_points)
                @assert length(rate_vector) == length(test_points)
                push!(rates, rate_vector)
            end
        end
    end
    posterior_mean_rate = zeros(length(test_points))
    for rate in rates
        posterior_mean_rate += rate / Float64(num_samples)
    end
    ymax = 0.010
    figure()
    plot(test_points, posterior_mean_rate, color="black")
    scatter(points, -rand(length(points)) * (ymax/6.), color="black", s=5)
    ax = gca()
    xlim = [0., T]
    plot(xlim, [0., 0.], "--")
    ax[:set_xlim](xlim)
    ax[:set_ylim](-ymax/5., ymax)
    savefig("posterior_mean_rate.pdf")
end

function plot_trace_plot()
    figure(figsize=(8, 4))

    # reversible jump
    (trace, _) = generate(model, (T,), observations)
    rate1 = Float64[]
    num_clusters_vec = Int[]
    burn_in = 0
    for iter=1:burn_in + 1000 
        trace = mcmc_step(trace)
        if iter > burn_in
            push!(num_clusters_vec, trace[K])
        end
    end
    subplot(2, 1, 1)
    plot(num_clusters_vec, "b")

    savefig("trace_plot.pdf")
end

println("showing prior samples...")
show_prior_samples()

println("showing posterior samples...")
show_posterior_samples()

println("estimating posterior mean rate...")
plot_posterior_mean_rate()

println("making trace plot...")
plot_trace_plot()
