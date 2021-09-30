using Gen

#########################
# resampling combinator #
#########################

# take as input a generative function (the proposal) and arguments to that
# generative function and a procedure for computing the unnormalized weight

# arguments to the generative function:
# 1 the arguments to the model
# 2 the observations on the model
# 3 the arguments to the proposal
# 4 the number of particles (K)

# the proposal should not produce entire traces of the model
# it should just produce proposed traces...

# the generative function the same return value as the proposal

struct ResampleGF{T,U} <: GenerativeFunction{T,Trace}
    model::GenerativeFunction
    proposal::GenerativeFunction{T,U}
end

# construct using default constructor (ResampleGF(my_gen_fn))

struct ResampleGFTrace{T,U} <: Gen.Trace

    gen_fn::ResampleGF{T,U}

    # arguments to the model generaative function
    model_args::Tuple

    # arguments to the proposal generative function
    proposal_args::Tuple

    # when combined with choices made by proposal,
    # uniquely determine a model trace
    observations::ChoiceMap 

    # number of particles
    num_particles::Int

    # the chosen trace of proposal generative function
    chosen_particle::U

    # score
    score::Float64
end

Gen.get_gen_fn(trace::ResampleGFTrace) = trace.gen_fn

function Gen.get_args(trace::ResampleGFTrace)
    return (trace.model_args, trace.observations, trace.proposal_args, trace.num_particles)
end

Gen.get_score(trace::ResampleGFTrace) = trace.score

Gen.get_retval(trace::ResampleGFTrace) = get_retval(trace.chosen_particle)

Base.getindex(trace::ResampleGFTrace, addr) = trace.chosen_particle[addr]

Gen.get_choices(trace::ResampleGFTrace) = get_choices(trace.chosen_particle)

Gen.project(trace::ResampleGFTrace, ::EmptySelection) = 0.0

Gen.project(trace::ResampleGFTrace, ::AllSelection) = trace.score

Gen.project(trace::ResampleGFTrace, ::Selection) = error("not implemented")

function Gen.simulate(gen_fn::ResampleGF, args::Tuple)
    (model_args, observations, proposal_args, num_particles) = args

    proposal_traces = Vector{U}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)
    model_scores = Vector{Float64}(undef, num_particles)

    for i in 1:num_particles

        # sample from proposal
        proposal_trace = simulate(gen_fn.proposal, proposal_args)
        proposal_traces[i] = proposal_trace
        proposed_choices = get_choices(proposal_trace)

        # combine with observations to form a model trace
        constraints = merge(observations, proposed_choices)
        (model_trace, model_score) = generate(model, model_args, constraints)

        # let's require generate to be deterministic for now
        @assert isapprox(model_score, get_score(model_trace)) 

        # record the model joint density (model_score) and importance weight
        model_scores[i] = model_score
        log_weights[i] = model_score - get_score(proposal_trace)
    end
    
    # sample particle in proposal to weights
    log_total_weight = Gen.logsumexp(log_weights)
    normalized_weights = exp.(log_weights .- log_total_weight)
    chosen_idx = categogorical(normalized_weights)
    chosen_particle = proposal_traces[chosen_idx]

    # compute score (our estimate of the marginal density of our choices)
    log_ml_estimate = log_total_weight - log(num_particles)
    score = model_scores[chosen_idx] - log_ml_estimate

    return ResampleGFTrace(
        gen_fn,
        model_args,
        proposal_args,
        observations,
        num_particles,
        chosen_particle,
        score)
end

function Gen.generate(gen_fn::ResampleGF, args::Tuple, constraints::ChoiceMap)
    (model_args, observations, proposal_args, num_particles) = args
end

# returns a generative function that has the same trace structure as the
# 'proposal' generative function

# the number of particles (K) is an argument to the generative function

# simulate() will run simulate() on the proposal K times
# then it will sample one, then it will return a trace
# that contains that, with a score..

# generate() will only (initially) run given constraints
# that uniquely determine a trace of the proposal


