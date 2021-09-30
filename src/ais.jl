using Gen

#######
# AIS #
#######

"""

    (lml_est, trace, weights) = ais(
        model::GenerativeFunction, constraints::ChoiceMap,
        args_seq::Vector{Tuple}, argdiffs::Tuple,
        mcmc_kernel::Function)

Run annealed importance sampling, returning the log marginal likelihood estimate (`lml_est`).
"""
function ais(
        model::GenerativeFunction, constraints::ChoiceMap,
        args_seq::Vector{<:Tuple}, argdiffs::Tuple, mcmc_kernel::Function)
    init_trace, init_weight = generate(model, args_seq[1], constraints)
    _ais(init_trace, init_weight, args_seq, argdiffs, mcmc_kernel)
end

# TODO: make the generative function and the MCMC code below use this interface
# because it can exploit incremental computation
function ais(
        trace::Trace, selection::Selection,
        args_seq::Vector{<:Tuple}, argdiffs::Tuple, mcmc_kernel::Function)
    init_trace, = update(init_trace, args_seq[1], argdiffs, EmptyChoiceMap())
    init_weight = project(trace, ComplementSelection(selection))
    _ais(init_trace, init_weight, args_seq, argdiffs, mcmc_kernel)
end

function _ais(
        trace::Trace, init_weight::Float64, args_seq::Vector{<:Tuple},
        argdiffs::Tuple, mcmc_kernel::Function)
    @assert get_args(trace) == args_seq[1]

    # run forward AIS
    weights = Float64[]
    lml_est = init_weight
    push!(weights, init_weight)
    for intermediate_args in args_seq[2:end-1]
        trace = mcmc_kernel(trace)
        (trace, weight, _, _) = update(trace, intermediate_args, argdiffs, EmptyChoiceMap())
        lml_est += weight
        push!(weights, weight)
    end
    trace = mcmc_kernel(trace)
    (trace, weight, _, _) = update(
        trace, args_seq[end], argdiffs, EmptyChoiceMap())
    lml_est += weight
    push!(weights, weight)

    # do mh at the very end
    trace = mcmc_kernel(trace)

    (lml_est, trace, weights)
end

function reverse_ais(
        model::GenerativeFunction, constraints::ChoiceMap,
        args_seq::Vector, argdiffs::Tuple,
        mh_rev::Function, output_addrs::Selection; safe=true)

    # construct final model trace from the inferred choices and all the fixed choices
    (trace, should_be_score) = generate(model, args_seq[end], constraints)
    init_score = get_score(trace)
    if safe && !isapprox(should_be_score, init_score) # check it's deterministic
        error("Some random choices may have been unconstrained")
    end
    ais_score = init_score

    # do mh at the very beginning
    trace = mh_rev(trace)

    # run backward AIS
    lml_est = 0.
    weights = Float64[]
    for model_args in reverse(args_seq[1:end-1])
        (trace, weight, _, _) = update(trace, model_args, argdiffs, EmptyChoiceMap())
        safe && isnan(weight) && error("NaN weight")
        ais_score += weight # we are adding because the weights are the reciprocal of the forward weight
        lml_est -= weight
        push!(weights, -weight)
        trace = mh_rev(trace)
    end

    # get pi_1(z_0) / q(z_0) -- the weight that would be returned by the initial 'generate' call
    # select the addresses that would be constrained by the call to generate inside to AIS.simulate()
    @assert get_args(trace) == args_seq[1]
    #score_from_project = project(trace, ComplementSelection(output_addrs))
    score_from_project = project(trace, output_addrs)
    ais_score -= score_from_project
    println("changed!")
    lml_est += score_from_project
    push!(weights, score_from_project)
    if isnan(score_from_project)
        error("NaN score_from_project")
    end

    (lml_est, ais_score, reverse(weights))
end



######################################
# AIS generative function combinator #
######################################

struct AISTrace <: Gen.Trace
    gen_fn::GenerativeFunction
    args::Tuple
    score::Float64
    choices::ChoiceMap
    weights::Vector{Float64}
end

Gen.get_gen_fn(trace::AISTrace) = trace.gen_fn
Gen.get_args(trace::AISTrace) = trace.args
Gen.get_retval(trace::AISTrace) = trace.weights
Gen.get_choices(trace::AISTrace) = trace.choices
Gen.get_score(trace::AISTrace) = trace.score

struct AISGF <: GenerativeFunction{Vector{Float64},AISTrace} end
const selection_ais_gf = AISGF()

function Gen.simulate(gen_fn::AISGF, args::Tuple)
     (model::GenerativeFunction, model_constraints::ChoiceMap,
        args_seq::Vector{<:Tuple}, argdiffs::Tuple,
        mh_fwd::Function, mh_rev::Function, output_addrs::Selection) = args

    (lml_est, trace, weights) = ais(
        model, model_constraints,
        args_seq, argdiffs,
        mh_fwd) 

    ais_score = get_score(trace) - lml_est
    output = get_selected(get_choices(trace), output_addrs)
    AISTrace(gen_fn, args, ais_score, output, weights)
end

function Gen.generate(gen_fn::AISGF, args::Tuple, constraints::ChoiceMap)
     (model::GenerativeFunction, model_constraints::ChoiceMap,
        args_seq::Vector{<:Tuple}, argdiffs::Tuple,
        mh_fwd::Function, mh_rev::Function, output_addrs::Selection) = args

    combined_choices = merge(model_constraints, constraints)

    (_, ais_score, weights) = reverse_ais(
        model, combined_choices,
        args_seq, argdiffs, 
        mh_rev, output_addrs)

    trace = AISTrace(gen_fn, args, ais_score, constraints, weights)
    (trace, ais_score)
end

#####################
# MH move using AIS #
#####################

function selection_ais_mh(trace, selection::Selection, num_particles::Int)
    mh(trace, selection_ais_gf, (selection, num_particles))
end

export selection_sir_mh
function selection_ais_mh_involution(trace, fwd_choices, fwd_ret, fwd_args)
    (selection, _) = fwd_args
    bwd_choices = get_selected(get_choices(trace), selection)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_trace, weight = update(trace, args, argdiffs, fwd_choices)
    (new_trace, bwd_choices, weight + 0.)
end


function make_ais_mh_move(output_addrs, mh_fwd, mh_rev, args_seq, argdiffs)

    gf = AISGF()

    @gen function ais_proposal(trace)
        @trace(gf(get_gen_fn(trace), constraints, args_seq, argdiffs, mh_fwd, mh_rev), :ais)
    end

    function involution(trace, fwd_choices, fwd_ret, fwd_args)
        ais_choices = get_selected(get_choices(trace), output_addrs)
        bwd_choices = choicemap()
        set_submap!(bwd_choices, :ais, ais_choices)

        constraints = get_submap(fwd_choices, :ais)
        args = get_args(trace)
        argdiffs = map((_) -> NoChange(), args)
        new_trace, weight = update(trace, args, argdiffs, constraints)

        (new_trace, bwd_choices, weight + 0.)
    end

    function ais_move(trace; check_round_trip=false)
        mh(trace, ais_proposal, (), involution, check_round_trip=check_round_trip)
    end

    ais_move
end

export AISGF, ais, reverse_ais, make_ais_mh_move
