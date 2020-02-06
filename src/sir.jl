using Gen

########################################
# SIR (sampling importance resampling) #
########################################

function selection_sir(trace::Trace, selection::Selection, num_particles::Int)
    weights = Vector{Float64}(undef, num_particles)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    traces = Vector{Trace}(undef, num_particles)
    # Q: How is this better than just doing selection MH in a loop?
    # A: It can be parallelized.
    offset = project(trace, ComplementSelection(selection))
    for i=1:num_particles
        (traces[i], weight, _) = regenerate(trace, args, argdiffs, selection)
        weights[i] = weight + offset
    end
    idx = categorical(exp.(weights .- logsumexp(weights)))
    lml_est = logsumexp(weights) - log(num_particles)
    (lml_est, traces[idx], weights)
end

function conditional_selection_sir(trace::Trace, selection::Selection, num_particles::Int)
    weights = Vector{Float64}(undef, num_particles)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    # Q: How is this better than just doing selection MH in a loop?
    # A: It can be parallelized.
    input_trace_weight = project(trace, ComplementSelection(selection))
    weights[1] = input_trace_weight
    for i=2:num_particles
        (_, weight, _) = regenerate(trace, args, argdiffs, selection)
        weights[i] = weight + input_trace_weight
    end
    lml_est = logsumexp(weights) - log(num_particles)
    (lml_est, weights)
end

export selection_sir, conditional_selection_sir

######################################
# SIR generative function combinator #
######################################

struct SelectionSIRTrace <: Gen.Trace
    gen_fn::GenerativeFunction
    args::Tuple
    score::Float64
    choices::ChoiceMap
    weights::Vector{Float64}
end

Gen.get_gen_fn(trace::SelectionSIRTrace) = trace.gen_fn
Gen.get_args(trace::SelectionSIRTrace) = trace.args
Gen.get_retval(trace::SelectionSIRTrace) = trace.weights
Gen.get_choices(trace::SelectionSIRTrace) = trace.choices
Gen.get_score(trace::SelectionSIRTrace) = trace.score

struct SelectionSIRGF <: GenerativeFunction{Vector{Float64},SelectionSIRTrace} end
const selection_sir_gf = SelectionSIRGF()

function Gen.simulate(gen_fn::SelectionSIRGF, args::Tuple)
    (trace::Trace, selection::Selection, num_particles::Int) = args
    (lml_est, new_trace, weights) = selection_sir(trace, selection, num_particles)
    score = get_score(new_trace) - lml_est
    output = get_selected(get_choices(new_trace), selection)
    SelectionSIRTrace(gen_fn, args, score, output, weights)
end

function Gen.generate(gen_fn::SelectionSIRGF, args::Tuple, constraints::ChoiceMap)
    (trace::Trace, selection::Selection, num_particles::Int) = args
    model_args = get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    new_trace, = update(trace, model_args, argdiffs, constraints)
    (lml_est, weights) = conditional_selection_sir(new_trace, selection, num_particles)
    score = get_score(new_trace) - lml_est
    trace = SelectionSIRTrace(gen_fn, args, score, constraints, weights)
    (trace, score)
end

export selection_sir_gf

#####################
# MH move using SIR #
#####################

function selection_sir_mh_involution(trace, fwd_choices, fwd_ret, fwd_args)
    (selection, _) = fwd_args
    bwd_choices = get_selected(get_choices(trace), selection)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_trace, weight = update(trace, args, argdiffs, fwd_choices)
    (new_trace, bwd_choices, weight + 0.)
end

function selection_sir_mh(trace, selection::Selection, n::Int; check_round_trip=false)
    mh(trace, selection_sir_gf, (selection, n), selection_sir_mh_involution;
        check_round_trip=check_round_trip)
end


export selection_sir_mh
