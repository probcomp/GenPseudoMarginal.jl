using Gen

########################################################
# Selection-based SIR (sampling importance resampling) #
########################################################

function selection_sir(trace::Trace, selection::Selection, num_particles::Int)
    weights = Vector{Float64}(undef, num_particles)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    traces = Vector{Trace}(undef, num_particles)
    offset = project(trace, ComplementSelection(selection))
    Threads.@threads for i=1:num_particles
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
    traces = Vector{Trace}(undef, num_particles)
    traces[1] = trace
    input_trace_weight = project(trace, ComplementSelection(selection))
    weights[1] = input_trace_weight
    Threads.@threads for i=2:num_particles
        (traces[i], weight, _) = regenerate(trace, args, argdiffs, selection)
        weights[i] = weight + input_trace_weight
    end
    idx = categorical(exp.(weights .- logsumexp(weights)))
    lml_est = logsumexp(weights) - log(num_particles)
    (lml_est, traces[idx], weights)
end

export selection_sir, conditional_selection_sir

######################################################
# Selection-based SIR generative function combinator #
######################################################

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
    (lml_est, _, weights) = conditional_selection_sir(new_trace, selection, num_particles)
    score = get_score(new_trace) - lml_est
    trace = SelectionSIRTrace(gen_fn, args, score, constraints, weights)
    (trace, score)
end

export selection_sir_gf

#####################
# MH move using SIR #
#####################

# Q: How is this different from doing rejection sampling on these choices?
# It is expensive like rejection sampling, but doesn't require a bound, and has
# predictable running time, and can still accept even if you don't have an
# exact sample.

# Q: How is this different than just doing selection MH in a loop?
# A: It can be parallelized. Also, it can be composed with other proposals in a
# single MH step.

# Q: How is this different than particle Gibbs?
# A: It is more compositional (it can be composed with other proposals in an MH
# step). Also it may be less sticky.

function selection_sir_mh(trace, selection::Selection, num_particles::Int)
    mh(trace, selection_sir_gf, (selection, num_particles))
end

export selection_sir_mh

#################################
# particle Gibbs move using SIR #
#################################

function sir_pgibbs(trace, selection::Selection, num_particles::Int)
    (_, new_trace, _) = conditional_selection_sir(trace, selection, num_particles)
    new_trace
end

export sir_pgibbs
