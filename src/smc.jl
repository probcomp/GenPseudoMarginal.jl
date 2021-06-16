using Gen
using Parameters: @with_kw
import FunctionalCollections

export PFPseudoMarginalGF
export encapsulate

@with_kw struct PFPseudoMarginalGF{T,U} <: GenerativeFunction{Nothing,Trace}
    model::GenerativeFunction{T,U}

    # a function from an integer >= 1 (step) to a collection of addresses
    data_addrs::Function 

    # a function from (arg-Tuple, T) to a Tuple such that
    # reuse_particle_system(input, output) == true and such that get_T(output)
    # == T
    get_step_args::Function 

    # a function from an integer >= 1 (step) to a generative function
    get_proposal::Function = (T -> nothing)

    num_particles::Int

    # a function from an argument tuple to an integer >= 1
    get_T::Function 

    reuse_particle_system::Function # a function of two argument tuples
end

struct PFPseudoMarginalTrace{T,U} <: Gen.Trace
    gen_fn::PFPseudoMarginalGF{T,U}
    pf_state::Gen.ParticleFilterState{U}
    all_args::FunctionalCollections.PersistentVector{Tuple}
    all_argdiffs::FunctionalCollections.PersistentVector{Tuple}
    all_data::FunctionalCollections.PersistentVector{ChoiceMap}
    distinguished_particle::Int
    T::Int
end

Gen.get_gen_fn(trace::PFPseudoMarginalTrace) = trace.gen_fn
Gen.get_args(trace::PFPseudoMarginalTrace) = trace.all_args[end]
Gen.get_score(trace::PFPseudoMarginalTrace) = log_ml_estimate(trace.pf_state)
Gen.get_retval(trace::PFPseudoMarginalTrace) = get_retval(trace.pf_state.traces[trace.distinguished_particle])
Base.getindex(trace::PFPseudoMarginalTrace, addr) = trace.pf_state.traces[trace.distinguished_particle][addr]
Gen.get_choices(trace::PFPseudoMarginalTrace) = merge(trace.all_data...)
Gen.project(trace::PFPseudoMarginalTrace, ::Selection) = error("not implemented")
Gen.project(trace::PFPseudoMarginalTrace, ::AllSelection) = log_ml_estimate(trace.pf_state)

# NOTE: to support projecting on empty selection, we would need to define an
# internal proposal on the observed data that we can assess. then, project on
# an empty selection would return the ratio of the SMC marginal likelihood
# estimate and this internal proposal density.
# we should consider removing this from the interface...
Gen.project(trace::PFPseudoMarginalTrace, ::EmptySelection) = 0.0 # TODO

function reuse_particle_system(gen_fn::PFPseudoMarginalGF, prev_args, args, argdiffs)
    return gen_fn.reuse_particle_system(prev_args, args, argdiffs)
end

get_T(gen_fn::PFPseudoMarginalGF, args) = gen_fn.get_T(args)
data_addrs(gen_fn::PFPseudoMarginalGF, T) = gen_fn.data_addrs(T)
num_particles(gen_fn::PFPseudoMarginalGF) = gen_fn.num_particles
get_model(gen_fn::PFPseudoMarginalGF) = gen_fn.model
get_step_args(gen_fn::PFPseudoMarginalGF, args, T) = gen_fn.get_step_args(args, T)

function sample_distinguished_particle(pf_state::Gen.ParticleFilterState)
    (_, log_normalized_weights) = Gen.normalize_weights(pf_state.log_weights)
    weights = exp.(log_normalized_weights)
    return categorical(weights / sum(weights))
end

function pseudomarginal_particle_filter_step!(
        pf_state, args, argdiffs, constraints,
        ::Nothing)
    return particle_filter_step!(pf_state, args, argdiffs, constraints)
end

function pseudomarginal_particle_filter_step!(
        pf_state, args, argdiffs, constraints,
        proposal::GenerativeFunction)
    return particle_filter_step!(
        pf_state, args, argdiffs, constraints,
        proposal, (constraints,))
end

function pseudomarginal_initialize_particle_filter(
        model, args, data, num_particles,
        ::Nothing)
    return initialize_particle_filter(
        model, args, data, num_particles)
end

function pseudomarginal_initialize_particle_filter(
        model, args, data, num_particles,
        proposal::GenerativeFunction)
    return initialize_particle_filter(
        model, args, data,
        proposal, (model, args, data,), # NOTE the proposal accepts arguments to the model and data
        num_particles)
end

function extend_pf(
        pf_state::Gen.ParticleFilterState, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap,
        prev_T::Int, new_T::Int, get_proposal::Function)
    # TODO avoid copying with a better data structure
    pf_state = Gen.ParticleFilterState(
        copy(pf_state.traces),
        copy(pf_state.new_traces),
        copy(pf_state.log_weights),
        pf_state.log_ml_est,
        copy(pf_state.parents))
    @assert new_T == prev_T + 1
    @assert maybe_resample!(pf_state; ess_threshold=Inf) # always resample
    log_incremental_weights, = pseudomarginal_particle_filter_step!(
        pf_state, args, argdiffs, constraints, get_proposal(new_T))
    log_increment = logsumexp(log_incremental_weights) - log(length(log_incremental_weights))
    distinguished_particle = sample_distinguished_particle(pf_state)
    return (pf_state, log_increment, distinguished_particle)
end

function fresh_pf(
        gen_fn::PFPseudoMarginalGF,
        all_args::AbstractVector{Tuple},
        all_argdiffs::AbstractVector{Tuple},
        all_data::AbstractVector{ChoiceMap},
        get_proposal::Function)
    @assert length(all_args) == length(all_data)
    @assert length(all_argdiffs) == (length(all_args)-1)
    pf_state = pseudomarginal_initialize_particle_filter(
        gen_fn.model, all_args[1], all_data[1], gen_fn.num_particles,
        get_proposal(1))
    for (T, args, argdiffs, data) in zip(2:length(all_args), all_args[2:end], all_argdiffs, all_data[2:end])
        @assert maybe_resample!(pf_state; ess_threshold=Inf) # always resample
        pseudomarginal_particle_filter_step!(pf_state, args, argdiffs, data, get_proposal(T))
    end
    distinguished_particle = sample_distinguished_particle(pf_state)
    return (pf_state, distinguished_particle)
end

function Gen.simulate(gen_fn::PFPseudoMarginalGF, args::Tuple)
    error("not implemented")
    # 1. simulate from the model
    # model_trace = simulate(gen_fn.model, args)
    # 2. run conditional SMC
    # 3. record marginal likelihood estimate and particle system
end

function Gen.generate(gen_fn::PFPseudoMarginalGF{T,U}, args::Tuple, constraints::ChoiceMap) where {T,U}
    if get_T(gen_fn, args) != 1
        error("not implemented")
    end
    all_args = FunctionalCollections.PersistentVector{Tuple}()
    all_args = FunctionalCollections.push(all_args, args)
    all_data = FunctionalCollections.PersistentVector{ChoiceMap}()
    all_data = FunctionalCollections.push(all_data, constraints)
    (pf_state, distinguished_particle) = fresh_pf(
        gen_fn, all_args, Tuple[], all_data, gen_fn.get_proposal) # all_argdiffs is unused
    all_argdiffs = FunctionalCollections.PersistentVector{Tuple}()
    trace = PFPseudoMarginalTrace{T,U}(
        gen_fn, pf_state, all_args, all_argdiffs, all_data, distinguished_particle, get_T(gen_fn, args))
    log_weight = log_ml_estimate(pf_state)
    return (trace, log_weight)
end

function Gen.update(
        trace::PFPseudoMarginalTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    gen_fn = trace.gen_fn
    prev_args = get_args(trace)
    prev_T = get_T(gen_fn, prev_args)
    new_T = get_T(gen_fn, args)

    if reuse_particle_system(gen_fn, prev_args, args, argdiffs)
        if new_T > prev_T
            if new_T != prev_T + 1
                error("support for extending the particle system by more than one step has not been implemented")
                # TODO this can be implemented pretty easily..
            end
            if isempty(get_selected(constraints, complement(select(data_addrs(gen_fn, new_T)...))))
                # handled case
                # use case: extending particle system via update, within outer SMC
                # reuse_particle_system(prev_args, args) == true,
                # and there are no constraints other than for the new time step's observations
                (pf_state, log_increment, distinguished_particle) = extend_pf(
                    trace.pf_state, args, argdiffs, constraints, prev_T, new_T,
                    gen_fn.get_proposal)
                all_args = FunctionalCollections.push(trace.all_args, args)
                all_argdiffs = FunctionalCollections.push(trace.all_argdiffs, argdiffs)
                all_data = FunctionalCollections.push(trace.all_data, constraints)
                trace = PFPseudoMarginalTrace(trace.gen_fn, pf_state, all_args, all_argdiffs, all_data, distinguished_particle, new_T)
                weight = log_increment
                retdiff = NoChange()
                discard = EmptyChoiceMap()
                return (trace, weight, retdiff, discard)
            else
                error("support for extending the particle system while also updating previous choices has not been implemented")
                # TODO this will require a full fresh of the particle system
            end
        elseif new_T < prev_T
            # 1. contract the SMC particle system by the necessary number of steps
            # 2. record the new marginal likelihood estimate as the score
            # 3. for the weight, return the sum of the log averages of the incremental weights for each removed time step
            # (NOTE this sub-case is not a priority to implement)
            error("support for contracting the particle system has not been implemented")
        else
            @assert new_T == prev_T
            # handled case
            # this is a no-op;
            # we don't check if the args changed; we're relying on the user's reuse_particle_system
            weight = 0.0
            retdiff = NoChange()
            discard = EmptyChoiceMap()
            return (trace, weight, retdiff, discard)
        end
    elseif isempty(constraints)
        if (new_T == prev_T)
            # handled case
            # the args have changed such that we are not reusing the particle system
            # the observed data is not modified (no new constraints)
            # use case: MCMC moves that affect the args, but that don't extend the number of steps
            # (e.g. MCMC rejuvenation moves within outer SMC)
            # TODO where do we get the argdiffs? let's reuse them from the other equivalence class... (TODO hackk..)
            prev_ml_estimate = log_ml_estimate(trace.pf_state)
            all_args = FunctionalCollections.PersistentVector{Tuple}([get_step_args(gen_fn, args, T) for T in 1:new_T])
            all_data = trace.all_data
            (pf_state, distinguished_particle) = fresh_pf(
                gen_fn, all_args, trace.all_argdiffs, all_data, gen_fn.get_proposal)
            trace = PFPseudoMarginalTrace(gen_fn, pf_state, all_args, trace.all_argdiffs, all_data, distinguished_particle, new_T)
            weight = log_ml_estimate(pf_state) - prev_ml_estimate
            retdiff = NoChange()
            discard = EmptyChoiceMap()
            return (trace, weight, retdiff, discard)
        else
            error("support for re-generating the particle system while also changing the number of time steps has not been implemented")
        end
    else
        error("support for re-generating the particle system while also updating previous choices has not been implemented")
    end

    # Q: what about the case where changing the args changes the data that gets sampled??
    # A: that is not allowed.. the set of data_addrs that is sampled must only depend on args through T

    @assert false
end

