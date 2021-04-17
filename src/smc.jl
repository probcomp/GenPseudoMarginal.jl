using Gen
using Parameters: @with_kw
using GenStateSpaceModels
import FunctionalCollections

# [ ] accept an init(), dynamics(), and observation() generative functions..

# init(path, obs_model_params) -> state
# dynamics(path, obs_model_params, prev_state) -> state
# emission(path, obs_model_params, state) -> nothing


# GenStateSpaceModels has monolithic 'dynamics_params' and 'emission_params'
# that's fine

# args will be 
# TODO return value will be nothing..
# we can implement trace getters that return aux data..
# we could sample a distinguished particle and expose the steps variables that way..

struct PFPseudoMarginalTrace{T,U}
    gen_fn::GenerativeFunction{T,U}
    particle_filter_state::ParticleFilterState{U}
    all_args::FunctionalCollections.PersistentVector{Tuple}
    all_data::FunctionalCollections.PersistentVector{ChoiceMap}
    T::Int
end

Base.get_args(trace::PFPseudoMarginalTrace) = trace.args
Base.get_score(trace::PFPseudoMarginalTrace) = trace.lml_estimate

struct PFPseudoMarginalGF{T,U} <: GenerativeFunction{T,PFPseudoMarginalTrace{U}}
    model::GenerativeFunction{T,U}

    # a function from an integer >= 1 (step) to a collection of addresses
    data_addrs::Function 

    # a function from (arg-Tuple, T) to a Tuple such that
    # reuse_particle_system(input, output) == true and such that get_T(output)
    # == T
    get_step_args::Function 

    # a function from an integer >= 1 (step) to a generative function # TODO currently just using fwd sim.
    #get_proposal::Function 

    num_particles::Int

    # a function from an argument tuple to an integer >= 1
    get_T::Function 

    reuse_particle_system::Function # a function of two argument tuples
end

function reuse_particle_system(gen_fn::PFPseudoMarginalGF, prev_args, args, argdiffs)
    return gen_fn.reuse_particle_system(prev_args, args, argdiffs)
end

get_T(gen_fn::PFPseudoMarginalGF, args) = gen_fn.get_T(args)
data_addrs(gen_fn::PFPseudoMarginalGF, t) = gen_fn.data_addrs(t)
num_particles(gen_fn::PFPseudoMarginalGF) = gen_fn.num_particles
get_model(gen_fn::PFPseudoMarginalGF) = gen_fn.model

function extend_pf(
        pf_state::Gen.ParticleFilterState, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap,
        prev_T::Int, new_T::Int)
    # TODO avoid copying with a better data structure
    pf_state = Gen.ParticleFilterState(
        copy(trace.pf_state.traces),
        copy(trace.pf_state.new_traces),
        copy(trace.pf_state.log_weights),
        trace.pf_state.log_ml_est,
        copy(trace.pf_state.parents))
    @assert new_T == prev_T + 1
    @assert maybe_resample!(pf_state; ess_threshold=Inf) # always resample
    particle_filter_step!(pf_state, args, argdiffs, constraints)
    return pf_state
end

function fresh_pf(
        gen_fn::PFPseudoMarginalGF,
        all_args::AbstractVector{Tuple}, all_data::AbstractVector{ChoiceMap},
        argdiffs::Tuple)
    @assert length(all_args) == length(all_data)
    pf_state = initialize_particle_filter(
        gen_fn.model, all_args[1], all_data[1], gen_fn.num_particles)
    #@assert get_T(all_args[1]) == 1
    #T = 2
    #prev_args = all_args[
    for (args, data) in zip(all_args[2:end], all_data[2:end])
        #@assert T == get_T(gen_fn, args)
        @assert maybe_resample!(pf_state; ess_threshold=Inf) # always resample
        particle_filter_step!(pf_state, args, argdiffs, data)
    end
    return pf_state
end

function Gen.simulate(gen_fn::PFPseudoMarginalGF, args::Tuple)
    error("not implemented")
    # 1. simulate from the model
    # 2. run conditional SMC
    # 3. record marginal likelihood estimate and particle system
end

function Gen.generate(gen_fn::PFPseudoMarginalGF{T,U}, args::Tuple, constraints::ChoiceMap) where {T,U}
    if get_T(args) != 1
        error("not implemented")
    end
    all_args = FunctionalCollections.PersistentVector{Tuple}()
    all_args = FunctionalCollections.push(args)
    all_data = FunctionalCollections.PersistentVector{ChoiceMap}()
    all_data = FunctionalCollections.push(constraints)
    pf_state = fresh_pf(gen_fn, all_args, all_data, ()) # argdiffs is unused
    trace = PFPseudoMarginalTrace{T,U}(
        gen_fn, pf_state, all_args, all_data, get_T(args))
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
            end
            if isempty(get_selected(constraints, complement(select(data_addrs(new_T)...))))
                # handled case
                # use case: extending particle system via update, within outer SMC
                # reuse_particle_system(prev_args, args) == true,
                # and there are no constraints other than for the new time step's observations
                (pf_state, log_increment) = extend_pf(trace.particle_filter_state, args, argdiffs, constraints, prev_T, new_T)
                all_args = FunctionalCollections.push(trace.all_args, args)
                all_data = FunctionalCollections.push(trace.all_data, constraints)
                trace = PseudoMarginalTrace(trace.gen_fn, pf_state, all_args, all_data, new_T)
                weight = log_increment
                retdiff = NoChange()
                discard = EmptyChoiceMap()
                return (trace, weight, retdiff, discard)
            else
                error("support for extending the particle system while also updating previous choices has not been implemented")
            end
        elseif new_T < prev_T
            # 1. contract the SMC particle system by the necessary number of steps
            # 2. record the new marginal likelihood estimate as the score
            # 3. for the weight, return the sum of the log averages of the incremental weights for each removed time step
            # (NOTE this sub-case is not a priority to implement)
            error("support for retracting the particle system has not been implemented")
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
            prev_ml_estimate = log_ml_estimate(trace.pf_state)
            all_args = FunctionalCollections.PersistentVector{Tuple}([get_step_args(args, T) for T in 1:new_T])
            all_data = trace.all_data
            pf_state = fresh_pf(gen_fn, all_args, all_data, num_particles(gen_fn))
            trace = PseudoMarginalTrace(gen_fn, pf_state, all_args, all_data, new_T)
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
