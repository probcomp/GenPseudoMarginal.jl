import Random

@testset "smc" begin

    Random.seed!(1)

    function hmm_forward_alg(prior::Vector{Float64},
            emission_dists::AbstractArray{Float64,2}, transition_dists::AbstractArray{Float64,2},
            emissions::Vector{Int})
        marg_lik = 1.
        alpha = prior # p(z_1)
        for i=2:length(emissions)
    
            # p(z_{i-1} , y_{i-1} | y_{1:i-2}) for each z_{i-1}
            prev_posterior = alpha .* emission_dists[emissions[i-1], :]
    
            # p(y_{i-1} | y_{1:i-2})
            denom = sum(prev_posterior)
    
            # p(z_{i-1} | y_{1:i-1})
            prev_posterior = prev_posterior / denom
    
            # p(z_i | y_{1:i-1})
            alpha = transition_dists * prev_posterior
    
            # p(y_{1:i-1})
            marg_lik *= denom
        end
        prev_posterior = alpha .* emission_dists[emissions[end], :]
        denom = sum(prev_posterior)
        marg_lik *= denom
        return marg_lik
    end
    
    num_hidden_states = 3
    num_emission_states = 2
    init_state_prior = [0.5, 0.3, 0.2]
    emission_dists = [ # 2 x 3
        0.1 0.4 0.2;
        0.9 0.6 0.8
    ]
    transition_dists = [ # 3 x 3
        0.8 0.1 0.1;
        0.1 0.8 0.1;
        0.1 0.1 0.8
    ]
    
    @gen function hidden_markov_model(init_state_prior, T::Int)
        @assert T >= 1
        z = ({(:z, 1)} ~ categorical(init_state_prior))
        {(:y, 1)} ~ categorical(emission_dists[:,z])
        for t in 2:T
            z = ({(:z, t)} ~ categorical(transition_dists[:,z]))
            {(:y, t)} ~ categorical(emission_dists[:,z])
        end
    end

    gen_fn = PFPseudoMarginalGF(
        model = hidden_markov_model,
        data_addrs = T -> [(:y, T)],
        get_step_args = (args, T) -> (args[1], T,),
        num_particles = 100000,
        get_T = (args) -> args[2],
        reuse_particle_system = (args1, args2, argdiffs) -> (argdiffs[1] == NoChange())
    )
    
    ys = [1, 2]
    log_marginal_likelihood_1 = log(hmm_forward_alg(init_state_prior, emission_dists, transition_dists, ys[1:1]))
    log_marginal_likelihood_2 = log(hmm_forward_alg(init_state_prior, emission_dists, transition_dists, ys[1:2]))

    # generate with one time step
    @time trace, log_weight = generate(gen_fn, (init_state_prior, 1,), choicemap(((:y, 1), ys[1])))
    @test isapprox(get_score(trace), log_marginal_likelihood_1, rtol=1e-2)
    @test isapprox(log_weight, log_marginal_likelihood_1, rtol=1e-2)
    @test get_choices(trace) == choicemap(((:y, 1), ys[1]))

    # extending by one time step
    @time new_trace, log_weight, retdiff, discard = update(
        trace, (init_state_prior, 2,), (NoChange(), UnknownChange(),), choicemap(((:y, 2), ys[2])))
    @test isapprox(log_weight, log_marginal_likelihood_2 - log_marginal_likelihood_1, rtol=1e-2)
    @test isapprox(get_score(new_trace), log_marginal_likelihood_2, rtol=1e-2)
    @test get_choices(new_trace) == choicemap(((:y, 1), ys[1]), ((:y, 2), ys[2]))

    # re-executing the two-time-step particle system
    alternate_init_state_prior = [0.3, 0.3, 0.4]
    alternate_log_marginal_likelihood_2 = log(hmm_forward_alg(alternate_init_state_prior, emission_dists, transition_dists, ys[1:2]))
    @time new_trace, log_weight, retdiff, discard = update(
        new_trace, (alternate_init_state_prior, 2,), (UnknownChange(), NoChange(),), choicemap())
    @test isapprox(log_weight, alternate_log_marginal_likelihood_2 - log_marginal_likelihood_2, rtol=1e-2)
    @test isapprox(get_score(new_trace), alternate_log_marginal_likelihood_2, rtol=1e-2)
    @test get_choices(new_trace) == choicemap(((:y, 1), ys[1]), ((:y, 2), ys[2]))
end
