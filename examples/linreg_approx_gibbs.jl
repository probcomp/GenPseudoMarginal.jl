using Gen
using GenSMC: selection_sir_mh, sir_pgibbs
import Random

#@gen function datum(x::Float64, slope::Float64, intercept::Float64, std::Float64)
#end

#data = Map(datum)

@gen function model(xs::Vector{Float64})
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    #std = 0.1
    std = @trace(inv_gamma(1, 1), :std)
    for (i, x) in enumerate(xs)
        @trace(normal(x * slope + intercept, std), (:y, i))
    end
end

function make_data_set(n)
    Random.seed!(1)
    true_noise = 0.1
    true_slope = -1
    true_intercept = 2
    xs = collect(range(-5, stop=5, length=n))
    ys = Float64[]
    for (i, x) in enumerate(xs)
        y = true_slope * x + true_intercept + randn() * true_noise
        push!(ys, y)
    end
    (xs, ys)
end

function do_inference(xs, ys, num_iters, num_particles)
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)
    println("init score: $(get_score(trace))")

    scores = Vector{Float64}(undef, num_iters)
    for i=1:num_iters
        println("iter $i")

        # NOTE: SIR and PGibbs are redundant here, using both for illustration

        # SIR MH moves on the parameters
        trace, acc1 = selection_sir_mh(trace, select(:slope), Int(num_particles))
        trace, acc2 = selection_sir_mh(trace, select(:intercept), Int(num_particles))
        trace, acc3 = selection_sir_mh(trace, select(:std), Int(num_particles))
        println("acc: $((acc1, acc2, acc3))")

        # PGibbs moves 
        trace = sir_pgibbs(trace, select(:slope), Int(num_particles))
        trace = sir_pgibbs(trace, select(:intercept), Int(num_particles))
        trace = sir_pgibbs(trace, select(:std), Int(num_particles))

        score = get_score(trace)
        scores[i] = score

        # print
        slope = trace[:slope]
        intercept = trace[:intercept]
        println("score: $score, slope: $slope, intercept: $intercept")
    end
    println("slope: $(trace[:slope]), intercept: $(trace[:intercept])")
    return scores
end

(xs, ys) = make_data_set(100)
@time do_inference(xs, ys, 10, Int(1e2))
@time do_inference(xs, ys, 10, Int(1e2))

