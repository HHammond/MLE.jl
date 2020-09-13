module MLE

__precompile__(true)

using Distributions
using Optim
using LinearAlgebra


const EULER_MASCHERONI = big"0.57721566490153286060651209008240243104215933593992"

const REAL_SUPPORT = [-Inf, Inf]
const NON_NEGATIVE_SUPPORT = [0, Inf]
const UNIT_SUPPORT = Float64[0, 1]


"""
    parameters(D)

Extract the names of parameters from the distribution D to be estimated.

# Examples:

```jldoctest
julia> parameters(Normal)
(:μ, :σ)
```

"""
parameters(::Type{D}) where {D <: UnivariateDistribution} = fieldnames(D)

"""
    mle_starting_point(D, data)

Compute a starting point for minimizing log likelihood of the distribution D
for the dataset.

# Examples:

```jldoctest
julia> mle_starting_point(Normal, [1,2,3])
(μ = 2.0, σ = 1.0)
```

"""
function mle_starting_point(::Type{D}, data) where {D<:UnivariateDistribution}
    # ones(Float64, fieldcount(D))
    params = parameters(D)
    (; zip(parameters(D), ones(Float64, shape(params)))...)
end

function mle_starting_point(::Type{Gumbel}, data)
    μ = mean(data)
    θ = var(data)
    (;μ, θ)
end

parameter_support(::Type{Gumbel}) = (μ=REAL_SUPPORT, θ=NON_NEGATIVE_SUPPORT)

parameter_support(::Type{Weibull}) = (α=NON_NEGATIVE_SUPPORT, θ=NON_NEGATIVE_SUPPORT)
mle_starting_point(::Type{Weibull}, data) = (;α = mean(data), θ = var(data))

mle_starting_point(::Type{Normal}, data) = (; μ = mean(data), σ = var(data))
parameter_support(::Type{Normal}) = (μ=REAL_SUPPORT, σ=NON_NEGATIVE_SUPPORT)

mle_starting_point(::Type{Poisson}, data) = (; λ = mean(data))
parameter_support(::Type{Poisson}) = (; λ=NON_NEGATIVE_SUPPORT)

mle_starting_point(::Type{NegativeBinomial}, data) = (; p = mean(data) / maximum(data), r=maximum(data))
parameter_support(::Type{NegativeBinomial}) = (; p=UNIT_SUPPORT, r=NON_NEGATIVE_SUPPORT)

mle_starting_point(::Type{Gamma}, data) = (α=mean(data), θ=var(data))
parameter_support(::Type{Gamma}) = (α=NON_NEGATIVE_SUPPORT, θ=NON_NEGATIVE_SUPPORT)

mle_starting_point(::Type{Binomial}, data) = (;p=mean(data) / maximum(data), n=maximum(data))
parameter_support(::Type{Binomial}) = (;p=UNIT_SUPPORT, n=NON_NEGATIVE_SUPPORT)
Binomial(n::Float64, p::Float64; check_args) = Binomial(Int(n), p; check_args)

mle_starting_point(::Type{Geometric}, data) = (; p=1.0/mean(data))
parameter_support(::Type{Geometric}) = (; p=UNIT_SUPPORT)

mle_starting_point(::Type{Beta}, data) = (α=1.0, β=1.0)
parameter_support(::Type{Beta}) = (α=NON_NEGATIVE_SUPPORT, β=NON_NEGATIVE_SUPPORT)

mle_starting_point(::Type{Exponential}, data) = (;θ = mean(data))
parameter_support(::Type{Exponential}) = (;θ=NON_NEGATIVE_SUPPORT)


struct MLEConfig{D}
    distribution::D
    data::AbstractArray
    params::Tuple
    params_to_solve::Vector{Bool}
    starting_point::Vector
    constraints::Optim.AbstractConstraints
    custom_bounds::Bool
end

function make_constraints(d::Type{<:UnivariateDistribution}, start, holding, bounds)
    params = parameters(d)

    lower = [ bounds[p][1] for p in params if p ∈ keys(start) && p ∉ keys(holding) ]
    upper = [ bounds[p][2] for p in params if p ∈ keys(start) && p ∉ keys(holding) ]

    TwiceDifferentiableConstraints(lower, upper)
end


function _get_bounds(d::Type{<:UnivariateDistribution}, bounds::Nothing)
    parameter_support(d)
end
function _get_bounds(d::Type{<:UnivariateDistribution}, bounds)
    params = parameters(d)

    foreach(params) do p
        if p ∉ keys(bounds)
            bounds = (; p=> REAL_SUPPORT, bounds...)
        end
    end
    bounds
end

function prepare_mle(d::Type{<:UnivariateDistribution},
                     data::AbstractArray;
                     start=nothing,
                     bounds=nothing,
                     holding=nothing,
                     kwargs...)

    params = parameters(d)


    start = isnothing(start) ? mle_starting_point(d, data) : start
    holding = isnothing(holding) ? (;) : holding

    to_solve = Bool[(p ∈ keys(start)) && (p ∉ keys(holding)) for p in params]

    starting_point = Float64[
        get(holding, p, get(start, p, missing)) |> Float64
        for p in params
    ]
    @assert !any(ismissing(v) for v in starting_point)

    custom_bounds = !isnothing(bounds)
    bounds = _get_bounds(d, bounds)
    constraints = make_constraints(d, start, holding, bounds)

    MLEConfig(d,
              data,
              params,
              to_solve,
              starting_point,
              constraints,
              custom_bounds)
end

function fit_numerical_mle(d::Type{<:UnivariateDistribution},
                 data::AbstractArray,
                 start,
                 bounds=nothing;
                 kwargs...)
    fit_numerical_mle(d, data; start=start, bounds=bounds, kwargs...)
end

"""
    fit_numerical_mle(D, data[, start, bounds, holding])

Use numerical methods to fit a MLE from a distribution with twice
differentiable density. MLEs are fit using Optim with the IPNewton constrained
solver.

"""
function fit_numerical_mle(d::Type{<:UnivariateDistribution},
                 data::AbstractArray;
                 start=nothing,
                 bounds=nothing,
                 holding=nothing,
                 kwargs...)

    config = prepare_mle(d, data; start, bounds, holding, kwargs...)
    mle_solve(config; kwargs...)
end

function mle_solve(config::MLEConfig; kwargs...)
    parameters = copy(config.starting_point)
    to_solve = config.params_to_solve

    function objective!(x)
        parameters[to_solve] = x
        dist = config.distribution(parameters...)

        -loglikelihood(dist, config.data)
    end

    vars = parameters[to_solve]
    func = TwiceDifferentiable(objective!, vars)

    opt = optimize(func,
                   config.constraints,
                   vars,
                   IPNewton(),
                   Optim.Options(),
                  )

    converged = Optim.converged(opt)
    loglike = -Optim.minimum(opt)
    result = Optim.minimizer(opt)
    fit_parameters = (; zip(config.params, parameters)...)
    n = Optim.iterations(opt)

    summary = (;
        converged,
        type=(all(to_solve) && !config.custom_bounds ? :MLE : :Restricted),
        fit_parameters,
        fit=config.distribution(parameters...; check_args=false),
        result,
        loglikelihood=loglike,
        n
    )

    if converged
        objective!(result)
        H = Optim.hessian!(func, result)

        vcov = inv(H)

        D = diag(vcov)
        if any(D .< 0)
            D = Complex.(D)
        end
        sd = sqrt.(D)

        (; summary..., H, vcov, sd)
    else
        summary
    end
end


export parameter_support, mle_starting_point, fit_numerical_mle

end # module
