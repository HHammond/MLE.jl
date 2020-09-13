using Test
using MLE
using Distributions
using Random
using Optim
using RCall


@testset "Test MLE" begin
    Random.seed!(1234)
    data = rand(Gumbel(1, 1), 1_000)

    @testset "Test Normal" begin
	f = fit_numerical_mle(Normal, data)
	@test f.converged

	f = fit_numerical_mle(Normal, data, (μ=10, σ=2))
	@test f.converged

	f = fit_numerical_mle(Normal, data, (μ=10, σ=2), bounds=(μ=[5,Inf], ))
	@test f.converged
    end

    f = fit_numerical_mle(Gumbel, data)
    @test f.converged
    @test isapprox(f.result, [0.9843, 0.9863]; atol=1e-3)

    f = fit_numerical_mle(Gumbel, data, holding=(; θ=1))
    @test f.converged

    pois_data = [1,2,1,3]

    f = fit_numerical_mle(Poisson, pois_data)
    @test f.converged
    @test isapprox(f.result, [1.75])

    R"""
    library(MASS)
    r_mle <- fitdistr($pois_data, "poisson")
    """
    @rget r_mle
    @test isapprox(f.result[1], r_mle[:estimate])
    @test isapprox(f.sd[1], r_mle[:sd])
    @test isapprox(f.loglikelihood, r_mle[:loglik])

    f = fit_numerical_mle(NegativeBinomial, pois_data)
    @test f.converged
    @test isapprox(mean(f.fit), 1.75; atol=1e-4)

    R"""
    library(MASS)
    r_mle <- fitdistr($pois_data, "negative binomial")
    """
    @rget r_mle
    @test isapprox(mean(f.fit), r_mle[:estimate][2]; atol=1e-4)

    f = fit_numerical_mle(NegativeBinomial, pois_data, holding=(;r=0.5))
    @test f.converged
    @test isapprox(mean(f.fit), 1.75; atol=1e-4)

    binom_data = [0, 1, 2, 0, 1, 2]
    f = fit_numerical_mle(Binomial, binom_data, holding=(;n=4))
    @test f.converged
    @test isapprox(f.result[1], 0.25; atol=1e-4)
    @test isapprox(f.result[1], mean(binom_data) / 4)

    f = fit_numerical_mle(Geometric, pois_data)
    @test f.converged
    @test isapprox(mean(f.fit), mean(pois_data); atol=1e-4)

    beta_data = [0.1, 0.2, 0.1, 0.2, 0.9]
    f = fit_numerical_mle(Beta, beta_data)
    @test f.converged
    @test isapprox(mean(f.fit), 0.35409; atol=1e-4)

    R"""
    library(MASS)
    r_mle <- fitdistr($beta_data, "beta", start=list(shape1=1, shape2=1))
    """
    @rget r_mle
    @test isapprox(f.result, r_mle[:estimate]; atol=1e-4)
    @test isapprox(f.vcov, r_mle[:vcov]; atol=1e-4)
    @test isapprox(f.sd, r_mle[:sd]; atol=1e-4)
    @test isapprox(f.loglikelihood, r_mle[:loglik])

    expon_data = [0.1, 0.2, 1.0, .3, .05]
    f = fit_numerical_mle(Exponential, expon_data)
    @test f.converged
    @test isapprox(mean(f.fit), mean(expon_data); atol=1e-2)

    R"""
    library(MASS)
    r_mle <- fitdistr($expon_data, "exponential")
    """
    @rget r_mle
    @test isapprox(1 ./ f.result[1], r_mle[:estimate]; atol=1e-2)
    @test isapprox(f.loglikelihood, r_mle[:loglik])

    f = fit_numerical_mle(Weibull, expon_data)
    @test f.converged

    R"""
    library(MASS)
    r_mle <- fitdistr($expon_data, "weibull")
    """
    @rget r_mle
    @test isapprox(f.result, r_mle[:estimate]; atol=1e-4)
    @test isapprox(f.sd, r_mle[:sd]; atol=1e-4)
    @test isapprox(f.loglikelihood, r_mle[:loglik])
end
