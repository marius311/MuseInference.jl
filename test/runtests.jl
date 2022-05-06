
using MuseInference, Soss, Turing, Distributions, Random, 
    Zygote, MeasureTheory, Test, StableRNGs

##

rng = StableRNG(0)

@testset "MuseInference" begin
        
    @testset "Turing w/ Distributions" begin 

        Turing.@model function turing_funnel()
            θ ~ Distributions.Normal(0, 3)
            z ~ Distributions.MvNormal(512, exp(θ/2))
            x ~ Distributions.MvNormal(z, 1)
        end

        @testset "$name" for (name,autodiff) in [
            ("ForwardDiff", MuseInference.ForwardDiffBackend()), 
            ("Zygote", MuseInference.ZygoteBackend())
        ]
            
            (;x) = rand(copy(rng), turing_funnel() | (θ=0,))
            prob = TuringMuseProblem(turing_funnel() | (;x); autodiff)
            MuseInference.check_self_consistency(prob, (θ=1,), has_volume_factor=true, rng=copy(rng))
            result = muse(prob, (θ=1,); rng=copy(rng), get_covariance=true)
            @test result.dist.μ / result.dist.σ < 1

        end

    end

    ##

    @testset "Soss w/ Distributions" begin 

        soss_funnel = Soss.@model (σ) begin
            θ ~ Distributions.Normal(0, σ)
            z ~ Distributions.MvNormal(512, exp(θ/2))
            x ~ Distributions.MvNormal(z, 1)
        end

        @testset "$name" for (name,autodiff) in [
            ("ForwardDiff", MuseInference.ForwardDiffBackend()), 
            ("Zygote", MuseInference.ZygoteBackend())
        ]
            (;x) = Soss.predict(copy(rng), soss_funnel(3), θ=0)
            prob = SossMuseProblem(soss_funnel(3) | (;x); autodiff)
            MuseInference.check_self_consistency(prob, (θ=1,), has_volume_factor=false, rng=copy(rng))
            result = muse(prob, (θ=1,); rng=copy(rng), get_covariance=true)
            @test result.dist.μ / result.dist.σ < 1

        end

    end

    @testset "Soss w/ MeasureTheory" begin 

        soss_funnel = Soss.@model (σ) begin
            θ ~ MeasureTheory.Normal(0, σ)
            z ~ MeasureTheory.Normal(0, exp(θ/2)) ^ 512
            x ~ For(z) do zᵢ
                MeasureTheory.Normal(zᵢ, 1)
            end
        end

        @testset "$name" for (name,autodiff) in [
            ("ForwardDiff", MuseInference.ForwardDiffBackend()), 
            # ("Zygote", MuseInference.ZygoteBackend()) # broken
        ]
            (;x) = Soss.predict(copy(rng), soss_funnel(3), θ=0)
            prob = SossMuseProblem(soss_funnel(3) | (;x); autodiff)
            MuseInference.check_self_consistency(prob, (θ=1,), has_volume_factor=false, rng=copy(rng))
            result = muse(prob, (θ=1,); rng=copy(rng), get_covariance=true)
            @test result.dist.μ / result.dist.σ < 1

        end

    end

end