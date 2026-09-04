@testset "Genotype matrix ownership" begin
    source = Float32[0 1 2; 2 1 0; 1 2 1]

    copied = get_genotypes(source;
                           method="BayesC",
                           quality_control=false,
                           center=false,
                           double_precision=false)
    @test copied.genotypes !== source
    @test copied.genotypes == source

    adopted_source = copy(source)
    adopted = get_genotypes(adopted_source;
                            method="BayesC",
                            quality_control=false,
                            center=false,
                            double_precision=false,
                            copy_genotypes=false)
    @test adopted.genotypes === adopted_source

    centered_source = copy(source)
    centered = get_genotypes(centered_source;
                             method="BayesC",
                             quality_control=false,
                             center=true,
                             double_precision=false,
                             copy_genotypes=false)
    @test centered.genotypes === centered_source
    @test centered_source != source
    @test all(abs.(vec(mean(centered_source; dims=1))) .< 10eps(Float32))

    double_source = Float64.(source)
    @test_throws ErrorException get_genotypes(double_source;
                                               method="BayesC",
                                               quality_control=false,
                                               center=false,
                                               double_precision=false,
                                               copy_genotypes=false)

    @test_throws ErrorException get_genotypes(DataFrame(ID=["1", "2"], m1=[0, 1]);
                                               method="BayesC",
                                               quality_control=false,
                                               center=false,
                                               copy_genotypes=false)
end
