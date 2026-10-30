@testset "Genotype matrix ownership" begin
    source = Float32[0 1 2; 2 1 0; 1 2 1]

    copied = get_genotypes(source; copy_genotypes=true,
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
                            double_precision=false)
    @test adopted.genotypes === adopted_source

    centered_source = copy(source)
    centered = get_genotypes(centered_source;
                             method="BayesC",
                             quality_control=false,
                             center=true,
                             double_precision=false)
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

    frame = DataFrame(ID=["1", "2", "3"], m1=[0, 1, 2])
    before_frame = deepcopy(frame)
    from_frame = get_genotypes(frame; quality_control=false, center=false)
    @test from_frame.genotypes == reshape(Float32[0, 1, 2], 3, 1)
    @test frame == before_frame

    from_file = get_genotypes(JWAS.Datasets.dataset("genotypes.txt", dataset_name="demo_7animals"),
                              1.0; quality_control=false, center=false)
    @test from_file.nObs == 7

    double_owned = copy(double_source)
    from_double = get_genotypes(double_owned; quality_control=false, center=false,
                                double_precision=true)
    @test from_double.genotypes === double_owned
    converted = get_genotypes(double_source; quality_control=false, center=false,
                               copy_genotypes=true, double_precision=false)
    @test eltype(converted.genotypes) === Float32
    @test converted.genotypes == Float32.(double_source)
end
