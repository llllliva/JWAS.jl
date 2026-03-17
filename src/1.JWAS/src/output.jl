# ===============================================================================================================
# This script handles the output of analysis, including results (posterior mean and variances) and MCMC samples.
# Features:
# - contains public functions
# - returns point estimates and standard deveiations (as a dictionary) from runMCMC function
# - outputs MCMC samples in text file format
# ===============================================================================================================

# =============================================================================
#                                KEY TERMS & NOTATIONS
# =============================================================================
# 
# PEV: Prediction Error Variance of a random effect, PEV=var(u-u^hat)=var(u|y), 
#      i.e., the variance of the posterior distribution of u. 
#
#

# ===============================================================================================================
#                                PUBLIC FUNCTIONS
# ===============================================================================================================
"""
    prediction_setup(mme::MME)

* (internal function) Create incidence matrices for individuals of interest based on a usere-defined
prediction equation, defaulting to genetic values including effects defined with genomic and pedigre
information. For now, genomic data is always included.
* J and ϵ are always included in single-step analysis (added in SSBR.jl)
"""
function prediction_setup(model)
    if model.MCMCinfo.prediction_equation == false
        prediction_equation = []
        if model.pedTrmVec != 0
            for i in model.pedTrmVec
                push!(prediction_equation,i)
            end
        end
    else
        prediction_equation = string.(strip.(split(model.MCMCinfo.prediction_equation,"+")))
        if model.MCMCinfo.output_heritability != false
            printstyled("User-defined prediction equation is provided. ","The heritability is the ",
            "proportion of phenotypic variance explained by the value defined by the prediction equation.\n",
            bold=false,color=:green)
        end
        for i in prediction_equation
            term_symbol = Symbol(split(i,":")[end])
            if !(haskey(model.modelTermDict,i) || (isdefined(Main,term_symbol) && typeof(getfield(Main,term_symbol)) == Genotypes))
                error("Terms $i in the prediction equation is not found.")
            end
        end
    end
    printstyled("Predicted values for individuals of interest will be obtained as the summation of ",
    prediction_equation, " (Note that genomic data is always included for now).",bold=false,color=:green)
    if length(prediction_equation) == 0 && model.M == false
        println("Default or user-defined prediction equation are not available.")
        model.MCMCinfo.outputEBV = false
    end
    filter!(e->(e in keys(model.modelTermDict)),prediction_equation) #remove "genotypes" for now
    model.MCMCinfo.prediction_equation = prediction_equation
end

"""
    outputEBV(model,IDs::Array)

Output estimated breeding values and prediction error variances for IDs.
"""
function outputEBV(model,IDs)
    IDs = map(string,vec(IDs))
    model.output_ID=IDs
end

"""
    outputMCMCsamples(mme::MME,trmStrs::AbstractString...)

Get MCMC samples for specific location parameters.
"""
function outputMCMCsamples(mme::MME,trmStrs::AbstractString...)
    for trmStr in trmStrs
      res = []
      #add trait name to variables,e.g, age => "y1:age"
      #"age" may be in trait 1 but not trait 2
      for (m,model) = enumerate(mme.modelVec)
          strVec  = split(model,['=','+'])
          strpVec = [strip(i) for i in strVec]
          if trmStr in strpVec || trmStr in ["J","ϵ"]
              res = [res;string(mme.lhsVec[m])*":"*trmStr]
          end
      end
      for trmStr in res
          trm     = mme.modelTermDict[trmStr]
          push!(mme.outputSamplesVec,trm)
      end
    end
end

mutable struct StreamEBVSampleOutput
    bin_io::IOStream
    ids_path::String
    meta_path::String
    trait::String
    ids::Vector{String}
    nObs::Int
    nSamples::Int
    mean::Vector{Float64}
    mean2::Vector{Float64}
    buffer32::Vector{Float32}
end

function _stream_ebv_store(outfile)
    return get(outfile, "__stream_ebv__", Dict{String,StreamEBVSampleOutput}())
end

function _is_stream_output_mode(mme)
    return mme.M != 0 && any(Mi->Mi.storage_mode == :stream, mme.M)
end

function _stream_ebv_prefix(output_folder::AbstractString, trait_name::AbstractString)
    return joinpath(output_folder, "MCMC_samples_EBV_" * trait_name)
end

function _create_stream_ebv_sample_output(output_folder::AbstractString,
                                          trait_name::AbstractString,
                                          ids::AbstractVector{<:AbstractString})
    prefix = _stream_ebv_prefix(output_folder, trait_name)
    bin_path = prefix * ".bin"
    ids_path = prefix * ".ids.txt"
    meta_path = prefix * ".meta"

    _write_string_lines(ids_path, map(String, vec(ids)))
    bin_io = open(bin_path, "w")

    printstyled("The file $(bin_path) is created to save binary MCMC EBV samples for $(trait_name).\n",
                bold=false,color=:green)

    return StreamEBVSampleOutput(
        bin_io,
        ids_path,
        meta_path,
        String(trait_name),
        map(String, vec(ids)),
        length(ids),
        0,
        zeros(Float64, length(ids)),
        zeros(Float64, length(ids)),
        zeros(Float32, length(ids))
    )
end

function _write_stream_ebv_meta!(output::StreamEBVSampleOutput)
    entries = [
        "trait" => output.trait,
        "nSamples" => string(output.nSamples),
        "nObs" => string(output.nObs),
        "dtype" => "Float32"
    ]
    _write_streaming_manifest(output.meta_path, entries)
    return nothing
end

function _write_stream_ebv_sample!(output::StreamEBVSampleOutput, values::AbstractVector)
    length(values) == output.nObs || error("Stream EBV sample length must match nObs.")

    output.nSamples += 1
    nsamples = output.nSamples
    @inbounds for i in 1:output.nObs
        sample_i = Float64(values[i])
        output.mean[i] += (sample_i - output.mean[i]) / nsamples
        output.mean2[i] += (sample_i * sample_i - output.mean2[i]) / nsamples
        output.buffer32[i] = Float32(sample_i)
    end
    write(output.bin_io, output.buffer32)
    return nothing
end

function _stream_ebv_summary(output::StreamEBVSampleOutput)
    if output.nSamples <= 1
        pev = fill(NaN, output.nObs)
    else
        scale = output.nSamples / (output.nSamples - 1)
        pev = max.(0.0, scale .* (output.mean2 .- output.mean .^ 2))
    end
    return (IDs=copy(output.ids), EBV=copy(output.mean), PEV=pev, nSamples=output.nSamples)
end

function _finalize_mcmc_sample_outputs!(outfile)
    stream_summaries = Dict{String,NamedTuple}()

    for (key, value) in outfile
        if key == "__stream_ebv__"
            for (trait_key, stream_output) in value
                close(stream_output.bin_io)
                _write_stream_ebv_meta!(stream_output)
                stream_summaries[trait_key] = _stream_ebv_summary(stream_output)
            end
        elseif value isa IOStream
            close(value)
        end
    end

    return isempty(stream_summaries) ? false : stream_summaries
end

"""
    readEBVsamples(output_folder, trait; mmap=true)

Read posterior EBV samples from either dense text files or stream binary files.
Returns `(IDs, samples)` where `samples` is `nSamples x nObs`.
"""
function readEBVsamples(output_folder, trait; mmap::Bool=true)
    trait_name = string(trait)
    prefix = _stream_ebv_prefix(output_folder, trait_name)
    bin_path = prefix * ".bin"
    ids_path = prefix * ".ids.txt"
    meta_path = prefix * ".meta"

    if isfile(bin_path)
        meta = _read_streaming_manifest(meta_path)
        nSamples = parse(Int, meta["nSamples"])
        nObs = parse(Int, meta["nObs"])
        dtype = meta["dtype"]
        dtype == "Float32" || error("Unsupported EBV sample dtype '$dtype'.")

        ids = _read_string_lines(ids_path)
        length(ids) == nObs || error("EBV sample metadata/ID count mismatch for $ids_path.")

        sample_count = nSamples * nObs
        if mmap
            io = open(bin_path, "r")
            raw = Mmap.mmap(io, Vector{Float32}, sample_count)
            close(io)
            return ids, transpose(reshape(raw, nObs, nSamples))
        end

        raw = Vector{Float32}(undef, sample_count)
        open(bin_path, "r") do io
            read!(io, raw)
        end
        return ids, Matrix(transpose(reshape(raw, nObs, nSamples)))
    end

    txt_path = prefix * ".txt"
    if isfile(txt_path)
        samples, ids = readdlm(txt_path, ',', header=true)
        return map(string, vec(ids)), map(Float64, samples)
    end

    error("EBV sample files are not found for trait $(trait_name) in $(output_folder).")
end


# ===============================================================================================================
#                                PRIVATE FUNCTIONS
# ===============================================================================================================

################################################################################
#*******************************************************************************
#2. Return Output Results (Dictionary)
#*******************************************************************************
#Posterior means and variances are calculated for all parameters in the model
#when MCMC is running; Other paramters (e.g., EBV), which is a function of those
#are calculated from files storing MCMC samples at the end of MCMC.
################################################################################
function output_result(mme,output_folder,
                       solMean,meanVare,G0Mean,
                       solMean2 = missing,meanVare2 = missing,G0Mean2 = missing;
                       stream_ebv_summaries = false)
  output = Dict()
  location_parameters = reformat2dataframe([getNames(mme) solMean sqrt.(abs.(solMean2 .- solMean .^2))])
  output["location parameters"] = location_parameters
  if mme.MCMCinfo.RRM == false
      output["residual variance"]   = matrix2dataframe(string.(mme.lhsVec),meanVare,meanVare2)
  else
      output["residual variance"]   = matrix2dataframe(["1"],meanVare,meanVare2)
  end

  if mme.pedTrmVec != 0
    output["polygenic effects covariance matrix"]=matrix2dataframe(mme.pedTrmVec,G0Mean,G0Mean2) 
  end

  ntraits = length(mme.lhsVec)

  if mme.M != 0
      for Mi in mme.M
         ntraits_geno = mme.MCMCinfo.RRM == false ? Mi.ntraits : length(mme.lhsVec)
         traiti      = 1
         whichtrait  = fill(string(mme.lhsVec[traiti]),length(Mi.markerID))
         whichmarker = Mi.markerID
         whicheffect = Mi.meanAlpha[traiti]
         whicheffectsd = sqrt.(abs.(Mi.meanAlpha2[traiti] .- Mi.meanAlpha[traiti] .^2))
         whichdelta    = Mi.meanDelta[traiti]
          for traiti in 2:ntraits_geno
                whichtrait     = vcat(whichtrait,fill(string(mme.lhsVec[traiti]),length(Mi.markerID)))
                whichmarker    = vcat(whichmarker,Mi.markerID)
                whicheffect    = vcat(whicheffect,Mi.meanAlpha[traiti])
                whicheffectsd  = vcat(whicheffectsd,sqrt.(abs.(Mi.meanAlpha2[traiti] .- Mi.meanAlpha[traiti] .^2)))
                whichdelta     = vcat(whichdelta,Mi.meanDelta[traiti])
            end

          output["marker effects "*Mi.name]=DataFrame([whichtrait whichmarker whicheffect whicheffectsd whichdelta],[:Trait,:Marker_ID,:Estimate,:SD,:Model_Frequency])
          #output["marker effects variance "*Mi.name] = matrix2dataframe(string.(mme.lhsVec),Mi.meanVara,Mi.meanVara2)
          if Mi.estimatePi == true
              output["pi_"*Mi.name] = pi2dataframe(Mi, Mi.mean_pi, Mi.mean_pi2)
          end
          if Mi.G.estimate_scale == true
              output["ScaleEffectVar"*Mi.name] = matrix2dataframe(string.(mme.lhsVec),Mi.meanScaleVara,Mi.meanScaleVara2)
          end
          if Mi.annotations !== false
              ann = Mi.annotations
              annotation_names = ["Intercept"; ["Annotation_$i" for i in 1:(size(ann.design_matrix, 2)-1)]]
              annotation_sd = sqrt.(abs.(ann.mean_coefficients2 .- ann.mean_coefficients .^ 2))
              output["annotation coefficients "*Mi.name] = DataFrame(
                  Annotation=annotation_names,
                  Estimate=ann.mean_coefficients,
                  SD=annotation_sd,
              )
          end
      end
  end

  #Get EBV and PEV from MCMC samples text files
  if mme.output_ID != 0 && mme.MCMCinfo.outputEBV == true
      output_file = output_folder*"/MCMC_samples"
      EBVkeys = ["EBV"*"_"*string(mme.lhsVec[traiti]) for traiti in 1:ntraits]
      for EBVkey in EBVkeys
          if stream_ebv_summaries !== false && haskey(stream_ebv_summaries, EBVkey)
              summary = stream_ebv_summaries[EBVkey]
              IDs = summary.IDs
              EBV = summary.EBV
              PEV = summary.PEV
          else
              IDs, EBVsamples = readEBVsamples(output_folder, replace(EBVkey, "EBV_" => ""); mmap=false)
              EBV = vec(mean(EBVsamples,dims=1))
              PEV = vec(var(EBVsamples,dims=1))
          end
          if vec(IDs) == mme.output_ID
              output[EBVkey] = DataFrame([mme.output_ID EBV PEV],[:ID,:EBV,:PEV])
          else
              error("The EBV file is wrong.")
          end
      end
      if mme.MCMCinfo.output_heritability == true  && mme.MCMCinfo.single_step_analysis == false
          if mme.MCMCinfo.RRM != false
              genetic_trm = ["genetic_variance"]
          else
              genetic_trm = ["genetic_variance","heritability"]
          end
          for i in genetic_trm
              samplesfile = output_file*"_"*i*".txt"
              samples,names = readdlm(samplesfile,',',header=true)
              samplemean    = vec(mean(samples,dims=1))
              samplevar     = vec(std(samples,dims=1))
              output[i] = DataFrame([vec(names) samplemean samplevar],[:Covariance,:Estimate,:SD])
          end
      end
  end
  return output
end

#Reformat Output Array to DataFrame
function reformat2dataframe(res::Array)
    out_names = Array{String}(undef,size(res,1),3)
    for rowi in 1:size(res,1)
        out_names[rowi,:]=[strip(i) for i in split(res[rowi,1],':',keepempty=false)]
    end

    if size(out_names,2)==1 #convert vector to matrix
        out_names = reshape(out_names,length(out_names),1)
    end
    #out_names=permutedims(out_names,[2,1]) #rotate
    out_values   = map(Float64,res[:,2])
    out_variance = convert.(Union{Missing, Float64},res[:,3])
    out =[out_names out_values out_variance]
    out = DataFrame(out, [:Trait, :Effect, :Level, :Estimate,:SD])
    return out
end

#convert a scalar (single-trait), a matrix (multi-trait), a vector (mega-trait) to a DataFrame
function matrix2dataframe(names,meanVare,meanVare2) #also works for scalar
    if !(typeof(meanVare) <: Vector)
        names = repeat(names,inner=length(names)).*"_".*repeat(names,outer=length(names))
    end
    meanVare  = (typeof(meanVare)  <: Union{Number,Missing,Vector}) ? meanVare  : vec(meanVare)
    meanVare2 = (typeof(meanVare2) <: Union{Number,Missing,Vector}) ? meanVare2 : vec(meanVare2)
    stdVare   = sqrt.(abs.(meanVare2 .- meanVare .^2))
    DataFrame([names meanVare stdVare],[:Covariance,:Estimate,:SD])
end

function dict2dataframe(mean_pi,mean_pi2)
    if typeof(mean_pi) <: Union{Number,Missing}
        names = "π"
    else
        names = collect(keys(mean_pi))
    end
    mean_pi  = (typeof(mean_pi) <: Union{Number,Missing}) ? mean_pi : collect(values(mean_pi))
    mean_pi2 = (typeof(mean_pi2) <: Union{Number,Missing}) ? mean_pi2 : collect(values(mean_pi2))
    stdpi    = sqrt.(abs.(mean_pi2 .- mean_pi .^2))
    DataFrame([names mean_pi stdpi],[:π,:Estimate,:SD])
end

function collapse_pi_for_output(Mi, pi_value)
    if Mi.annotations === false && Mi.ntraits == 1 && pi_value isa AbstractVector
        return pi_value[1]
    end
    return pi_value
end

function pi2dataframe(Mi, mean_pi, mean_pi2)
    mean_pi_out = collapse_pi_for_output(Mi, mean_pi)
    mean_pi2_out = collapse_pi_for_output(Mi, mean_pi2)
    return dict2dataframe(mean_pi_out, mean_pi2_out)
end

"""
    getEBV(model::MME,traiti)

(internal function) Get breeding values for individuals defined by outputEBV(),
defaulting to all genotyped individuals. This function is used inside MCMC functions for
one MCMC samples from posterior distributions.
e.g.,
non-NNBayes_partial (multi-classs Bayes) : y1=M1*α1[1]+M2*α2[1]+M3*α3[1]
                                           y2=M1*α1[2]+M2*α2[2]+M3*α3[2];
NNBayes_partial:     y1=M1*α1[1]
                     y2=M2*α2[1]
                     y3=M3*α3[1];
"""
function getEBV(mme,traiti)
    traiti_name = string(mme.lhsVec[traiti])
    EBV=zeros(length(mme.output_ID))

    location_parameters = reformat2dataframe([getNames(mme) mme.sol zero(mme.sol)])
    for term in keys(mme.output_X)
        mytrait, effect = split(term,':')
        if mytrait == traiti_name
            sol_term     = map(Float64,location_parameters[(location_parameters[!,:Effect].==effect).&(location_parameters[!,:Trait].==traiti_name),:Estimate])
            if length(sol_term) == 1 #1-element Array{Float64,1} doesn't work below; Will be deleted
                sol_term = sol_term[1]
            end
            EBV_term = mme.output_X[term]*sol_term
            if length(sol_term) == 1 #1-element Array{Float64,1} doesn't work below; Will be deleted
                EBV_term = vec(EBV_term)
            end
            EBV += EBV_term
        end
    end
    if mme.M != 0
        for Mi in mme.M
            if Mi.storage_mode == :stream
                Mi.output_stream_backend == false && error("Stream output backend is not initialized.")
                tmp = Vector{Float64}(undef, Mi.output_stream_backend.nObs)
                streaming_mul_alpha!(tmp, Mi.output_stream_backend, Mi.α[traiti])
                EBV += tmp
            else
                EBV += Mi.output_genotypes*Mi.α[traiti]
            end
        end
    end
    return EBV
end
################################################################################
#*******************************************************************************
#3. Save MCMC Samples to Text Files
#*******************************************************************************
#MCMC samples for all hyperparameters, all marker effects, and
#location parameters defined by outputMCMCsamples() are saved
#into files every output_samples_frequency iterations.
################################################################################
"""
    output_MCMC_samples_setup(mme,nIter,output_samples_frequency,file_name="MCMC_samples")

(internal function) Set up text files to save MCMC samples.
"""
function output_MCMC_samples_setup(mme,nIter,output_samples_frequency,file_name="MCMC_samples")
  ntraits = size(mme.lhsVec,1)
  stream_ebv_mode = _is_stream_output_mode(mme)

  outfile = Dict{String,Any}()
  outfile["__stream_ebv__"] = Dict{String,StreamEBVSampleOutput}()

  outvar = ["residual_variance"]
  if mme.pedTrmVec != 0
      push!(outvar,"polygenic_effects_variance")
  end

  if mme.M != 0 && mme.MCMCinfo.output_marker_parameter_samples
      for Mi in mme.M
          geno_names = mme.MCMCinfo.RRM == false ? Mi.trait_names : string.(mme.lhsVec)
          for traiti in geno_names
              push!(outvar,"marker_effects_"*Mi.name*"_"*traiti)
          end
          push!(outvar,"marker_effects_variances"*"_"*Mi.name)
          push!(outvar,"pi"*"_"*Mi.name)
      end
  end

  for trmi in mme.outputSamplesVec
      push!(outvar, trmi.trmStr)
  end

  for effect in mme.rndTrmVec
      trmStri = join(effect.term_array, "_")
      push!(outvar, trmStri*"_variances")
  end

  if mme.MCMCinfo.outputEBV == true
      for traiti in 1:ntraits
          key = "EBV_"*string(mme.lhsVec[traiti])
          if stream_ebv_mode
              outfile["__stream_ebv__"][key] = _create_stream_ebv_sample_output(dirname(file_name),
                                                                                 string(mme.lhsVec[traiti]),
                                                                                 mme.output_ID)
          else
              push!(outvar, key)
          end
      end
      if mme.MCMCinfo.output_heritability == true && mme.MCMCinfo.single_step_analysis == false
          push!(outvar,"genetic_variance")
          if mme.MCMCinfo.RRM == false
              push!(outvar,"heritability")
          else
              printstyled("heritability is not computed for Random Regression Model. \n",bold=false,color=:green)
          end
      end
  end

  for t in 1:mme.nModels
      if mme.traits_type[t] ∈ ["categorical","categorical(binary)","censored"]
          push!(outvar,"liabilities_"*string(mme.lhsVec[t]))
          if mme.traits_type[t] ∈ ["categorical","categorical(binary)"]
              push!(outvar,"threshold_"*string(mme.lhsVec[t]))
          end
      end
  end

  for i in outvar
      file_i = file_name*"_"*replace(i,":"=>".")*".txt"
      if isfile(file_i)
        printstyled("The file "*file_i*" already exists!!! It is overwritten by the new output.\n",bold=false,color=:red)
      else
        printstyled("The file "*file_i*" is created to save MCMC samples for "*i*".\n",bold=false,color=:green)
      end
      outfile[i] = open(file_i,"w")
  end

  mytraits = map(string,mme.lhsVec)
  if mme.R.constraint == false
      varheader = repeat(mytraits,inner=length(mytraits)).*"_".*repeat(mytraits,outer=length(mytraits))
  else
      varheader = transubstrarr(map(string,mme.lhsVec))
  end
  writedlm(outfile["residual_variance"],transubstrarr(varheader),',')

  for trmi in mme.outputSamplesVec
      writedlm(outfile[trmi.trmStr],transubstrarr(getNames(trmi)),',')
  end

  for effect in mme.rndTrmVec
    trmStri = join(effect.term_array, "_")
    thisheader = repeat(effect.term_array,inner=length(effect.term_array)).*"_".*repeat(effect.term_array,outer=length(effect.term_array))
    writedlm(outfile[trmStri*"_variances"],transubstrarr(thisheader),',')
  end

  if mme.M != 0 && mme.MCMCinfo.output_marker_parameter_samples
      for Mi in mme.M
          geno_names = mme.MCMCinfo.RRM == false ? Mi.trait_names : string.(mme.lhsVec)
          for traiti in geno_names
              writedlm(outfile["marker_effects_"*Mi.name*"_"*traiti],transubstrarr(Mi.markerID),',')
          end
      end
  end

  if mme.pedTrmVec != 0
      pedtrmvec = mme.pedTrmVec
      thisheader = repeat(pedtrmvec,inner=length(pedtrmvec)).*"_".*repeat(pedtrmvec,outer=length(pedtrmvec))
      writedlm(outfile["polygenic_effects_variance"],transubstrarr(thisheader),',')
  end

  if mme.MCMCinfo.outputEBV == true
      if !stream_ebv_mode
          for traiti in 1:ntraits
              writedlm(outfile["EBV_"*string(mme.lhsVec[traiti])],transubstrarr(mme.output_ID),',')
          end
      end
      if mme.MCMCinfo.output_heritability == true && mme.MCMCinfo.single_step_analysis == false
          varheader = repeat(mytraits,inner=length(mytraits)).*"_".*repeat(mytraits,outer=length(mytraits))
          writedlm(outfile["genetic_variance"],transubstrarr(varheader),',')
          if mme.MCMCinfo.RRM == false
              writedlm(outfile["heritability"],transubstrarr(map(string,mme.lhsVec)),',')
          end
      end
  end

  return outfile
end
"""
    output_MCMC_samples(mme,vRes,G0,outfile=false)

(internal function) Save MCMC samples every output_samples_frequency iterations to the text file.
"""
function output_MCMC_samples(mme,vRes,G0,
                             outfile=false)
    ntraits     = size(mme.lhsVec,1)
    stream_ebv_outputs = outfile == false ? Dict{String,StreamEBVSampleOutput}() : _stream_ebv_store(outfile)
    #location parameters
    output_location_parameters_samples(mme,mme.sol,outfile)
    #random effects variances
    for effect in  mme.rndTrmVec
        trmStri   = join(effect.term_array, "_")
        writedlm(outfile[trmStri*"_variances"],vec(inv(effect.Gi.val))',',')
    end

    if mme.R.constraint == true
        vRes=diag(vRes)
    end
    writedlm(outfile["residual_variance"],(typeof(vRes) <: Number) ? vRes : vec(vRes)' ,',')

    if mme.pedTrmVec != 0
        writedlm(outfile["polygenic_effects_variance"],vec(G0)',',')
    end
    if mme.M != 0 && outfile != false && mme.MCMCinfo.output_marker_parameter_samples
      for Mi in mme.M
         ntraits_geno = mme.MCMCinfo.RRM == false ? Mi.ntraits : length(mme.lhsVec)
         geno_names = mme.MCMCinfo.RRM == false ? Mi.trait_names : string.(mme.lhsVec)
         for traiti in 1:ntraits_geno
            writedlm(outfile["marker_effects_"*Mi.name*"_"*geno_names[traiti]],Mi.α[traiti]',',')
         end
          
         if Mi.G.val != false
              if mme.nModels == 1
                  writedlm(outfile["marker_effects_variances"*"_"*Mi.name],Mi.G.val',',')
              else
                  if Mi.method == "BayesB"
                      writedlm(outfile["marker_effects_variances"*"_"*Mi.name],hcat([x for x in Mi.G.val]...),',')
                  else
                      writedlm(outfile["marker_effects_variances"*"_"*Mi.name],Mi.G.val,',')
                  end
              end
          end
          pi_output = collapse_pi_for_output(Mi, Mi.π)
          writedlm(outfile["pi"*"_"*Mi.name],pi_output,',')
          if !(typeof(pi_output) <: Number) #add a blank line
              println(outfile["pi"*"_"*Mi.name])
          end
      end
    end

    if mme.MCMCinfo.outputEBV == true
         EBVmat = myEBV = getEBV(mme,1)
         trait_key = "EBV_"*string(mme.lhsVec[1])
         if haskey(stream_ebv_outputs, trait_key)
             _write_stream_ebv_sample!(stream_ebv_outputs[trait_key], myEBV)
         else
             writedlm(outfile[trait_key],myEBV',',')
         end
         for traiti in 2:ntraits
             myEBV = getEBV(mme,traiti) #actually BV
             trait_key = "EBV_"*string(mme.lhsVec[traiti])
             if haskey(stream_ebv_outputs, trait_key)
                 _write_stream_ebv_sample!(stream_ebv_outputs[trait_key], myEBV)
             else
                 writedlm(outfile[trait_key],myEBV',',')
             end
             EBVmat = [EBVmat myEBV]
         end

         if mme.MCMCinfo.output_heritability == true && mme.MCMCinfo.single_step_analysis == false
             #single-trait: a scalar ;  multi-trait: a matrix; mega-trait: a vector
             if mme.M != 0 && mme.M[1].G.constraint==true
                mygvar = Diagonal(vec(var(EBVmat,dims=1)))
             else
                mygvar = cov(EBVmat)
             end
             genetic_variance = (ntraits == 1 ? mygvar : vec(mygvar)')
             if mme.MCMCinfo.RRM == false
                 vRes = mme.R.constraint==true ? Diagonal(vRes) : vRes #change to diagonal matrix to avoid error
                 heritability = (ntraits == 1 ? mygvar/(mygvar+vRes) : (diag(mygvar)./(diag(mygvar)+diag(vRes)))')
                 writedlm(outfile["heritability"],heritability,',')
             end
             writedlm(outfile["genetic_variance"],genetic_variance,',')
         end
    end
    #categorical/binary/censored traits
    if !isempty(intersect(mme.traits_type, ["categorical","categorical(binary)","censored"]))
        ySparse = reshape(mme.ySparse,:,ntraits) #liability (=mme.ySparse)
        for t in 1:mme.nModels
            if mme.traits_type[t] ∈ ["categorical","categorical(binary)","censored"] #save liability
                writedlm(outfile["liabilities_"*string(mme.lhsVec[t])], ySparse[:,t]', ',')
                if mme.traits_type[t] ∈ ["categorical","categorical(binary)"] #save thresholds
                    writedlm(outfile["threshold_"*string(mme.lhsVec[t])], mme.thresholds[t]', ',')
                end
            end
        end
    end
end
"""
    output_location_parameters_samples(mme::MME,sol,outfile)

(internal function) Save MCMC samples for location parameers
"""
function output_location_parameters_samples(mme::MME,sol,outfile)
    for trmi in  mme.outputSamplesVec
        trmStr = trmi.trmStr
        startPosi  = trmi.startPos
        endPosi    = startPosi + trmi.nLevels - 1
        samples4locations = sol[startPosi:endPosi]
        writedlm(outfile[trmStr],samples4locations',',')
    end
end
"""
    transubstrarr(vec)

(internal function) Transpose a column vector of strings (vec' doesn't work here)
"""
function transubstrarr(vec)
    lvec=length(vec)
    res =Array{String}(undef,1,lvec)
    for i in 1:lvec
        res[1,i]=vec[i]
    end
    return res
end

#output mean and variance of posterior distribution of parameters of interest
function output_posterior_mean_variance(mme,nsamples)
    mme.solMean   += (mme.sol - mme.solMean)/nsamples
    mme.solMean2  += (mme.sol .^2 - mme.solMean2)/nsamples
    mme.meanVare  += (mme.R.val - mme.meanVare)/nsamples
    mme.meanVare2 += (mme.R.val .^2 - mme.meanVare2)/nsamples

    if mme.pedTrmVec != 0
        polygenic_pos = findfirst(i -> i.randomType=="A", mme.rndTrmVec)
        mme.G0Mean  += (inv(mme.rndTrmVec[polygenic_pos].Gi.val)  - mme.G0Mean )/nsamples
        mme.G0Mean2 += (inv(mme.rndTrmVec[polygenic_pos].Gi.val) .^2  - mme.G0Mean2 )/nsamples
    end
    if mme.M != 0
        for Mi in mme.M
            for trait in 1:Mi.ntraits
                Mi.meanAlpha[trait] += (Mi.α[trait] - Mi.meanAlpha[trait])/nsamples
                Mi.meanAlpha2[trait]+= (Mi.α[trait].^2 - Mi.meanAlpha2[trait])/nsamples
                Mi.meanDelta[trait] += (Mi.δ[trait] - Mi.meanDelta[trait])/nsamples
            end
            if Mi.estimatePi == true
                if Mi.ntraits == 1 || mme.M[1].G.constraint==true #may need to change for multiple M
                    Mi.mean_pi += (Mi.π-Mi.mean_pi)/nsamples
                    Mi.mean_pi2 += (Mi.π .^2-Mi.mean_pi2)/nsamples
                else
                    for i in keys(Mi.π)
                      Mi.mean_pi[i] += (Mi.π[i]-Mi.mean_pi[i])/nsamples
                      Mi.mean_pi2[i] += (Mi.π[i].^2-Mi.mean_pi2[i])/nsamples
                    end
                end
            end
            if Mi.method != "BayesB"
                Mi.meanVara += (Mi.G.val - Mi.meanVara)/nsamples
                Mi.meanVara2 += (Mi.G.val .^2 - Mi.meanVara2)/nsamples
            end
            if Mi.G.estimate_scale == true
                Mi.meanScaleVara += (Mi.G.scale - Mi.meanScaleVara)/nsamples
                Mi.meanScaleVara2 += (Mi.G.scale .^2 - Mi.meanScaleVara2)/nsamples
            end
            if Mi.annotations !== false
                ann = Mi.annotations
                ann.mean_coefficients += (ann.coefficients - ann.mean_coefficients) / nsamples
                ann.mean_coefficients2 += (ann.coefficients .^ 2 - ann.mean_coefficients2) / nsamples
            end
        end
    end
end
