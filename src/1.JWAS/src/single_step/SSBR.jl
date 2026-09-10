"""
(internal) Incomplete Genomic Data (Single-Step)

* reorder in A (pedigree) as ids for genotyped then non-genotyped inds
* impute genotypes for non-genotyped individuals
* add ϵ (imputation errors) and J as variables in data for non-genotyped inds
"""
function SSBRrun(mme,df_whole,train_index,big_memory=false)
    ped      = mme.ped                      #pedigree
    println("calculating A inverse")
    flush(stdout)
    @time Ai_nn,Ai_ng = calc_Ai(ped,mme.M[1],mme)     #get A inverse
    for geno in mme.M
        println("imputing missing genotypes for ", geno.name)
        flush(stdout)
        @time impute_genotypes(geno,ped,mme,Ai_nn,Ai_ng,df_whole[train_index,:],big_memory) #impute genotypes for non-genotyped inds
        println("completed imputing genotypes for ", geno.name)
        if geno.genetic_variance == false
            error("Please input the genetic variance using add_genotypes()")
        end
    end
    #add model terms for SSBR
    add_term(mme,"ϵ") #imputation residual
    if mme.MCMCinfo.fitting_J_vector == true
        add_term(mme,"J") #centering parameter
    end
    #add data for ϵ and J (add columns in input phenotypic data)
    isnongeno = [ID in mme.ped.setNG for ID in df_whole[!,1]] #true/false
    data_ϵ    = String.(df_whole[!,1])
    data_ϵ[.!isnongeno].="missing"
    df_whole[!,Symbol("ϵ")]=data_ϵ

    if mme.MCMCinfo.fitting_J_vector == true
        df_whole[!,Symbol("J")],mme.output_X["J"]=make_JVecs(mme,df_whole,Ai_nn,Ai_ng)
        set_covariate(mme,"J")
    end
    set_random(mme,"ϵ",total_single_step_genetic_variance(mme.M),Vinv=Ai_nn,names=ped.IDs[1:size(Ai_nn,1)])
    #trick to avoid errors (PedModule.getIDs(ped) [nongeno ID;geno ID])
    mme.output_X["ϵ"]=mkmat_incidence_factor(mme.output_ID,ped.IDs)[:,1:size(Ai_nn,1)]

    #add trait name to output_X
    for traiti in mme.lhsVec
        if mme.MCMCinfo.fitting_J_vector == true
            mme.output_X[string(traiti)*":J"] = mme.output_X["J"]
        end
        mme.output_X[string(traiti)*":ϵ"] = mme.output_X["ϵ"]
    end
    if mme.MCMCinfo.fitting_J_vector == true
        delete!(mme.output_X, "J")
        outputMCMCsamples(mme,"J")
    end
    delete!(mme.output_X, "ϵ")
    outputMCMCsamples(mme,"ϵ")
end

function total_single_step_genetic_variance(genotypes)
    total = copy(genotypes[1].genetic_variance.val)
    for geno in genotypes[2:end]
        total = total + geno.genetic_variance.val
    end
    return total
end

############################################################################
# Phenotypes (any order )
############################################################################

############################################################################
# Pedigree
############################################################################
function calc_Ai(ped,geno,mme)
    #reorder Ai to [non-genotyped ids; genotyped ids]
    num_pedn = PedModule.genoSet!(geno.obsID,ped)
    #calculate Ai, Ai_nn, Ai_ng
    Ai       = PedModule.AInverse(ped)
    Ai_nn    = Ai[1:num_pedn,1:num_pedn]
    Ai_ng    = Ai[1:num_pedn,(num_pedn+1):size(Ai,1)]
    return Ai_nn,Ai_ng
end
############################################################################
# Genotypes
############################################################################
function _impute_genotypes_in_chunks(genotypes,Zgeno,lhs,rhs,Zpheno,Zout;
                                     markerperchunk::Integer=1000,
                                     show_progress::Bool=true)
    markerperchunk > 0 || error("markerperchunk must be positive.")
    data_type = eltype(genotypes)
    nmarkers = size(genotypes,2)
    Mpheno = Matrix{data_type}(undef,size(Zpheno,1),nmarkers)
    Mout = Matrix{data_type}(undef,size(Zout,1),nmarkers)
    nchunks = cld(nmarkers,markerperchunk)
    progress = show_progress ? Progress(nchunks; desc="imputing genotypes for non-genotyped individuals") : nothing

    for chunk_start in 1:markerperchunk:nmarkers
        myrange = chunk_start:min(chunk_start+markerperchunk-1,nmarkers)

        # Apply the observed-genotype row permutation to only one marker chunk.
        # The historical implementation formed Zgeno*genotypes for every marker
        # at once, keeping another complete animal-by-marker matrix in memory.
        Mg_chunk = Zgeno*view(genotypes,:,myrange)
        Mped_chunk = lhs\(rhs*Mg_chunk)

        # Sparse pedigree calculations retain their promoted solve precision.
        # Conversion happens once, directly into the final storage precision.
        view(Mpheno,:,myrange) .= Zpheno*Mped_chunk
        view(Mout,:,myrange) .= Zout*Mped_chunk
        show_progress && next!(progress)
    end

    return Mpheno,Mout
end

function impute_genotypes(geno,ped,mme,Ai_nn,Ai_ng,df,big_memory=false)
    data_type = eltype(geno.genotypes) #e.g. Float32
    num_pedn = size(Ai_nn,1)
    #reorder genotypes to get Mg with the same order as Ai_gg
    Zgeno = mkmat_incidence_factor(ped.IDs[(num_pedn+1):end],geno.obsID)
    #**********************************************************
    #impute genotypes for non-genotyped inds
    #**********************************************************
    #two approached are available with similar speed and memory
    #appraoch 1
    #Ai_nn*Mn = -Ai_ng*Mg => [Ai_nn\(-Ai_ng*Mg)
    #                         Mg];
    #
    #appraoch 2
    #[Ai_nn 0 [Mn    [-Ai_ng
    #    0 I]* Mg] =       I]*Mg
    ngeno   = size(Ai_ng,2)
    block12 = spzeros(num_pedn, ngeno)
    block21 = spzeros(ngeno,num_pedn)
    block22 = sparse(I, ngeno, ngeno)
    lhs     = [Ai_nn block12
               block21 block22];
    rhs     = [-Ai_ng
               block22];
    #genotypes are imputed for all individuals in pedigree and aligned to
    #phenotyped individuals by genotype chunks to avoid memory issues.
    Zpheno = mkmat_incidence_factor(df[!,1],ped.IDs)  #aligned to phenotypes
    Zout = mkmat_incidence_factor(mme.output_ID,ped.IDs) #aligned to outputIDs
    if big_memory == false
        Mpheno,Mout = _impute_genotypes_in_chunks(
            geno.genotypes,Zgeno,lhs,rhs,Zpheno,Zout;
            markerperchunk=1000,
        )
    else
         Mg = Zgeno*geno.genotypes
         Mpheno = data_type.(Zpheno*(lhs\(rhs*Mg)))
         Mout   = data_type.(Zout*(lhs\(rhs*Mg)))
    end

    # Adopt the final matrices directly. Replacing geno.genotypes releases the
    # object's reference to the observed matrix after chunked imputation.
    geno.output_genotypes = Mout
    geno.genotypes        = Mpheno
    geno.obsID            = df[!,1]
    geno.nObs             = length(geno.obsID)
    GC.gc()
end
############################################################################
# Fixed effects (J)
############################################################################
function make_JVecs(mme,df_whole,Ai_nn,Ai_ng)
    Jg = -ones(size(Ai_ng,2))
    Jn = Ai_nn\(-Ai_ng*Jg)
    J  = [Jn;
          Jg]
    Z  = mkmat_incidence_factor(df_whole[!,1],mme.ped.IDs)
    if mme.output_ID != 0
        Zo  = mkmat_incidence_factor(mme.output_ID,mme.ped.IDs)
        ZoJ = Zo*J
    else
        ZoJ=0
    end
    return Z*J,ZoJ
end

"""
add to model an extra term: imputation_residual
"""
function add_term(mme,term::AbstractString)
    for m in 1:mme.nModels
        modelterm = ModelTerm(term,m,string(mme.lhsVec[m]))
        push!(mme.modelTerms,modelterm)
        mme.modelTermDict[modelterm.trmStr]=modelterm
    end
end
