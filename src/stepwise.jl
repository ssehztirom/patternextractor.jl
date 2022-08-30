 function annotatesamplesbypattern(traindata,samples,features,originallabels,fillupempty=true)
	trainpatterns = traindata[:,features]
	samplepatterns = samples
	if size(samples,2)> length(features)
		samplepatterns = samples[:,features]
	end
	codes = [join(Int.(trainpatterns[i,:])) for i = 1:size(trainpatterns,1) ]
	codesmap = Dict()
	for i = 1:length(codes)
		codesmap[codes[i]] = originallabels[i]
	end
	samplelabel = fill("XYZ",size(samplepatterns)[1])
	for i = 1:size(samplepatterns)[1]
		newlabel = trykey(join(Int.(samplepatterns[i,:])),codesmap,"")
		samplelabel[i] = newlabel
	end
	if fillupempty
		samplelabel[samplelabel.==""] = [rand(unique(originallabels)) for i = 1:sum(samplelabel.=="")]
	end
	return samplelabel
end

 function trykey(x,dict,returnval = 0)
    if haskey(dict,x)
        return dict[x]
    else
        return returnval
    end
end




function evaluatevariable(curset,i,visible_training,hidden_training,visible_test,hidden_test,evolution=true)
	h = size(hidden_training.data)[2]
	scores = zeros(h)
	curx_train = initcurx(curset,visible_training)
	curx_test = nothing
	m0constr = ifelse(length(curset)==0,makeconstraint(curx_train),makeconstraint(curx_train,evolution))
	updatecurx!(curx_train,visible_training,length(curset)+2,i)
	if !isnothing(visible_test)
		curx_test = initcurx(curset,visible_test)
		updatecurx!(curx_test,visible_test,length(curset)+2,i)
	end
	for j=1:h
		updatecurx!(curx_train,hidden_training,1,j)
		if !isnothing(hidden_test)
			updatecurx!(curx_test,hidden_test,1,j)
		end
		scores[j] = gof(curx_train,m0constr,curx_test)
	end
	return scores
end

function extract_pattern(kvariables,xs,zs)
	return select_k_variables(kvariables,[converttoleveldata([xsbin,zsbin])])
end

function select_k_variables(kvariables,sampleset;testtraining=false,newdata=nothing,verbose=true,evolution=true)
	if length(sampleset)==0
		println("sample set is empty")
		error()
	end
	n,p =size(sampleset[1][1].data)
	curset = Int[]
	scores = []
	requiredsets = ifelse(isnothing(newdata),1,ifelse(newdata=="i",kvariables * p,kvariables)) * ifelse(testtraining,2,1)
	if requiredsets > length(sampleset)
		println("number of data sets does not match the requirements")
		error()
	end
	visible_training,hidden_training =pop!(sampleset)
	visible_test,hidden_test = nothing,nothing
	if 	testtraining
		visible_test,hidden_test =pop!(sampleset)
	end
	for k=1:kvariables
		curscores = zeros(p,size(hidden_training.data)[2])
		if (k > 1) & (!isnothing(newdata))
			println("new set for k")
			visible_training,hidden_training = pop!(sampleset)
			if 	testtraining
				visible_test,hidden_test =pop!(sampleset)
			end
		end
		
		for i=1:p
			if (i in curset) 
            	continue
        	end 
			if (i > 1) & (newdata=="i")
				println("new set for i")
				visible_training,hidden_training =pop!(sampleset)
				if testtraining 
					visible_test,hidden_test =pop!(sampleset)
				end
			end
			# println(i)
			curscores[i,:] = evaluatevariable(curset,i,visible_training,hidden_training,visible_test,hidden_test,evolution)
		end
		push!(scores,curscores)
		# println(findmax(curscores))
		push!(curset,findmax(curscores)[2][1])
	end
	return [findmax(i)[1] for i = scores],curset,scores
end

function initcurx(startset,visible)
	varno = length(startset)+2 # number of variables to be modeled
	lastvisiblepos = varno-1
	visiblerange = 2:lastvisiblepos
    curx = LogLinearModels.LevelData(Matrix{Float64}(undef,size(visible.data,1),varno)) 
    curx.data[:,visiblerange] .= visible.data[:,startset] 
	curx.levelno[visiblerange] .= visible.levelno[startset]
	return curx
end

function converttoleveldata(samples)
	return LogLinearModels.LevelData(samples[1]),LogLinearModels.LevelData(samples[2])
end

function updatecurx!(curx,visible,curxpos,xpos)
	curx.data[:,curxpos] .= visible.data[:,xpos]
	curx.levelno[curxpos] = visible.levelno[xpos]
end

function gof(curx_train,m0constr,curx_test)
	freqval = LogLinearModels.freqtab(curx_train,fillzeros=true)
	freqvaltest = freqval 
	if !isnothing(curx_test)
		freqvaltest = LogLinearModels.freqtab(curx_test,fillzeros=true)
	end
    return LogLinearModels.gsquare(freqvaltest, LogLinearModels.ipf(freqval,m0constr,maxit=100))
end

function makeconstraint(curx,evolution =false)
	varno=size(curx.data)[2]
	lastvisiblepos=varno - 1
	if evolution 
		return [collect(1:lastvisiblepos),collect(2:varno)]
	else
		return [collect(1:lastvisiblepos),varno]
	end
end


