

"Calculate difference in loglikelihood between iterations"
function calculate_DLL!(dLL, myAmica::SingleModelAmica, iter)
    if iter > 1
        dLL[iter] = myAmica.LL[iter] - myAmica.LL[iter-1]
    end
end

"add a unit dimension in front to be able to e.g. broadcast a (1000, 12, 3) with a (12, 3) array, transforms a from (12, 3) to (1, 12, 3)"
function push_dimension(a::AbstractArray)
    reshape(a, 1, size(a)...)
end
