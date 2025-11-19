#removes mean from nxN float matrix
function removeMean!(input)
    mn = mean(input, dims=1)
    (_, n) = size(input)
    for i in 1:n
        input[:, i] .= input[:, i] .- mn[i]
    end
    return mn
end

#Returns sphered data x. todo:replace with function from lib
function sphering!(x)
    (N, _) = size(x)
    F = svd(x' * x / N)
    S = F.U * diagm(1 ./ sqrt.(F.S)) * F.U'
    x .= x * S
    return S
end

#Adds means back to model centers
add_means_back!(myAmica::SingleModelAmica, removed_mean) = nothing

function add_means_back!(myAmica::MultiModelAmica, removed_mean)
    (_, _, m) = size(myAmica.models, 1)
    for h in 1:m
        myAmica.models[h].centers = myAmica.models[h].centers + removed_mean #add mean back to model centers
    end
end

"pre-calculate abs(y)^rho"
function update_y_rho!(myAmica::SingleModelAmica{T}) where T<:Real
    # addition compared to fortran: clamp y at 1e-16 to improve numerical stability
    myAmica.y_rho .= exp.(push_dimension(myAmica.shape) .* log.(abs.(myAmica.y)))
end