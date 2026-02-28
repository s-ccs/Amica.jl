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
