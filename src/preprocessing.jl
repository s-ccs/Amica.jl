#removes mean from nxN float matrix
@views function removeMean!(input)
    mn = mean(input, dims=1)
    (_, n) = size(input)
    for i in 1:n
        input[:, i] .= input[:, i] .- mn[i]
    end
    return mn
end

#Returns sphered data x. todo:replace with function from lib
@views function sphering!(x)
    (N, _) = size(x)
    F = svd(x' * x / N)

    T = eltype(x)
    mineig = T(1e-15)
    safe_s = max.(F.S, mineig)
    S = F.U * Diagonal(inv.(sqrt.(safe_s))) * F.U'

    x .= x * S
    return S
end
