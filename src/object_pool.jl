const Dims{N} = NTuple{N,Integer} where N

# Set to true to detect use-after-release bugs (fills released arrays with NaN)
const DEBUG_POOL = false

"""
    ObjectPool{T,A}

A pool of pre-allocated arrays that can be obtained and released for reuse.
Helps reduce memory allocations in performance-critical code. Supports any
array type (Vector, CuArray, etc.) and can reshape arrays on acquisition.

Note: This pool is NOT thread-safe. Use one pool per thread for multithreaded code.

# Type Parameters
- `T`: Element type of the arrays
- `A`: Array type (e.g., Vector{T}, CuArray{T})

# Fields
- `base_size::Int`: The maximum size of each array in the pool
- `max_arrays::Int`: Maximum number of arrays that can be obtained simultaneously
- `arrays::Vector{A}`: The pre-allocated flat arrays
- `available::Vector{Bool}`: Tracks which arrays are available for use
"""
mutable struct ObjectPool{T,A<:DenseArray{T,1}}
    base_size::Int
    max_arrays::Int
    arrays::Vector{A}
    available::Vector{Bool}

    function ObjectPool{T,A}(base_size::Int, max_arrays::Int) where {T,A<:DenseArray{T,1}}
        arrays = Vector[A(undef, base_size) for _ in 1:max_arrays]
        available = fill(true, max_arrays)
        new{T,A}(base_size, max_arrays, arrays, available)
    end
end


"""
    pool_acquire!(pool::ObjectPool{T,A}, dims::Dims{N}) -> AbstractArray{T,N}

Obtain an array from the pool with the specified dimensions. 
The pool stores flat arrays internally but returns a reshaped view.
Throws an error if all arrays are currently in use.

# Arguments
- `pool`: The ObjectPool to acquire from
- `dims`: A tuple of dimensions for the returned array (product must be ≤ base_size)

# Returns
- A reshaped view of type `AbstractArray{T,N}` with the specified dimensions

# Throws
- `ErrorException` if no arrays are available (all `max_arrays` are in use)
- `ArgumentError` if the requested size exceeds `base_size`

# Example
```julia
pool = ObjectPool(Float64, 1000, 10)
arr = pool_acquire!(pool, (10, 20))  # Returns a 10x20 view (200 elements)
matrix = pool_acquire!(pool, (50, 20))  # Returns a 50x20 view (1000 elements)
```
"""
@views function pool_acquire!(who, pool::ObjectPool{T,A}, dims::Dims{N}) where {T,A,N}
    total_size = prod(dims)
    if total_size > pool.base_size
        throw(ArgumentError("Requested size $total_size (dims=$dims) exceeds pool base_size $(pool.base_size)"))
    end

    for i in 1:pool.max_arrays
        if pool.available[i]
            pool.available[i] = false
            flat_view = pool.arrays[i][1:total_size]
            return reshape(flat_view, dims)
        end
    end
    error("ObjectPool exhausted: all $(pool.max_arrays) arrays are currently in use")
end

"""
    pool_release!(pool::ObjectPool{T,A}, arr::AbstractArray{T})

Release an array back to the pool, making it available for reuse.

# Arguments
- `pool`: The ObjectPool to release to
- `arr`: The array to release (must have been obtained from this pool)

# Throws
- `ErrorException` if the array does not belong to this pool
"""
function pool_release!(who, pool::ObjectPool{T,A}, arr::AbstractArray{T}) where {T,A}
    # Get the parent array (in case arr is a reshaped view)
    parent_arr = parent(arr)
    # If it's a view, get the underlying array
    while parent_arr !== arr && parent_arr isa SubArray
        parent_arr = parent(parent_arr)
    end

    for i in 1:pool.max_arrays
        if pool.arrays[i] === parent_arr || pointer(pool.arrays[i]) == pointer(arr)
            if pool.available[i]
                @warn "Array at index $i was already released"
            end

            # Poison the data to detect use-after-release
            if DEBUG_POOL && T <: AbstractFloat
                fill!(arr, T(NaN))
            end

            pool.available[i] = true
            return nothing
        end
    end
    error("Array does not belong to this ObjectPool")
end

"""
    available_count(pool::ObjectPool) -> Int

Returns the number of arrays currently available in the pool.
"""
function available_count(pool::ObjectPool)
    return count(pool.available)
end

"""
    in_use_count(pool::ObjectPool) -> Int

Returns the number of arrays currently in use from the pool.
"""
function in_use_count(pool::ObjectPool)
    return pool.max_arrays - available_count(pool)
end

"""
    reset!(pool::ObjectPool)

Reset the pool, marking all arrays as available.
"""
function reset!(pool::ObjectPool)
    fill!(pool.available, true)
    return nothing
end