#!/usr/bin/env julia

"""
Script to compare debug dump files before and after reparameterize
"""

function read_matrix(filename)
    lines = readlines(filename)
    matrix = []
    for line in lines
        if !isempty(strip(line))
            row = parse.(Float64, split(line))
            push!(matrix, row)
        end
    end
    return hcat(matrix...)'  # Transpose to get proper orientation
end

function compare_matrices(name, file1, file2)
    println("\n" * "="^80)
    println("Comparing: $name")
    println("="^80)

    if !isfile(file1)
        println("ERROR: File $file1 not found!")
        return
    end
    if !isfile(file2)
        println("ERROR: File $file2 not found!")
        return
    end

    mat1 = read_matrix(file1)
    mat2 = read_matrix(file2)'

    println("Matrix 1 size: ", size(mat1))
    println("Matrix 2 size: ", size(mat2))

    if size(mat1) != size(mat2)
        println("ERROR: Matrix sizes don't match!")
        return
    end

    diff = mat1 .- mat2
    abs_diff = abs.(diff)
    rel_diff = abs_diff ./ (abs.(mat2) .+ 1e-10)  # Add small epsilon to avoid division by zero

    println("\nAbsolute differences:")
    println("  Max: ", maximum(abs_diff))
    println("  Mean: ", sum(abs_diff) / length(abs_diff))
    println("  Min: ", minimum(abs_diff))

    println("\nRelative differences (%):")
    println("  Max: ", maximum(rel_diff) * 100)
    println("  Mean: ", sum(rel_diff) / length(rel_diff) * 100)

    # Find largest differences
    max_idx = argmax(abs_diff)
    println("\nLargest absolute difference at index $max_idx:")
    println("  Value 1: ", mat1[max_idx])
    println("  Value 2: ", mat2[max_idx])
    println("  Diff: ", diff[max_idx])

    # Show first few rows for inspection
    n_show = min(3, size(mat1, 1))
    m_show = min(3, size(mat1, 2))
    println("\nFirst $n_show×$m_show elements:")
    println("Matrix 1:")
    for i in 1:n_show
        println("  ", mat1[i, 1:m_show])
    end
    println("Matrix 2:")
    for i in 1:n_show
        println("  ", mat2[i, 1:m_show])
    end
    println("Differences:")
    for i in 1:n_show
        println("  ", diff[i, 1:m_show])
    end

    # Check if approximately equal
    if maximum(abs_diff) < 1e-6
        println("\n✓ Matrices are approximately equal (max diff < 1e-6)")
    elseif maximum(rel_diff) < 1e-6
        println("\n✓ Matrices are approximately equal (max relative diff < 1e-6)")
    else
        println("\n✗ Matrices have significant differences")
    end
end

# Main comparison
println("Debug Dump Comparison Tool")
println("="^80)


# Compare A
compare_matrices("A (mixing matrix)",
    "debug_A_before.txt",
    "debug_A_after.txt")

# Compare location
compare_matrices("Location (mu)",
    "debug_location_before.txt",
    "debug_location_after.txt")

# Compare scale
compare_matrices("Scale (sbeta)",
    "debug_scale_before.txt",
    "debug_scale_after.txt")

println("\n" * "="^80)
println("Comparison complete!")
println("="^80)
