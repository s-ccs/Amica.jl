
fp = fill(0.,1000000)

function test1!(fp)
fp .= Amica.ffun.(rand(1000000),2)
end

function test4!(fp)
    fp[:] .= Amica.ffun.(rand(1000000),2)
    end
function test2!(fp)
    fp = Amica.ffun.(rand(1000000),2)
end
function test3!(fp)
    fp[:] = Amica.ffun.(rand(1000000),2)
end


@benchmark test1!(fp)
@benchmark test2!(fp)
@benchmark test3!(fp)
@benchmark test4!(fp)